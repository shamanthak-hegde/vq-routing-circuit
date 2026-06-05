"""
VQ vs continuous visual embedding space visualization (Plan Rev. 4, Experiment A).

Compares VILA-U's projected visual tokens (from ~82 active codebook entries) with
LLaVA's continuous CLIP+MLP tokens, showing WHY the L0 circuit forms: VQ tokens
cluster in a low-dimensional discrete subspace, making them trivially routable as a
unit by attention; continuous tokens are spread across a high-dimensional manifold.

Two subcommands:

  collect  (GPU) — run prefill on N POPE records, extract cap.residual[0, vlo:vhi, :]
                   (projected visual tokens before any transformer layer), save to .pt

  plot     (CPU) — load two .pt files (vilau + llava), produce:
                   Panel 1: 2D PCA scatter (colored by record index)
                   Panel 2: Singular value decay curves (effective rank)
                   Panel 3: Cross-record nearest-neighbour cosine distance histogram

Usage
-----
    # Collect (run once per model in its own env)
    source activate vila-u
    python -m probe.tracing.embedding_viz collect \\
        --backend vilau \\
        --model_path mit-han-lab/vila-u-7b-256 \\
        --n_records 50 \\
        --out results/vis_embeddings_vilau.pt

    source activate sae
    python -m probe.tracing.embedding_viz collect \\
        --backend llava \\
        --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --n_records 50 \\
        --out results/vis_embeddings_llava.pt

    # Plot (CPU, any env with matplotlib + sklearn)
    python -m probe.tracing.embedding_viz plot \\
        --vilau results/vis_embeddings_vilau.pt \\
        --llava results/vis_embeddings_llava.pt \\
        --out figures/mechanism/embedding_viz.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image


# ── collect ───────────────────────────────────────────────────────────────────

def _load_hm(backend: str, model_path: str):
    if backend == "vilau":
        _root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "vila-u")
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, attn_implementation="eager"
        )
        model.eval()
        return VilaUHookManager(model, tokenizer, image_processor)

    if backend == "llava":
        _root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA")
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, model_base=None, model_name=model_name,
            attn_implementation="sdpa",
        )
        model.eval()
        return LlavaHookManager(model, tokenizer, image_processor)

    raise ValueError(f"Unknown backend {backend!r}; supported: vilau, llava")


@torch.no_grad()
def collect(args: argparse.Namespace) -> None:
    from probe import load_cache

    hm = _load_hm(args.backend, args.model_path)
    records, _, _ = load_cache()
    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    sample = pope[:args.n_records]

    all_embeddings: list[torch.Tensor] = []   # (n_visual, hidden) per record
    record_ids: list[str] = []
    n_visual_list: list[int] = []

    print(f"[embedding_viz] Collecting {len(sample)} records ({args.backend}) ...")
    for i, rec in enumerate(sample, 1):
        img = Image.open(rec.image_path).convert("RGB")
        cap = hm.run_prefill(img, rec.question)

        vlo, vhi = cap.token_index.visual_range
        # residual[0] = inputs to transformer layer 0 = post-projector embeddings
        vis_emb = cap.residual[0, vlo:vhi, :].float().cpu()  # (n_visual, hidden)
        all_embeddings.append(vis_emb)
        record_ids.append(rec.id)
        n_visual_list.append(int(vhi - vlo))

        if i % 10 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backend": args.backend,
        "model_path": args.model_path,
        "n_records": len(all_embeddings),
        "embeddings": all_embeddings,     # list of (n_visual, hidden) tensors
        "record_ids": record_ids,
        "n_visual_list": n_visual_list,
    }, str(out))
    print(f"Saved {len(all_embeddings)} records → {out}")


# ── analysis helpers ──────────────────────────────────────────────────────────

def _effective_rank(mat: torch.Tensor) -> float:
    """Effective rank via singular value entropy: exp(H(s / sum(s)))."""
    _, sv, _ = torch.linalg.svd(mat, full_matrices=False)
    sv = sv.float()
    sv = sv[sv > 1e-9]
    p = sv / sv.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return math.exp(entropy)


def _cross_record_nn_cosine_distances(
    embeddings: list[torch.Tensor],
    n_pairs: int = 200,
    seed: int = 0,
) -> list[float]:
    """For each sampled pair of records (i, j), find the minimum cosine distance
    between any token in record i and any token in record j.  Returns a list of
    per-pair minimum distances.  Small values → tokens are nearly identical across
    records (VQ clustering); large values → unique per-image tokens (continuous).
    """
    import random
    rng = random.Random(seed)
    n = len(embeddings)
    pairs = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(n_pairs)]
    pairs = [(i, j) for i, j in pairs if i != j]

    dists: list[float] = []
    for i, j in pairs:
        a = torch.nn.functional.normalize(embeddings[i], dim=-1)  # (Na, hidden)
        b = torch.nn.functional.normalize(embeddings[j], dim=-1)  # (Nb, hidden)
        sim = a @ b.T  # (Na, Nb)
        # nearest-neighbour: for each token in a, find max cosine sim in b
        nn_sim = sim.max(dim=-1).values  # (Na,)
        nn_dist = 1.0 - nn_sim.mean().item()  # average NN cosine distance
        dists.append(float(nn_dist))
    return dists


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA

    vilau_data = torch.load(args.vilau, map_location="cpu")
    llava_data = torch.load(args.llava, map_location="cpu")

    for label, data in [("VILA-U", vilau_data), ("LLaVA", llava_data)]:
        print(f"\n{label}: {data['n_records']} records, "
              f"n_visual={data['n_visual_list'][0]} (first record)")

    # Stack all visual tokens into one matrix per model
    vilau_mat = torch.cat(vilau_data["embeddings"], dim=0).float()  # (N_vilau, H)
    llava_mat = torch.cat(llava_data["embeddings"], dim=0).float()  # (N_llava, H)

    n_vilau_records = vilau_data["n_records"]
    n_llava_records = llava_data["n_records"]
    n_tok_vilau = vilau_mat.shape[0] // n_vilau_records  # tokens per record
    n_tok_llava = llava_mat.shape[0] // n_llava_records

    # Effective rank
    er_vilau = _effective_rank(vilau_mat)
    er_llava = _effective_rank(llava_mat)
    print(f"Effective rank — VILA-U: {er_vilau:.1f}, LLaVA: {er_llava:.1f}")

    # Cross-record NN distances
    nn_vilau = _cross_record_nn_cosine_distances(vilau_data["embeddings"])
    nn_llava = _cross_record_nn_cosine_distances(llava_data["embeddings"])
    print(f"Mean cross-record NN cosine dist — VILA-U: {np.mean(nn_vilau):.4f}, "
          f"LLaVA: {np.mean(nn_llava):.4f}")

    # PCA (fit on combined, project separately)
    combined = torch.cat([vilau_mat, llava_mat], dim=0).numpy()
    pca = PCA(n_components=2, random_state=0)
    pca.fit(combined)
    pv2 = pca.explained_variance_ratio_[:2]

    pc_vilau = pca.transform(vilau_mat.numpy())  # (N_vilau, 2)
    pc_llava = pca.transform(llava_mat.numpy())  # (N_llava, 2)

    # Singular value decay (normalised)
    def _sv_decay(mat, k=64):
        _, sv, _ = torch.linalg.svd(mat, full_matrices=False)
        sv = sv[:k].float()
        return (sv / sv[0]).tolist()

    sv_vilau = _sv_decay(vilau_mat)
    sv_llava = _sv_decay(llava_mat)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cmap = plt.get_cmap("tab20")

    # Panel 1: PCA scatter — VILA-U
    ax = axes[0]
    for r in range(min(n_vilau_records, 20)):
        sl = slice(r * n_tok_vilau, (r + 1) * n_tok_vilau)
        ax.scatter(pc_vilau[sl, 0], pc_vilau[sl, 1],
                   s=4, alpha=0.6, color=cmap(r % 20), rasterized=True)
    ax.set_title(f"VILA-U visual tokens (PCA)\neff. rank={er_vilau:.1f}", fontsize=11)
    ax.set_xlabel(f"PC1 ({pv2[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pv2[1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    # Panel 2: PCA scatter — LLaVA
    ax = axes[1]
    n_llava_plot = min(n_llava_records, 20)
    for r in range(n_llava_plot):
        sl = slice(r * n_tok_llava, (r + 1) * n_tok_llava)
        ax.scatter(pc_llava[sl, 0], pc_llava[sl, 1],
                   s=4, alpha=0.6, color=cmap(r % 20), rasterized=True)
    ax.set_title(f"LLaVA visual tokens (PCA)\neff. rank={er_llava:.1f}", fontsize=11)
    ax.set_xlabel(f"PC1 ({pv2[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pv2[1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    # Panel 3: singular value decay + NN distance inset
    ax = axes[2]
    ks = list(range(1, len(sv_vilau) + 1))
    ax.plot(ks, sv_vilau, color="#e74c3c", linewidth=2, label=f"VILA-U (eff.rank={er_vilau:.0f})")
    ax.plot(ks, sv_llava, color="#3498db", linewidth=2, label=f"LLaVA (eff.rank={er_llava:.0f})")
    ax.set_title("Singular value decay\n(normalised to σ₁=1)", fontsize=11)
    ax.set_xlabel("Singular value rank")
    ax.set_ylabel("Normalised σ")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Inset: NN distance histogram
    ax_in = ax.inset_axes([0.38, 0.45, 0.58, 0.50])
    ax_in.hist(nn_vilau, bins=15, alpha=0.7, color="#e74c3c", density=True,
               label=f"VILA-U μ={np.mean(nn_vilau):.3f}")
    ax_in.hist(nn_llava, bins=15, alpha=0.7, color="#3498db", density=True,
               label=f"LLaVA μ={np.mean(nn_llava):.3f}")
    ax_in.set_xlabel("Cross-record NN dist", fontsize=8)
    ax_in.set_ylabel("Density", fontsize=8)
    ax_in.tick_params(labelsize=7)
    ax_in.legend(fontsize=7)
    ax_in.set_title("Cross-record NN", fontsize=8)

    fig.suptitle(
        "Projected visual token geometry: VQ (VILA-U) vs continuous (LLaVA)\n"
        f"VILA-U clusters → low eff. rank → routable as unit by L0 attention",
        fontsize=10, y=1.01
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out}")

    # Print summary stats
    print("\n=== Embedding geometry summary ===")
    print(f"  VILA-U effective rank: {er_vilau:.1f}  (low → discrete clustering)")
    print(f"  LLaVA  effective rank: {er_llava:.1f}  (high → continuous distribution)")
    print(f"  VILA-U mean cross-record NN dist: {np.mean(nn_vilau):.4f}")
    print(f"  LLaVA  mean cross-record NN dist: {np.mean(nn_llava):.4f}")
    print(f"  Ratio (LLaVA/VILA-U NN dist): {np.mean(nn_llava)/max(np.mean(nn_vilau),1e-6):.1f}×")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQ vs continuous visual token geometry (embedding_viz)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Collect visual embeddings (GPU)")
    p_collect.add_argument("--backend", required=True, choices=["vilau", "llava"])
    p_collect.add_argument("--model_path", required=True)
    p_collect.add_argument("--n_records", type=int, default=50)
    p_collect.add_argument("--out", required=True)

    p_plot = sub.add_parser("plot", help="Plot geometry (CPU)")
    p_plot.add_argument("--vilau", required=True, help="VILA-U .pt file from collect")
    p_plot.add_argument("--llava", required=True, help="LLaVA .pt file from collect")
    p_plot.add_argument("--out", required=True, help="Output PNG path")

    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
