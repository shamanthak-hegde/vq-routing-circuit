"""Layer-wise linear probing for yes-bias direction (N-066).

For each decoder layer, trains a logistic regression on the prompt_last residual
vectors to predict whether the model says "yes".  If yes-bias is committed early
(as the VQ routing hypothesis predicts), VILA-U should have high probe accuracy
from L0; LLaVA should have low accuracy until L8+.

Usage
-----
    # Step 1: collect activations (GPU required)
    source activate vila-u
    python -m probe.tracing.linear_probe collect \\
        --backend vilau --model_path mit-han-lab/vila-u-7b-256 \\
        --n_records 200 --out results/residuals_vilau.pt

    source activate sae
    python -m probe.tracing.linear_probe collect \\
        --backend llava --model_path liuhaotian/llava-v1.6-vicuna-7b \\
        --n_records 200 --out results/residuals_llava.pt

    # Step 2: train probes + plot (CPU, no GPU needed)
    python -m probe.tracing.linear_probe train \\
        --residuals results/residuals_vilau.pt \\
                    results/residuals_llava.pt \\
        --names vilau llava \\
        --out results/linear_probe.json \\
        --figure figures/linear_probe.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


# ── Activation collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_residuals(
    hm,
    records,
    n_records: int = 200,
    seed: int = 0,
) -> dict:
    """Collect prompt_last residuals and yes-prediction labels.

    Returns
    -------
    dict with:
      'residuals': (n_records, n_layers, hidden) float32 tensor (CPU)
      'labels':    (n_records,) int tensor — 1 if model predicts yes, else 0
      'correct':   (n_records,) int tensor — 1 if prediction matches GT
      'gt_yes':    (n_records,) int tensor — 1 if GT answer is yes
    """
    import random
    from PIL import Image as _Image

    pope = [r for r in records if r.corruption_mode == "gaussian_noise"]
    rng = random.Random(seed)
    sample = rng.sample(pope, min(n_records, len(pope)))

    yes_id = hm.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = hm.tokenizer.encode("no", add_special_tokens=False)[0]

    all_residuals = []
    labels = []
    correct_flags = []
    gt_yes_flags = []

    for i, rec in enumerate(sample, 1):
        img = _Image.open(rec.image_path).convert("RGB")
        cap = hm.run_prefill(img, rec.question)
        prompt_last = cap.token_index.prompt_last
        # (n_layers, hidden) slice at prompt_last
        per_layer = cap.residual[:, prompt_last, :].float().cpu()
        all_residuals.append(per_layer)

        last_logits = cap.logits[prompt_last].float()
        pred_yes = int(last_logits[yes_id] > last_logits[no_id])
        labels.append(pred_yes)
        correct_flags.append(int(pred_yes == (rec.answer == "yes")))
        gt_yes_flags.append(int(rec.answer == "yes"))

        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    return {
        "residuals": torch.stack(all_residuals),   # (N, L, H)
        "labels": torch.tensor(labels),             # (N,)
        "correct": torch.tensor(correct_flags),     # (N,)
        "gt_yes": torch.tensor(gt_yes_flags),       # (N,)
        "n_records": len(sample),
        "n_layers": all_residuals[0].shape[0],
        "hidden": all_residuals[0].shape[1],
    }


# ── Per-layer probe training ──────────────────────────────────────────────────

def train_probes(
    data: dict,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """Fit logistic regression at each layer; return per-layer accuracy.

    Parameters
    ----------
    data           : output of collect_residuals()
    test_fraction  : held-out fraction for test accuracy
    seed           : RNG for train/test split

    Returns
    -------
    list of dicts: [{'layer': l, 'train_acc': ..., 'test_acc': ...,
                     'weight_norm': ...}, ...]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = data["residuals"].numpy()   # (N, L, H)
    y = data["labels"].numpy()      # (N,)
    n, n_layers, h = X.shape

    results = []
    for l in range(n_layers):
        Xl = X[:, l, :]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xl, y, test_size=test_fraction, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        train_acc = clf.score(X_tr, y_tr)
        test_acc = clf.score(X_te, y_te)
        w_norm = float((clf.coef_**2).sum()**0.5)
        results.append({
            "layer": l,
            "train_acc": round(float(train_acc), 4),
            "test_acc": round(float(test_acc), 4),
            "weight_norm": round(w_norm, 4),
        })

    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_probe_curves(
    probe_results: dict[str, list[dict]],
    out_path: str | Path,
) -> None:
    """Plot per-layer test accuracy curves for all models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"vilau": "#e74c3c", "llava": "#3498db", "unitok": "#2ecc71"}
    linestyles = {"vilau": "-", "llava": "--", "unitok": ":"}

    for name, probes in probe_results.items():
        layers = [p["layer"] for p in probes]
        accs = [p["test_acc"] for p in probes]
        color = colors.get(name, None)
        ls = linestyles.get(name, "-")
        ax.plot(layers, accs, label=name, color=color, linestyle=ls, linewidth=2)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Test accuracy (predicts-yes probe)")
    ax.set_title("Layer-wise linear probe: when is yes-bias committed?")
    ax.legend()
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_backend(args):
    if args.backend == "vilau":
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "vila-u"))
        if root not in sys.path:
            sys.path.insert(0, root)
        from vila_u.model.builder import load_pretrained_model
        from probe.hooks import VilaUHookManager
        tokenizer, model, ip, _ = load_pretrained_model(args.model_path, attn_implementation="eager")
        model.eval()
        return VilaUHookManager(model, tokenizer, ip)

    if args.backend == "llava":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LLaVA"))
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from probe.hooks import LlavaHookManager
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, ip, _ = load_pretrained_model(
            args.model_path, model_base=None, model_name=model_name)
        model.eval()
        return LlavaHookManager(model, tokenizer, ip)

    raise ValueError(f"Unknown backend {args.backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise linear probing for yes-bias direction (N-066)"
    )
    subp = parser.add_subparsers(dest="mode", required=True)

    # --- collect mode (GPU)
    col_p = subp.add_parser("collect",
                             help="Collect prompt_last residuals (GPU required)")
    col_p.add_argument("--backend", required=True, choices=["vilau", "llava", "unitok"])
    col_p.add_argument("--model_path", required=True)
    col_p.add_argument("--tokenizer_path", default=None)
    col_p.add_argument("--n_records", type=int, default=200)
    col_p.add_argument("--seed", type=int, default=0)
    col_p.add_argument("--out", required=True, help="Output .pt file")

    # --- train mode (CPU)
    tr_p = subp.add_parser("train",
                            help="Train probes + generate figure (CPU)")
    tr_p.add_argument("--residuals", nargs="+", required=True,
                      help="Paths to .pt files from collect mode")
    tr_p.add_argument("--names", nargs="+", required=True,
                      help="Model names (same order as --residuals)")
    tr_p.add_argument("--test_fraction", type=float, default=0.3)
    tr_p.add_argument("--out", required=True, help="Output JSON path")
    tr_p.add_argument("--figure", default=None, help="Optional figure path (.png)")

    args = parser.parse_args()

    if args.mode == "collect":
        hm = _load_backend(args)
        from probe import load_cache
        records, _, _ = load_cache()
        print(f"Collecting residuals for {args.backend} ({args.n_records} records) …")
        data = collect_residuals(hm, records, n_records=args.n_records, seed=args.seed)
        data["backend"] = args.backend
        data["model_path"] = args.model_path
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)
        print(
            f"Saved residuals: {list(data['residuals'].shape)}  "
            f"yes_rate={data['labels'].float().mean().item():.2%}  "
            f"→ {out}"
        )

    elif args.mode == "train":
        if len(args.residuals) != len(args.names):
            parser.error("--residuals and --names must have the same length")

        probe_results: dict[str, list[dict]] = {}
        for name, path in zip(args.names, args.residuals):
            data = torch.load(path, map_location="cpu")
            print(f"Training probes for {name} ({data['n_records']} records, "
                  f"{data['n_layers']} layers) …")
            probe_results[name] = train_probes(data, test_fraction=args.test_fraction)
            # Summary: layer 0 and half
            n_layers = data["n_layers"]
            l0 = probe_results[name][0]["test_acc"]
            lhalf = probe_results[name][n_layers // 2]["test_acc"]
            llast = probe_results[name][-1]["test_acc"]
            print(f"  {name}: L0={l0:.3f}, L{n_layers//2}={lhalf:.3f}, L{n_layers-1}={llast:.3f}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "probe_results": probe_results,
            "names": args.names,
            "test_fraction": args.test_fraction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        print(f"Results: {out}")

        if args.figure:
            Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
            plot_probe_curves(probe_results, args.figure)


if __name__ == "__main__":
    main()
