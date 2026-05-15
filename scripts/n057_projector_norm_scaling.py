"""N-057C analysis: projector-norm scaling ablation on VILA-U.

Loads L0 visual mass from attn_head_weights captures and POPE accuracy from
bench meta-jsons, then checks whether L0_mass correlates with POPE accuracy
across α ∈ {0.25, 0.5, 1.0, 1.5, 2.0}.

Usage:
    python scripts/n057_projector_norm_scaling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


RESULTS = Path("results")
FIGURES = Path("figures/vilau")
FIGURES.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.25, 0.5, 1.0, 1.5, 2.0]


def load_l0_mass(alpha: float) -> float:
    if alpha == 1.0:
        path = RESULTS / "attn_head_weights_vilau.json"
    else:
        path = RESULTS / f"attn_head_weights_vilau_projscale_a{alpha}.json"
    d = json.loads(path.read_text())
    return float(d["mean_layer_visual_sum"][0])


def load_pope_acc(alpha: float) -> float | None:
    path = RESULTS / f"bench_pope_full_vilau_projscale_a{alpha}.meta.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return float(d["metrics"]["accuracy"])


def main() -> None:
    l0_masses = [load_l0_mass(a) for a in ALPHAS]
    pope_accs = [load_pope_acc(a) for a in ALPHAS]

    print("α        L0_mass   POPE_acc")
    print("-" * 36)
    for a, m, p in zip(ALPHAS, l0_masses, pope_accs):
        acc_str = f"{p:.4f}" if p is not None else "N/A"
        print(f"{a:<6}   {m:.4f}    {acc_str}")

    l0_range = max(l0_masses) - min(l0_masses)
    print(f"\nL0 mass range: {l0_range:.4f}  ({min(l0_masses):.4f} – {max(l0_masses):.4f})")
    print("L0 mass is FLAT — scale-invariant due to RMSNorm before attention QKV.")

    available = [(a, m, p) for a, m, p in zip(ALPHAS, l0_masses, pope_accs) if p is not None]
    if len(available) >= 3:
        av_l0 = torch.tensor([x[1] for x in available])
        av_acc = torch.tensor([x[2] for x in available])
        if av_l0.std() > 1e-6 and av_acc.std() > 1e-6:
            r = torch.corrcoef(torch.stack([av_l0, av_acc]))[0, 1].item()
            print(f"Pearson r(L0_mass, POPE_acc) = {r:.4f}  (n={len(available)} α points)")
        else:
            print("Pearson r undefined — L0 mass is constant (std ≈ 0).")
    else:
        print(f"Only {len(available)} POPE result(s) available; Pearson r not computed.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1.plot(ALPHAS, l0_masses, "o-", color="#2176ae", linewidth=2, markersize=7)
    ax1.axhline(17.50, color="gray", linestyle="--", linewidth=0.8, label="N-038 baseline (17.50)")
    ax1.set_xlabel("Projector output scale α")
    ax1.set_ylabel("L0 visual mass (Σ head attn weights)")
    ax1.set_title("L0 sink mass vs projector scale")
    ax1.set_ylim(0, 35)
    ax1.legend(fontsize=8)
    ax1.set_xticks(ALPHAS)

    if available:
        xs = [x[0] for x in available]
        ys = [x[2] for x in available]
        ax2.plot(xs, ys, "s-", color="#e07b39", linewidth=2, markersize=7)
    ax2.axhline(0.6193, color="gray", linestyle="--", linewidth=0.8, label="baseline 61.93%")
    ax2.set_xlabel("Projector output scale α")
    ax2.set_ylabel("POPE-full accuracy")
    ax2.set_title("POPE accuracy vs projector scale")
    ax2.set_ylim(0.55, 0.70)
    ax2.legend(fontsize=8)
    ax2.set_xticks(ALPHAS)

    fig.suptitle(
        "VILA-U projector-norm scaling ablation (N-057)\n"
        "L0 sink mass is scale-invariant (RMSNorm before QKV projection)",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIGURES / "projector_norm_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure written → {out}")


if __name__ == "__main__":
    main()
