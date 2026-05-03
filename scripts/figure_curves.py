"""4-panel training-dynamics figure: train loss, val ACC, val QWK, val AUC
across epochs for selected models. Uses each model's history.json.
"""
import json, argparse
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["oct_baseline_s0", "oct_ssm_s0", "oct_swin_s0", "oct_swin_ssm_s0"])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default=str(ROOT / "paper" / "paper_latex" / "figures" / "fig_curves.pdf"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    runs = args.runs
    labels = args.labels or [r.replace("oct_", "").replace("_s0", "") for r in runs]
    assert len(labels) == len(runs)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    titles = ["(a) Training loss",
              "(b) Validation accuracy",
              "(c) Validation QWK",
              "(d) Validation macro AUC"]
    keys = ["train_loss", ("val", "acc"), ("val", "qwk"), ("val", "macro_auc")]

    for tag, label in zip(runs, labels):
        hp = ROOT / "artifacts" / tag / "history.json"
        if not hp.exists():
            print(f"[skip] {tag}: no history")
            continue
        h = json.load(open(hp))
        epochs = [e["epoch"] for e in h]
        for j, k in enumerate(keys):
            if isinstance(k, tuple):
                vals = [e[k[0]][k[1]] for e in h]
            else:
                vals = [e[k] for e in h]
            axes[j].plot(epochs, vals, marker="o", lw=2, label=label, alpha=0.85)

    for j in range(4):
        axes[j].set_xlabel("epoch")
        axes[j].set_title(titles[j], loc="left", fontweight="bold")
        axes[j].grid(alpha=0.3)
        if j > 0:
            axes[j].legend(fontsize=8, frameon=True)

    fig.suptitle("Training dynamics on Kermany 2017 (10-epoch cosine schedule)", fontsize=12, y=1.05)
    plt.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.savefig(args.out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
