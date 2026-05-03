"""Two-panel publication figure: row-normalised confusion matrix + per-class
one-vs-rest ROC curves. Generated for both in-domain (Kermany test) and
zero-shot (OCTID) for the headline model.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc

from data import get_kermany_loaders, get_octid_loader, KERMANY_NAMES
from model import OCTSSRet


def collect(model, loader, device):
    ys, probs = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            out = model(x)
            ys.append(batch["label"].numpy())
            probs.append(F.softmax(out["logits_cls"], dim=-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


def plot_panel(y, p, names, title, save_path):
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    cm_n = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # (a) Row-normalised confusion matrix
    ax = axes[0]
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
    ax.set_title("(a) Row-normalised confusion matrix", loc="left", fontweight="bold", fontsize=11)
    for i in range(len(names)):
        for j in range(len(names)):
            txt_color = "white" if cm_n[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)",
                    ha="center", va="center", color=txt_color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised count")

    # (b) Per-class one-vs-rest ROC
    ax = axes[1]
    for k, name in enumerate(names):
        present = (y == k).astype(int)
        if present.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(present, p[:, k])
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.005])
    ax.set_title("(b) Per-class one-vs-rest ROC", loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    fig.savefig(str(save_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[saved] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=str(ROOT / "paper" / "paper_latex" / "figures"))
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    model = OCTSSRet(num_classes=4, backbone=sa.get("backbone","resnet18"),
                     use_ssm=sa.get("use_ssm", False),
                     ssm_layers=sa.get("ssm_layers", 2),
                     ssm_state=sa.get("ssm_state", 16),
                     pretrained=False).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()

    # In-domain Kermany test
    _, _, kerm_test = get_kermany_loaders(batch_size=32, image_size=sa["image_size"], num_workers=2)
    yk, pk = collect(model, kerm_test, device)
    plot_panel(yk, pk, KERMANY_NAMES,
               title=f"In-domain: Kermany 2017 test split (n={len(yk)})",
               save_path=Path(args.out_dir) / "fig_confusion_kermany.pdf")

    # Zero-shot OCTID
    octid = get_octid_loader(batch_size=16, image_size=sa["image_size"], num_workers=2)
    yo, po = collect(model, octid, device)
    NAMES_OCTID = ["CNV (MH)", "DME (DR)", "DRUSEN (AMD)", "NORMAL"]
    plot_panel(yo, po, NAMES_OCTID,
               title=f"Zero-shot cross-cohort transfer: OCTID (n={len(yo)})",
               save_path=Path(args.out_dir) / "fig_confusion_octid.pdf")


if __name__ == "__main__":
    main()
