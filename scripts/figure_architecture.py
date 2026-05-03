"""Matplotlib backup architecture diagram for OCT-SSRet (Swin-V2-T + bidir SSM)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "paper_latex" / "figures" / "fig_architecture.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(0, 15); ax.set_ylim(0, 6.5); ax.axis("off")

C = {"input": "#dbeafe", "swin": "#dbeafe", "ssm": "#86efac", "pool": "#cbd5e1",
     "head": "#dbeafe", "loss": "#fecaca", "output": "#fcd34d", "note": "#fde6c8"}


def block(x, y, w, h, label, sub="", color="#e5e7eb", fontsize=10, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                linewidth=1.0, edgecolor="#475569", facecolor=color, zorder=3))
    ax.text(x + w/2, y + h/2 + (0.13 if sub else 0), label,
            ha="center", va="center", fontsize=fontsize, weight=("bold" if bold else "normal"), zorder=4)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.20, sub, ha="center", va="center",
                fontsize=fontsize - 2, color="#475569", zorder=4)


def arrow(x1, y1, x2, y2, color="#475569", lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 color=color, linewidth=lw, linestyle=ls, mutation_scale=12, zorder=2))


# Title
ax.text(7.5, 6.1,
        "OCT-SSRet: Hybrid Swin-Transformer + Mamba-Style Selective State-Space Bottleneck for OCT Disease Classification",
        ha="center", fontsize=12, weight="bold")

# Band 1: input
block(0.2, 3.0, 1.5, 1.2, "OCT B-scan", "x: B×3×256×256\nFOV+CLAHE", C["input"], fontsize=9)

# Band 2: Swin backbone (stacked stages)
block(2.1, 2.7, 2.0, 1.7, "ResNet-18 Encoder", "ImageNet1K pretrained\n11.18M params", C["swin"], fontsize=10)
# stage stripes
for i, s in enumerate(["stem: /2, 64ch", "layer1-2: /4-/8, 64-128ch", "layer3: /16, 256ch", "layer4: /32, 512ch"]):
    ax.text(3.1, 4.0 - 0.30*i, s, ha="center", va="center",
            fontsize=8, color="#1f4e79")
arrow(1.7, 3.5, 2.1, 3.5)

# tensor between Swin and SSM
ax.text(4.3, 3.5, "feat_map\nB×512×8×8\n→ tokens\nB×64×768",
        ha="center", va="center", fontsize=8, color="#475569",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1"))
arrow(4.1, 3.3, 5.7, 3.3)

# Band 3: SSM bottleneck (the headline)
block(5.7, 2.3, 3.4, 2.4,
      "Mamba-style Bidirectional\nSelective SSM",
      "pure-PyTorch, 2 layers\n~8.6M params  (NEW)\n→ forward scan\n← reverse scan",
      C["ssm"], fontsize=10)
# tiny equations in SSM block
ax.text(7.4, 2.7, r"$\Delta, B, C \leftarrow f(\mathrm{input})$" + "\n"
                  r"$h_t = \exp(\Delta A)\,h_{t-1} + \Delta B \cdot u_t$" + "\n"
                  r"$y_t = C\,h_t + D \cdot u_t$",
        ha="center", va="bottom", fontsize=7.5, color="#1f4e29")

# tensor between SSM and pool
arrow(9.1, 3.3, 10.0, 3.3)
ax.text(9.6, 3.55, "tokens_ssm\nB×64×768", ha="center", va="bottom",
        fontsize=7.5, color="#475569")

# Band 4: pool + head
block(10.0, 3.6, 1.6, 0.8, "Mean-Pool +\nLayerNorm", "B×768", C["pool"], fontsize=9)
block(10.0, 2.6, 1.6, 0.8, "Dropout(0.3) +\nLinear(768→4)", "softmax", C["head"], fontsize=9)
arrow(10.8, 3.6, 10.8, 3.4)

# Band 5: loss + output
block(12.0, 3.8, 2.6, 0.8, r"$L_{\text{focal}}$ class-bal CE",
      "β=0.999, γ=2.0", C["loss"], fontsize=9)
block(12.0, 2.6, 2.6, 0.8, "Predicted class",
      "{CNV, DME, DRUSEN, NORMAL}", C["output"], fontsize=9)
arrow(11.6, 3.0, 12.0, 3.0)
arrow(11.6, 4.2, 12.0, 4.2)

# Annotation strip
block(0.3, 1.0, 14.3, 0.8,
      "Empirical operating profile", " "
      "Essentially neutral in-domain on Kermany 2017 (matches Swin-V2-T baseline). "
      "Improves zero-shot cross-cohort transfer to OCTID at zero retraining cost. "
      "Pure-PyTorch SSM (no custom CUDA), portable across GPUs incl. Blackwell sm_120.",
      C["note"], fontsize=9, bold=False)

# Bottom legend
ax.text(0.3, 0.4, "Solid arrow: forward + backward gradient flow.   "
                  "Green box (SSM) = headline architectural contribution.",
        fontsize=9, color="#475569")

plt.tight_layout()
plt.savefig(OUT, format="pdf", bbox_inches="tight")
plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"[saved] {OUT}")
