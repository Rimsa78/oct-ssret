"""Grad-CAM on the OCT-SSRet final residual stage, one panel per Kermany class."""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from data import get_kermany_loaders, KERMANY_NAMES
from model import OCTSSRet


def grad_cam(model, img, target_class):
    feats, grads = [], []
    last_stage = model.backbone[-1] if model.backbone is not None else model._vit.encoder.layers[-1]
    h_f = last_stage.register_forward_hook(lambda m, i, o: feats.append(o))
    h_b = last_stage.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    out = model(img)
    sel = out["logits_cls"][torch.arange(img.shape[0]), target_class].sum()
    model.zero_grad(); sel.backward()
    h_f.remove(); h_b.remove()
    f = feats[0]
    g = grads[0]
    if f.dim() == 4:
        f = f.permute(0, 2, 3, 1); g = g.permute(0, 2, 3, 1)
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * f).sum(dim=-1, keepdim=True))
    cam = cam.permute(0, 3, 1, 2)
    cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[:, 0]
    cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (cam.amax(dim=(1, 2), keepdim=True) - cam.amin(dim=(1, 2), keepdim=True) + 1e-6)
    return cam.detach().cpu().numpy(), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "paper" / "paper_latex" / "figures" / "fig_gradcam.pdf"))
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    model = OCTSSRet(num_classes=4, backbone=sa.get("backbone","resnet18"),
                     use_ssm=sa.get("use_ssm", False),
                     ssm_layers=sa.get("ssm_layers", 2),
                     ssm_state=sa.get("ssm_state", 16)).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()

    _, _, test_loader = get_kermany_loaders(batch_size=16, image_size=sa["image_size"], num_workers=2)
    examples = {}
    for batch in test_loader:
        for i, y in enumerate(batch["label"].tolist()):
            if y not in examples:
                examples[y] = (batch["image"][i:i+1].clone(), int(y))
        if len(examples) == 4: break

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10, "axes.labelsize": 9})
    cmap = "inferno"; norm = Normalize(vmin=0.0, vmax=1.0)
    n = sum(1 for k in range(4) if k in examples)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 3.0 * n))
    if n == 1: axes = axes[None, :]

    for row, k in enumerate(sorted(examples)):
        img, y = examples[k]; img = img.to(device)
        cam, out = grad_cam(model, img, torch.tensor([y], device=device))
        prob = F.softmax(out["logits_cls"], dim=-1).detach().cpu().numpy()[0]
        pred = int(np.argmax(prob))
        rgb = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cam0 = cam[0]
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"(a) OCT B-scan  --  GT: {KERMANY_NAMES[y]}", loc="left", fontsize=10, fontweight="bold")
        axes[row, 1].imshow(cam0, cmap=cmap, norm=norm, interpolation="bilinear")
        mark = "OK" if pred == y else "X"
        axes[row, 1].set_title(f"(b) Grad-CAM (inferno) -- pred {KERMANY_NAMES[pred]} [{mark}], p={prob[y]:.2f}",
                               loc="left", fontsize=10, fontweight="bold")
        axes[row, 2].imshow(rgb)
        axes[row, 2].imshow(cam0, cmap=cmap, alpha=0.45, norm=norm, interpolation="bilinear")
        axes[row, 2].set_title("(c) Overlay", loc="left", fontsize=10, fontweight="bold")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax); cb.set_label("Grad-CAM activation", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.suptitle("OCT-SSRet Grad-CAM on the final ResNet-18 residual stage, one row per Kermany class", fontsize=11, y=0.995)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.96, bottom=0.02, hspace=0.20, wspace=0.05)
    plt.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.savefig(args.out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
