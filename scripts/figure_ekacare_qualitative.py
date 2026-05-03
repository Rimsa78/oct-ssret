"""Qualitative panel on the ekacare paired-modality cohort (n=50, paired
fundus + OCT B-scan + ILM + RPE annotations). Used as a deployment-style
qualitative figure: shows that the OCT-SSRet attention overlays plausibly
on retinal layers even on a held-out cohort with a different label space
(glaucoma binary instead of Kermany 4-class).
"""
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

from data import get_ekacare_loader, KERMANY_NAMES
from model import OCTSSRet


def grad_cam(model, img, target_class):
    feats, grads = [], []
    last_stage = model.backbone[-1]
    h_f = last_stage.register_forward_hook(lambda m, i, o: feats.append(o))
    h_b = last_stage.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    out = model(img)
    sel = out["logits_cls"][torch.arange(img.shape[0]), target_class].sum()
    model.zero_grad(); sel.backward()
    h_f.remove(); h_b.remove()
    f = feats[0]; g = grads[0]
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
    ap.add_argument("--out", default=str(ROOT / "paper" / "paper_latex" / "figures" / "fig_ekacare.pdf"))
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

    loader = get_ekacare_loader(batch_size=1, image_size=sa["image_size"], num_workers=2)

    # collect 4 examples (2 healthy, 2 glaucomatous if available)
    examples = []
    seen = {0: 0, 1: 0}
    for batch in loader:
        y = int(batch["label"][0])
        if seen[y] >= 2: continue
        examples.append((batch["image"], y, batch["ilm"][0], batch["rpe"][0]))
        seen[y] += 1
        if all(v >= 2 for v in seen.values()): break

    if not examples:
        print("[ekacare] no usable examples found")
        return

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10, "axes.labelsize": 9})
    cmap = "inferno"; norm = Normalize(vmin=0.0, vmax=1.0)
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(14.0, 3.0 * n))
    if n == 1: axes = axes[None, :]

    for row, (img, y, ilm, rpe) in enumerate(examples):
        img_d = img.to(device)
        # use predicted-class CAM since ekacare is binary glaucoma, not Kermany 4-class
        with torch.no_grad():
            out0 = model(img_d)
            target = int(out0["logits_cls"].argmax(dim=-1).item())
        cam, out = grad_cam(model, img_d, torch.tensor([target], device=device))
        prob = F.softmax(out["logits_cls"], dim=-1).detach().cpu().numpy()[0]
        rgb = (img_d[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"(a) OCT B-scan  --  GT: glaucoma={'yes' if y==1 else 'no'}",
                               loc="left", fontsize=10, fontweight="bold")
        axes[row, 1].imshow(np.array(ilm))
        axes[row, 1].set_title("(b) ILM annotation", loc="left", fontsize=10, fontweight="bold")
        axes[row, 2].imshow(cam[0], cmap=cmap, norm=norm, interpolation="bilinear")
        axes[row, 2].set_title(f"(c) Grad-CAM -> {KERMANY_NAMES[target]}, p={prob[target]:.2f}",
                               loc="left", fontsize=10, fontweight="bold")
        axes[row, 3].imshow(rgb)
        axes[row, 3].imshow(cam[0], cmap=cmap, alpha=0.45, norm=norm, interpolation="bilinear")
        axes[row, 3].set_title("(d) Overlay", loc="left", fontsize=10, fontweight="bold")

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.010, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax); cb.set_label("Grad-CAM", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.suptitle("Qualitative cross-cohort attribution on the ekacare glaucoma OCT cohort (paired ILM/RPE annotations)",
                 fontsize=11, y=0.995)
    plt.subplots_adjust(left=0.02, right=0.92, top=0.96, bottom=0.02, hspace=0.20, wspace=0.05)
    plt.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.savefig(args.out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
