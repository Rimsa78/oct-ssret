"""5-crop test-time augmentation for OCT-SSRet on the Kermany held-out test
split. Averages softmax probabilities across 5 views: original, horizontal
flip, and three corner-shifted crops at 92% scale.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from data import get_kermany_loaders, KERMANY_NAMES
from model import OCTSSRet


def make_views(img: torch.Tensor, crop_frac: float = 0.92):
    B, C, H, W = img.shape
    views = [img, torch.flip(img, dims=[3])]
    ch = int(round(H * crop_frac)); cw = int(round(W * crop_frac))
    for (y0, x0) in [(0, 0), (0, W - cw), (H - ch, 0)]:
        crop = img[:, :, y0:y0 + ch, x0:x0 + cw]
        crop = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
        views.append(crop)
    return views


def bootstrap_ci(a, b, statfn, n_boot=1000, alpha=0.05, seed=17):
    rng = np.random.default_rng(seed)
    n = len(a); boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        try: boots[k] = statfn(a[idx], b[idx])
        except Exception: boots[k] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0: return float("nan"), float("nan")
    return float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop_frac", type=float, default=0.92)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    model = OCTSSRet(num_classes=4, backbone=sa.get("backbone","resnet18"),
                     use_ssm=sa.get("use_ssm", False),
                     ssm_layers=sa.get("ssm_layers", 2),
                     ssm_state=sa.get("ssm_state", 16),
                     ssm_bidirectional=(not sa.get("ssm_unidirectional", False)),
                     use_mlp=sa.get("use_mlp", False),
                     mlp_layers=sa.get("mlp_layers", 2),
                     mlp_expand=sa.get("mlp_expand", 4),
                     pretrained=False).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    print(f"[tta] ckpt epoch={ck['epoch']}  views=5  crop_frac={args.crop_frac}")

    _, _, test_loader = get_kermany_loaders(batch_size=args.batch, image_size=sa["image_size"], num_workers=2)
    ys, probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            sm_acc = None
            for v in make_views(x, args.crop_frac):
                out = model(v)
                sm = F.softmax(out["logits_cls"], dim=-1)
                sm_acc = sm if sm_acc is None else sm_acc + sm
            sm_avg = (sm_acc / 5).cpu().numpy()
            ys.append(batch["label"].numpy())
            probs.append(sm_avg)

    y = np.concatenate(ys); p = np.concatenate(probs); pred = p.argmax(1)
    acc = float(accuracy_score(y, pred))
    f1m = float(f1_score(y, pred, average="macro", labels=list(range(4)), zero_division=0))
    qwk = float(cohen_kappa_score(y, pred, labels=list(range(4)), weights="quadratic"))
    try:
        auc = float(roc_auc_score(np.eye(4)[y], p, average="macro", multi_class="ovr"))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(4))).tolist()
    cr = classification_report(y, pred, labels=list(range(4)),
                               target_names=KERMANY_NAMES, zero_division=0, output_dict=True)
    acc_ci = bootstrap_ci(y, pred, lambda a, b: float(accuracy_score(a, b)))
    qwk_ci = bootstrap_ci(y, pred, lambda a, b: float(cohen_kappa_score(a, b, labels=list(range(4)), weights="quadratic")))

    res = {"n": int(len(y)), "n_views": 5, "crop_frac": args.crop_frac,
           "acc_tta": acc, "macro_f1_tta": f1m, "qwk_tta": qwk, "macro_auc_tta": auc,
           "acc_tta_95CI": list(acc_ci), "qwk_tta_95CI": list(qwk_ci),
           "confusion_tta": cm, "classification_report_tta": cr}
    out_path = Path(args.ckpt).with_name("eval_tta.json")
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)
    print(f"\nACC TTA = {acc:.4f}  [{acc_ci[0]:.4f}-{acc_ci[1]:.4f}]")
    print(f"F1  TTA = {f1m:.4f}")
    print(f"QWK TTA = {qwk:.4f}  [{qwk_ci[0]:.4f}-{qwk_ci[1]:.4f}]")
    print(f"AUC TTA = {auc:.4f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
