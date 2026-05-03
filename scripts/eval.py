"""Evaluate an OCT-SSRet checkpoint on the Kermany held-out test split.

Reports accuracy, macro-F1, QWK, macro AUC (one-vs-rest), per-class
precision/recall/AUC, confusion matrix, and 95% bootstrap CIs on the
headline metrics.
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


def bootstrap_ci_paired(a, b, statfn, n_boot=1000, alpha=0.05, seed=17):
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots[k] = statfn(a[idx], b[idx])
        except Exception:
            boots[k] = np.nan
    boots = boots[~np.isnan(boots)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    model = OCTSSRet(num_classes=4, backbone=sa.get("backbone", "resnet18"),
                     use_ssm=sa.get("use_ssm", False),
                     ssm_layers=sa.get("ssm_layers", 2),
                     ssm_state=sa.get("ssm_state", 16),
                     pretrained=False).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    print(f"[eval] ckpt epoch={ck['epoch']}  backbone={sa.get('backbone','resnet18')}  use_ssm={sa.get('use_ssm', False)}")

    _, _, test_loader = get_kermany_loaders(
        batch_size=args.batch, image_size=sa["image_size"], num_workers=2)
    print(f"[eval] test n={len(test_loader.dataset)}")

    ys, probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            out = model(x)
            ys.append(batch["label"].numpy())
            probs.append(F.softmax(out["logits_cls"], dim=-1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(probs)
    pred = p.argmax(1)

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
    acc_ci = bootstrap_ci_paired(y, pred, lambda a, b: float(accuracy_score(a, b)))
    f1_ci  = bootstrap_ci_paired(y, pred, lambda a, b: float(f1_score(a, b, average="macro", labels=list(range(4)), zero_division=0)))
    qwk_ci = bootstrap_ci_paired(y, pred, lambda a, b: float(cohen_kappa_score(a, b, labels=list(range(4)), weights="quadratic")))

    res = {"n": int(len(y)),
           "acc": acc, "macro_f1": f1m, "qwk": qwk, "macro_auc": auc,
           "acc_95CI": list(acc_ci), "macro_f1_95CI": list(f1_ci), "qwk_95CI": list(qwk_ci),
           "confusion": cm, "classification_report": cr,
           "label_dist": np.bincount(y, minlength=4).tolist(),
           "pred_dist": np.bincount(pred, minlength=4).tolist()}
    out_path = Path(args.ckpt).with_name("eval_test.json")
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)

    print(f"\nACC = {acc:.4f}  [{acc_ci[0]:.4f}-{acc_ci[1]:.4f}]")
    print(f"F1  = {f1m:.4f}  [{f1_ci[0]:.4f}-{f1_ci[1]:.4f}]")
    print(f"QWK = {qwk:.4f}  [{qwk_ci[0]:.4f}-{qwk_ci[1]:.4f}]")
    print(f"AUC = {auc:.4f}")
    print("Per class:")
    for k, name in enumerate(KERMANY_NAMES):
        d = cr[name]
        try:
            ovr_auc = float(roc_auc_score((y == k).astype(int), p[:, k]))
        except Exception:
            ovr_auc = float("nan")
        print(f"  {k} {name:8s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1-score']:.3f} AUC={ovr_auc:.3f}  n={int(d['support'])}")
    print(f"\nConfusion matrix (rows=GT, cols=pred):")
    print(f"        " + "  ".join(f"{n:>5s}" for n in KERMANY_NAMES))
    for i, row in enumerate(cm):
        print(f"  {KERMANY_NAMES[i]:6s} " + "  ".join(f"{v:>5d}" for v in row))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
