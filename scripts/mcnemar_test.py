"""McNemar's test on the Kermany test predictions of two checkpoints.

Reports the contingency table (n_01, n_10) where:
  n_01 = #cases A wrong, B right
  n_10 = #cases A right,  B wrong
and the McNemar chi-squared p-value with continuity correction. A small
p-value indicates the two models differ significantly on this test set.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2

from data import get_kermany_loaders, KERMANY_NAMES
from model import OCTSSRet


def predict(ckpt_path, batch_size=32):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    sa = ck["args"]
    model = OCTSSRet(num_classes=4, backbone=sa.get("backbone","resnet18"),
                     use_ssm=sa.get("use_ssm", False),
                     ssm_layers=sa.get("ssm_layers", 2),
                     ssm_state=sa.get("ssm_state", 16),
                     pretrained=False).cuda()
    model.load_state_dict(ck["model"], strict=True); model.eval()
    _, _, test_loader = get_kermany_loaders(batch_size=batch_size, image_size=sa["image_size"], num_workers=2)
    ys, preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].cuda()
            out = model(x)
            ys.append(batch["label"].numpy())
            preds.append(F.softmax(out["logits_cls"], dim=-1).argmax(dim=-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def mcnemar(a_correct, b_correct):
    """a_correct, b_correct are bool arrays of length n. Returns (n_01, n_10, chi2_stat, p_value)."""
    n_01 = int(((~a_correct) & b_correct).sum())   # A wrong, B right
    n_10 = int((a_correct & (~b_correct)).sum())   # A right, B wrong
    if n_01 + n_10 == 0:
        return n_01, n_10, 0.0, 1.0
    # Continuity-corrected McNemar
    chi = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p = 1 - chi2.cdf(chi, df=1)
    return n_01, n_10, float(chi), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint A (e.g. baseline)")
    ap.add_argument("--b", required=True, help="checkpoint B (e.g. SSM)")
    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    args = ap.parse_args()

    print(f"[predict] {args.name_a}: {args.a}")
    y, pa = predict(args.a)
    print(f"[predict] {args.name_b}: {args.b}")
    _, pb = predict(args.b)

    a_correct = (pa == y); b_correct = (pb == y)
    a_acc = float(a_correct.mean()); b_acc = float(b_correct.mean())
    n_01, n_10, chi, p = mcnemar(a_correct, b_correct)
    both_right = int((a_correct & b_correct).sum())
    both_wrong = int(((~a_correct) & (~b_correct)).sum())

    res = {
        "n_test": int(len(y)),
        "name_a": args.name_a, "name_b": args.name_b,
        "acc_a": a_acc, "acc_b": b_acc,
        "n_both_right": both_right, "n_both_wrong": both_wrong,
        "n_only_a_right": n_10, "n_only_b_right": n_01,
        "chi2": chi, "p_value": p,
    }
    out_path = Path(args.b).with_name(f"mcnemar_vs_{args.name_a.replace(' ','_')}.json")
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)

    print()
    print(f"=== McNemar's test (Kermany test, n={len(y)}) ===")
    print(f"{args.name_a}: ACC={a_acc:.4f}    {args.name_b}: ACC={b_acc:.4f}")
    print(f"Contingency:  both right={both_right}  both wrong={both_wrong}")
    print(f"  only-{args.name_a}-right (n_10) = {n_10}")
    print(f"  only-{args.name_b}-right (n_01) = {n_01}")
    print(f"chi^2 (with continuity correction) = {chi:.4f}")
    print(f"p-value = {p:.4g}")
    sig = "*" if p < 0.05 else ""
    print(f"{'SIGNIFICANT (p<0.05)' if p < 0.05 else 'not significant'} {sig}")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
