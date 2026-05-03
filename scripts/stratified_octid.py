"""Per-class stratified OCTID transfer analysis.

Splits the OCTID zero-shot transfer into well-mapped vs. poorly-mapped
class subsets per the OCTID->Kermany re-mapping disclosed in src/data.py:
  Well-mapped:    NORMAL (Normal -> Normal), DRUSEN (AMD -> DRUSEN)
  Poorly-mapped:  CNV (Macular Hole -> CNV), DME (DR -> DME)

Reports per-class precision/recall/F1 and stratified accuracy for two
checkpoints (typically Swin baseline vs. Swin+SSM) so the SSM-vs-baseline
gap can be reported separately on the well-mapped vs. poorly-mapped subsets.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data import get_octid_loader, KERMANY_NAMES
from model import OCTSSRet


def predict(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    sa = ck["args"]
    m = OCTSSRet(num_classes=4, backbone=sa.get("backbone","resnet18"),
                 use_ssm=sa.get("use_ssm", False),
                 ssm_layers=sa.get("ssm_layers", 2),
                 ssm_state=sa.get("ssm_state", 16),
                 ssm_bidirectional=(not sa.get("ssm_unidirectional", False)),
                 use_mlp=sa.get("use_mlp", False),
                 mlp_layers=sa.get("mlp_layers", 2),
                 mlp_expand=sa.get("mlp_expand", 4),
                 pretrained=False).cuda()
    m.load_state_dict(ck["model"], strict=True); m.eval()
    loader = get_octid_loader(batch_size=16, image_size=sa["image_size"], num_workers=2)
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].cuda()
            out = m(x)
            ys.append(batch["label"].numpy())
            preds.append(F.softmax(out["logits_cls"], dim=-1).argmax(dim=-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def stratified(y, pred, well_classes, poor_classes):
    well_mask = np.isin(y, well_classes)
    poor_mask = np.isin(y, poor_classes)
    out = {}
    for name, mask in [("well_mapped", well_mask), ("poor_mapped", poor_mask), ("all", np.ones_like(y, dtype=bool))]:
        if mask.sum() == 0:
            out[name] = {"n": 0, "acc": float("nan"), "macro_f1": float("nan")}
            continue
        out[name] = {
            "n": int(mask.sum()),
            "acc": float(accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], average="macro",
                                        labels=list(range(4)), zero_division=0)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline checkpoint")
    ap.add_argument("--ssm",      required=True, help="SSM checkpoint")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "stratified_octid.json"))
    args = ap.parse_args()

    # Well-mapped: NORMAL=3, DRUSEN=2.  Poorly-mapped: CNV=0, DME=1.
    well = [2, 3]; poor = [0, 1]
    print("[stratified] baseline...")
    yb, pb = predict(args.baseline)
    print("[stratified] SSM...")
    ys, ps = predict(args.ssm)

    res = {
        "n_total": int(len(yb)),
        "well_mapped_classes": [KERMANY_NAMES[i] for i in well],
        "poorly_mapped_classes": [KERMANY_NAMES[i] for i in poor],
        "baseline": stratified(yb, pb, well, poor),
        "ssm":      stratified(ys, ps, well, poor),
    }
    # delta SSM - baseline per stratum
    res["delta"] = {}
    for k in ("well_mapped", "poor_mapped", "all"):
        res["delta"][k] = {
            "acc":      res["ssm"][k]["acc"] - res["baseline"][k]["acc"],
            "macro_f1": res["ssm"][k]["macro_f1"] - res["baseline"][k]["macro_f1"],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== Stratified OCTID transfer ===")
    print(f"Well-mapped classes ({res['well_mapped_classes']}):")
    print(f"  baseline: n={res['baseline']['well_mapped']['n']}  acc={res['baseline']['well_mapped']['acc']:.4f}  F1={res['baseline']['well_mapped']['macro_f1']:.4f}")
    print(f"  SSM:      n={res['ssm']['well_mapped']['n']}      acc={res['ssm']['well_mapped']['acc']:.4f}  F1={res['ssm']['well_mapped']['macro_f1']:.4f}")
    print(f"  delta:    acc {res['delta']['well_mapped']['acc']:+.4f}   F1 {res['delta']['well_mapped']['macro_f1']:+.4f}")
    print(f"\nPoorly-mapped classes ({res['poorly_mapped_classes']}):")
    print(f"  baseline: n={res['baseline']['poor_mapped']['n']}  acc={res['baseline']['poor_mapped']['acc']:.4f}  F1={res['baseline']['poor_mapped']['macro_f1']:.4f}")
    print(f"  SSM:      n={res['ssm']['poor_mapped']['n']}      acc={res['ssm']['poor_mapped']['acc']:.4f}  F1={res['ssm']['poor_mapped']['macro_f1']:.4f}")
    print(f"  delta:    acc {res['delta']['poor_mapped']['acc']:+.4f}   F1 {res['delta']['poor_mapped']['macro_f1']:+.4f}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
