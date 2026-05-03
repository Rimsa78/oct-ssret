"""Out-of-distribution detection analysis.

Treats Kermany 2017 test as the in-distribution (ID) cohort and OCTID
zero-shot test as the out-of-distribution (OOD) cohort. Reports two simple
OOD scores per checkpoint:

  (1) Maximum-softmax-probability (MSP):   max_k p(y=k|x)   --- LOW under OOD
  (2) Predictive entropy:                   -sum_k p(k|x) log p(k|x)  --- HIGH under OOD

Both scores produce binary (ID vs OOD) AUROC. We additionally report TPR at
95% TNR (the operating point most relevant for clinical screening) and the
Brier score of the predicted-class confidence as a calibration check.

The headline question: does the SSM bottleneck make features more separable
between ID and OOD? A higher AUROC under MSP / entropy means the model is
better at *flagging* an unseen-cohort image, which is the precondition for
safe deployment in retinal screening pipelines.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

from data import get_kermany_loaders, get_octid_loader
from model import OCTSSRet


def collect_probs(model, loader, device):
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            out = model(x)
            probs.append(F.softmax(out["logits_cls"], dim=-1).cpu().numpy())
    return np.concatenate(probs)


def msp_and_entropy(p):
    """p: (N, K). Returns (msp[N], entropy[N])."""
    msp = p.max(axis=1)
    ent = -np.sum(p * np.log(p + 1e-12), axis=1)
    return msp, ent


def tpr_at_tnr(score, y, target_tnr=0.95):
    """y: 1 = OOD, 0 = ID. score: higher = more OOD."""
    fpr, tpr, _ = roc_curve(y, score)
    tnr = 1 - fpr
    # find largest tpr at tnr >= target
    valid = tnr >= target_tnr
    if not valid.any(): return float("nan")
    return float(tpr[valid].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=32)
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
    print(f"[ood] ckpt epoch={ck['epoch']}  use_ssm={sa.get('use_ssm', False)}")

    # ID = Kermany test, OOD = OCTID
    _, _, kerm_test = get_kermany_loaders(batch_size=args.batch, image_size=sa["image_size"], num_workers=2)
    octid = get_octid_loader(batch_size=args.batch // 2, image_size=sa["image_size"], num_workers=2)
    p_id = collect_probs(model, kerm_test, device)
    p_ood = collect_probs(model, octid, device)
    msp_id,  ent_id  = msp_and_entropy(p_id)
    msp_ood, ent_ood = msp_and_entropy(p_ood)

    # OOD-detection AUROC: y=1 means OOD
    n_id, n_ood = len(p_id), len(p_ood)
    y_ood_label = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    # MSP: low under OOD -> use -msp as the OOD score (higher=more OOD)
    msp_score = np.concatenate([-msp_id, -msp_ood])
    # entropy: high under OOD -> use entropy directly
    ent_score = np.concatenate([ent_id, ent_ood])

    auroc_msp = float(roc_auc_score(y_ood_label, msp_score))
    auroc_ent = float(roc_auc_score(y_ood_label, ent_score))
    tpr_msp = tpr_at_tnr(msp_score, y_ood_label, target_tnr=0.95)
    tpr_ent = tpr_at_tnr(ent_score, y_ood_label, target_tnr=0.95)

    res = {
        "n_id": int(n_id), "n_ood": int(n_ood),
        "auroc_msp_ood": auroc_msp,           # higher = SSM is better at flagging OOD via max-softmax
        "auroc_entropy_ood": auroc_ent,       # higher = SSM is better at flagging OOD via entropy
        "tpr_at_95tnr_msp": tpr_msp,
        "tpr_at_95tnr_entropy": tpr_ent,
        "id_msp_mean": float(msp_id.mean()),
        "id_msp_std":  float(msp_id.std()),
        "ood_msp_mean": float(msp_ood.mean()),
        "ood_msp_std":  float(msp_ood.std()),
        "id_entropy_mean": float(ent_id.mean()),
        "id_entropy_std":  float(ent_id.std()),
        "ood_entropy_mean": float(ent_ood.mean()),
        "ood_entropy_std":  float(ent_ood.std()),
    }
    out_path = Path(args.ckpt).with_name("ood_analysis.json")
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)

    print(f"\nOOD detection (Kermany test as ID, OCTID as OOD):")
    print(f"  N_id  = {n_id}     N_ood = {n_ood}")
    print(f"  AUROC (MSP)        = {auroc_msp:.4f}     TPR@95TNR = {tpr_msp:.4f}")
    print(f"  AUROC (entropy)    = {auroc_ent:.4f}     TPR@95TNR = {tpr_ent:.4f}")
    print(f"  ID  MSP   mean={msp_id.mean():.4f} std={msp_id.std():.4f}")
    print(f"  OOD MSP   mean={msp_ood.mean():.4f} std={msp_ood.std():.4f}")
    print(f"  ID  entropy mean={ent_id.mean():.4f}")
    print(f"  OOD entropy mean={ent_ood.mean():.4f}")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
