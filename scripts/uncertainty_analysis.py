"""Uncertainty analysis: calibration (ECE, NLL, Brier), OOD detection
(MSP, entropy, energy), selective prediction (risk--coverage AURC).

For each checkpoint, reports:
  - In-domain calibration on Kermany 2017 test split
  - OOD AUROC (Kermany ID vs OCTID OOD) under three scoring rules
  - Selective-prediction risk--coverage curve and AURC (area under
    risk--coverage)
  - Confidence-shift histograms (saved as PNG)

This deepens the operating-profile claim of the paper from "small AUROC
gain on MSP" to a multi-axis uncertainty story.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

from data import get_kermany_loaders, get_octid_loader, KERMANY_NAMES
from model import OCTSSRet


def collect(model, loader, device):
    ys, probs, logits = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            out = model(x)
            probs.append(F.softmax(out["logits_cls"], dim=-1).cpu().numpy())
            logits.append(out["logits_cls"].cpu().numpy())
            ys.append(batch["label"].numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(logits)


# ----- Calibration metrics -----
def expected_calibration_error(probs, labels, n_bins=15):
    """Expected Calibration Error with equal-width binning on max-softmax confidence."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def negative_log_likelihood(probs, labels):
    eps = 1e-12
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())


def brier_multiclass(probs, labels, K=4):
    onehot = np.eye(K)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ----- OOD scores -----
def msp_score(probs):
    return -probs.max(axis=1)  # higher = more OOD (lower MSP)


def entropy_score(probs):
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def energy_score(logits, T=1.0):
    """Free-energy: -T * logsumexp(logits/T). Lower energy = more ID."""
    return -T * np.log(np.exp(logits / T).sum(axis=1) + 1e-12)


# ----- Selective prediction (risk--coverage) -----
def risk_coverage(confs, correct):
    """Returns (coverages, risks) sorted by descending confidence."""
    order = np.argsort(-confs)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    coverages = np.arange(1, n + 1) / n
    risks = 1.0 - cum_correct / np.arange(1, n + 1)
    return coverages, risks


def aurc(coverages, risks):
    """Area under the risk--coverage curve."""
    return float(np.trapezoid(risks, coverages))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out_fig_dir", default=str(ROOT / "paper" / "paper_latex" / "figures"))
    args = ap.parse_args()
    Path(args.out_fig_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    model = OCTSSRet(num_classes=4, backbone=sa.get("backbone","resnet18"),
                     use_ssm=sa.get("use_ssm", False),
                     ssm_layers=sa.get("ssm_layers", 2),
                     ssm_state=sa.get("ssm_state", 16),
                     ssm_bidirectional=(not sa.get("ssm_unidirectional", False)),
                     pretrained=False).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    print(f"[uncertainty] ckpt={args.ckpt}  use_ssm={sa.get('use_ssm', False)}")

    _, _, kerm_test = get_kermany_loaders(batch_size=args.batch, image_size=sa["image_size"], num_workers=2)
    octid = get_octid_loader(batch_size=args.batch // 2, image_size=sa["image_size"], num_workers=2)
    y_id, p_id, l_id   = collect(model, kerm_test, device)
    y_ood, p_ood, l_ood = collect(model, octid,    device)

    # Calibration on Kermany ID
    ece   = expected_calibration_error(p_id, y_id, n_bins=15)
    nll   = negative_log_likelihood(p_id, y_id)
    brier = brier_multiclass(p_id, y_id, K=4)

    # OOD detection (3 scores)
    y_ood_label = np.concatenate([np.zeros(len(p_id)), np.ones(len(p_ood))])
    msp_all  = np.concatenate([msp_score(p_id),     msp_score(p_ood)])
    ent_all  = np.concatenate([entropy_score(p_id), entropy_score(p_ood)])
    ene_all  = np.concatenate([energy_score(l_id),  energy_score(l_ood)])
    auroc_msp    = float(roc_auc_score(y_ood_label, msp_all))
    auroc_ent    = float(roc_auc_score(y_ood_label, ent_all))
    auroc_energy = float(roc_auc_score(y_ood_label, ene_all))

    # Selective prediction (in-domain)
    confs_id = p_id.max(axis=1)
    correct_id = (p_id.argmax(axis=1) == y_id).astype(float)
    cov, risk = risk_coverage(confs_id, correct_id)
    aurc_val = aurc(cov, risk)

    res = {
        "ckpt": args.ckpt,
        "n_id": int(len(y_id)), "n_ood": int(len(y_ood)),
        "calibration_id": {"ECE_15bin": ece, "NLL": nll, "Brier": brier,
                            "ID_max_conf_mean": float(confs_id.mean()),
                            "ID_max_conf_std":  float(confs_id.std())},
        "ood_detection_octid": {
            "AUROC_MSP":     auroc_msp,
            "AUROC_entropy": auroc_ent,
            "AUROC_energy":  auroc_energy,
            "OOD_max_conf_mean": float(p_ood.max(axis=1).mean()),
            "OOD_max_conf_std":  float(p_ood.max(axis=1).std()),
        },
        "selective_prediction_id": {
            "AURC": aurc_val,
            "risk_at_50_coverage": float(np.interp(0.5, cov, risk)),
            "risk_at_75_coverage": float(np.interp(0.75, cov, risk)),
            "risk_at_90_coverage": float(np.interp(0.9, cov, risk)),
        },
    }
    out_path = Path(args.ckpt).with_name("uncertainty.json")
    with out_path.open("w") as f:
        json.dump(res, f, indent=2)

    # ---- Confidence-shift histogram ----
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
    ax.hist(confs_id, bins=30, range=(0, 1), alpha=0.55,
            label=f"In-domain Kermany (n={len(p_id)})", color="#1f4e79", density=True)
    ax.hist(p_ood.max(axis=1), bins=30, range=(0, 1), alpha=0.55,
            label=f"Cross-cohort OCTID (n={len(p_ood)})", color="#a04040", density=True)
    ax.axvline(confs_id.mean(),       color="#1f4e79", ls="--", lw=1.5, label=f"ID mean = {confs_id.mean():.3f}")
    ax.axvline(p_ood.max(axis=1).mean(), color="#a04040", ls="--", lw=1.5, label=f"OOD mean = {p_ood.max(axis=1).mean():.3f}")
    ax.set_xlabel("max-softmax confidence")
    ax.set_ylabel("density")
    ax.set_title(f"Confidence shift: in-domain vs. cross-cohort\n(SSM={sa.get('use_ssm', False)})", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(alpha=0.25)
    suffix = "_ssm" if sa.get("use_ssm", False) else "_baseline"
    fig_path = Path(args.out_fig_dir) / f"fig_confidence_shift{suffix}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Risk-coverage figure ----
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(cov, risk, lw=2, color="#1f4e79")
    ax.fill_between(cov, 0, risk, alpha=0.15, color="#1f4e79")
    ax.set_xlabel("coverage (fraction of test set retained)")
    ax.set_ylabel("selective risk (error rate on retained subset)")
    ax.set_title(f"Risk-coverage curve on Kermany 2017 test\nAURC = {aurc_val:.4f}  (SSM={sa.get('use_ssm', False)})", fontsize=11)
    ax.set_xlim([0, 1]); ax.set_ylim([0, max(risk.max() * 1.1, 0.2)])
    ax.grid(alpha=0.3)
    fig_path = Path(args.out_fig_dir) / f"fig_risk_coverage{suffix}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Calibration (Kermany ID, n={len(y_id)}) ===")
    print(f"  ECE (15-bin) = {ece:.4f}")
    print(f"  NLL          = {nll:.4f}")
    print(f"  Brier        = {brier:.4f}")
    print(f"  ID max-conf  mean = {confs_id.mean():.4f}  std = {confs_id.std():.4f}")
    print(f"\n=== OOD detection (Kermany ID vs OCTID OOD) ===")
    print(f"  AUROC MSP     = {auroc_msp:.4f}")
    print(f"  AUROC entropy = {auroc_ent:.4f}")
    print(f"  AUROC energy  = {auroc_energy:.4f}")
    print(f"  OOD max-conf mean = {p_ood.max(axis=1).mean():.4f}")
    print(f"\n=== Selective prediction (Kermany ID) ===")
    print(f"  AURC         = {aurc_val:.4f}")
    print(f"  risk @ 50%   = {res['selective_prediction_id']['risk_at_50_coverage']:.4f}")
    print(f"  risk @ 75%   = {res['selective_prediction_id']['risk_at_75_coverage']:.4f}")
    print(f"  risk @ 90%   = {res['selective_prediction_id']['risk_at_90_coverage']:.4f}")
    print(f"\n[saved] {out_path}")
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
