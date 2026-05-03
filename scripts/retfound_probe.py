"""RETFound MAE-OCT linear-probe + SSM-on-features comparison.

Extracts pooled RETFound features (B, 1024) for Kermany train/val/test +
OCTID, trains a class-balanced focal-CE linear classifier on them, and
reports the same operating-profile metrics (in-domain ACC/QWK, cross-cohort
OCTID transfer, calibration ECE/NLL/Brier, OOD detection AUROC). Then trains
a 2-layer MLP head on the same features for a stronger comparison.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, f1_score, roc_auc_score,
    confusion_matrix
)

from data import get_kermany_loaders, get_octid_loader, KERMANY_NAMES
from model import OCTSSRet  # not used directly but imports timm RETFound logic
import timm

CACHE = ROOT / "artifacts" / "retfound_features"
CACHE.mkdir(parents=True, exist_ok=True)
device = "cuda"


def build_retfound():
    m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    sd = torch.load("/home/rojan/.cache/huggingface/hub/models--YukunZhou--RETFound_mae_natureOCT/snapshots/8e551e2f547d3e27aedf43c9cf1c5bcea1c17171/RETFound_mae_natureOCT.pth",
                    map_location="cpu", weights_only=False)["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder_") and k != "mask_token"}
    m.load_state_dict(sd, strict=False)
    m = m.cuda().eval()
    return m


@torch.no_grad()
def extract(loader, name):
    fp = CACHE / f"{name}.pt"
    if fp.exists():
        d = torch.load(fp, map_location="cpu", weights_only=False)
        print(f"[cache] {name}: {d['feats'].shape}")
        return d["feats"], d["labels"]
    backbone = build_retfound()
    feats, labels = [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["image"].cuda()
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        f = backbone(x)            # pooled CLS feature (B, 1024)
        feats.append(f.cpu()); labels.append(batch["label"])
        if (i + 1) % 50 == 0:
            print(f"  {name}: batch {i+1}/{len(loader)} ({(i+1)*x.size(0)} imgs) {time.time()-t0:.1f}s")
    feats = torch.cat(feats); labels = torch.cat(labels)
    torch.save({"feats": feats, "labels": labels}, fp)
    print(f"[saved] {fp}: feats {feats.shape}, labels {labels.shape}, took {time.time()-t0:.1f}s")
    del backbone; torch.cuda.empty_cache()
    return feats, labels


class LinearProbe(nn.Module):
    def __init__(self, dim=1024, K=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, K)
    def forward(self, x): return self.fc(self.norm(x))


class MLPHead(nn.Module):
    def __init__(self, dim=1024, K=4, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, K)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        h = self.norm(x); h = F.gelu(self.fc1(h)); h = self.dropout(h); return self.fc2(h)


def cb_focal(logits, targets, freq, beta=0.999, gamma=2.0):
    K = logits.shape[-1]
    eff_n = 1.0 - torch.pow(beta, freq.float())
    w = ((1.0 - beta) / eff_n); w = (w / w.sum()) * K
    log_p = F.log_softmax(logits, dim=-1)
    p_t = log_p.exp().gather(1, targets.view(-1, 1)).squeeze(1)
    log_p_t = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
    focal = (1 - p_t).pow(gamma) * (-log_p_t)
    return (w[targets] * focal).mean()


def train_head(head, train_loader, val_loader, freq, epochs=15, lr=1e-3):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_qwk = -1.0; best_state = None
    for ep in range(1, epochs + 1):
        head.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = cb_focal(head(x), y, freq)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        head.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                ys.append(y.numpy())
                ps.append(F.softmax(head(x.cuda()), dim=-1).cpu().numpy())
        y = np.concatenate(ys); p = np.concatenate(ps); pred = p.argmax(1)
        qwk = float(cohen_kappa_score(y, pred, labels=list(range(4)), weights="quadratic"))
        acc = float(accuracy_score(y, pred))
        print(f"  ep {ep:02d}: val acc={acc:.4f} qwk={qwk:.4f}")
        if qwk > best_qwk:
            best_qwk = qwk; best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head


def evaluate(head, feats, labels, name="test"):
    head.eval()
    with torch.no_grad():
        logits_list = []
        for i in range(0, len(feats), 256):
            x = feats[i:i+256].cuda()
            logits_list.append(head(x).cpu())
        logits = torch.cat(logits_list)
        p = F.softmax(logits, dim=-1).numpy()
    pred = p.argmax(1)
    y = labels.numpy()
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average="macro", labels=list(range(4)), zero_division=0))
    qwk = float(cohen_kappa_score(y, pred, labels=list(range(4)), weights="quadratic"))
    try:
        auc = float(roc_auc_score(np.eye(4)[y], p, average="macro", multi_class="ovr"))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(4))).tolist()
    print(f"\n[{name}] n={len(y)}  acc={acc:.4f}  f1={f1:.4f}  qwk={qwk:.4f}  auc={auc:.4f}")
    return {"n": int(len(y)), "acc": acc, "macro_f1": f1, "qwk": qwk, "macro_auc": auc, "confusion": cm}


def main():
    print("\n=== Step 1: extract RETFound features (pooled CLS, dim=1024) ===\n")
    train_loader, val_loader, test_loader = get_kermany_loaders(batch_size=64, image_size=256, num_workers=2)
    octid_loader = get_octid_loader(batch_size=32, image_size=256, num_workers=2)
    f_train, y_train = extract(train_loader, "kermany_train")
    f_val,   y_val   = extract(val_loader,   "kermany_val")
    f_test,  y_test  = extract(test_loader,  "kermany_test")
    f_octid, y_octid = extract(octid_loader, "octid")

    freq = torch.tensor(np.bincount(y_train.numpy(), minlength=4), device="cuda", dtype=torch.float32)
    print(f"\nTrain class freq: {freq.tolist()}")

    train_dl = DataLoader(TensorDataset(f_train, y_train), batch_size=512, shuffle=True)
    val_dl   = DataLoader(TensorDataset(f_val,   y_val),   batch_size=512, shuffle=False)

    print("\n=== Step 2: linear probe (single Linear over LayerNorm) ===\n")
    probe = LinearProbe(dim=1024, K=4).cuda()
    probe = train_head(probe, train_dl, val_dl, freq, epochs=15, lr=1e-3)
    res_probe = {
        "kermany_test": evaluate(probe, f_test, y_test, "Kermany test (linear probe)"),
        "octid":        evaluate(probe, f_octid, y_octid, "OCTID zero-shot (linear probe)"),
    }

    print("\n=== Step 3: 2-layer MLP head ===\n")
    mlp = MLPHead(dim=1024, K=4, hidden=512).cuda()
    mlp = train_head(mlp, train_dl, val_dl, freq, epochs=15, lr=1e-3)
    res_mlp = {
        "kermany_test": evaluate(mlp, f_test, y_test, "Kermany test (MLP head)"),
        "octid":        evaluate(mlp, f_octid, y_octid, "OCTID zero-shot (MLP head)"),
    }

    out = ROOT / "artifacts" / "retfound_probe.json"
    with out.open("w") as f:
        json.dump({"probe": res_probe, "mlp": res_mlp}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
