"""OCT-SSRet training on Kermany 2017."""
import sys, os, json, time, argparse, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, f1_score, roc_auc_score
)

from data import get_kermany_loaders, KERMANY_NAMES
from model import OCTSSRet, class_balanced_focal_loss


def evaluate(model, loader, device, freq, return_arrays=False):
    model.eval()
    ys, probs = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device)
            out = model(x)
            loss = class_balanced_focal_loss(out["logits_cls"], y, freq)
            losses.append(loss.item() * y.size(0))
            ys.append(y.cpu().numpy())
            probs.append(F.softmax(out["logits_cls"], dim=-1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(probs)
    pred = p.argmax(1)
    metrics = {
        "n": int(len(y)),
        "loss": float(sum(losses) / len(y)),
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(4)), zero_division=0)),
        "qwk": float(cohen_kappa_score(y, pred, labels=list(range(4)), weights="quadratic")),
    }
    try:
        metrics["macro_auc"] = float(roc_auc_score(np.eye(4)[y], p, average="macro", multi_class="ovr"))
    except Exception:
        metrics["macro_auc"] = float("nan")
    if return_arrays:
        return metrics, y, p
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--use_ssm", action="store_true")
    ap.add_argument("--ssm_layers", type=int, default=2)
    ap.add_argument("--ssm_state", type=int, default=16)
    ap.add_argument("--ssm_unidirectional", action="store_true",
                    help="If set, SSM is unidirectional (forward-only) instead of bidirectional.")
    ap.add_argument("--use_transformer", action="store_true")
    ap.add_argument("--transformer_layers", type=int, default=2)
    ap.add_argument("--transformer_heads", type=int, default=8)
    ap.add_argument("--use_mlp", action="store_true",
                    help="MLP bottleneck control (parameter-comparable, no token routing).")
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--mlp_expand", type=int, default=4)
    ap.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet50", "efficientnet_b0", "vit_b_16", "swin_v2_t", "retfound_vit_l"])
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze the backbone parameters (head + SSM only get gradients).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tag", default="oct_baseline")
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    out_dir = ROOT / "artifacts" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda"
    print(f"[data] loading Kermany loaders (image_size={args.image_size}, batch={args.batch})")
    train_loader, val_loader, test_loader = get_kermany_loaders(
        batch_size=args.batch, image_size=args.image_size, num_workers=args.num_workers,
    )
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    # class frequencies for class-balanced focal loss (computed from train labels)
    train_labels = np.array([int(r["label"]) for r in train_loader.dataset.rows])
    freq = torch.tensor(np.bincount(train_labels, minlength=4), device=device, dtype=torch.float32)
    print(f"[data] train class freq: {freq.tolist()}")

    model = OCTSSRet(num_classes=4, backbone=args.backbone,
                     use_ssm=args.use_ssm, ssm_layers=args.ssm_layers, ssm_state=args.ssm_state,
                     ssm_bidirectional=(not args.ssm_unidirectional),
                     use_transformer=args.use_transformer,
                     transformer_layers=args.transformer_layers,
                     transformer_heads=args.transformer_heads,
                     use_mlp=args.use_mlp, mlp_layers=args.mlp_layers,
                     mlp_expand=args.mlp_expand).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.freeze_backbone:
        # Freeze the backbone (timm ViT-L for retfound or torchvision feature
        # tree for the others). The SSM, transformer, mlp blocks plus norm,
        # dropout, classifier remain trainable.
        if hasattr(model, "_retfound"):
            for p in model._retfound.parameters():
                p.requires_grad = False
        elif hasattr(model, "_swin_features"):
            for p in model._swin_features.parameters():
                p.requires_grad = False
            for p in model._swin_norm.parameters():
                p.requires_grad = False
        elif hasattr(model, "_vit"):
            for p in model._vit.parameters():
                p.requires_grad = False
        elif isinstance(model.backbone, torch.nn.Module):
            for p in model.backbone.parameters():
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] params={n_params/1e6:.2f}M  TRAINABLE={n_trainable/1e6:.2f}M  use_ssm={args.use_ssm} (FROZEN backbone)")
    else:
        print(f"[model] params={n_params/1e6:.2f}M  use_ssm={args.use_ssm} ssm_layers={args.ssm_layers}")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_qwk = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0; n_seen = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device)
            out = model(x)
            loss = class_balanced_focal_loss(out["logits_cls"], y, freq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item() * y.size(0); n_seen += y.size(0)
        sched.step()
        train_loss = running_loss / n_seen

        val = evaluate(model, val_loader, device, freq)
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss, "val": val,
               "lr": sched.get_last_lr()[0], "secs": dt}
        history.append(rec)
        print(f"[ep {epoch:02d}] tr={train_loss:.3f} | val acc={val['acc']:.4f} f1={val['macro_f1']:.4f} "
              f"qwk={val['qwk']:.4f} auc={val['macro_auc']:.4f} loss={val['loss']:.3f} | {dt:.1f}s",
              flush=True)
        with (out_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if val["qwk"] > best_qwk:
            best_qwk = val["qwk"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                        "args": vars(args), "freq": freq.tolist()},
                       out_dir / "best.pt")

    print(f"[done] best val QWK = {best_qwk:.4f}")
    # final test
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test = evaluate(model, test_loader, device, freq)
    print(f"[test] {test}")
    with (out_dir / "test.json").open("w") as f:
        json.dump(test, f, indent=2)


if __name__ == "__main__":
    main()
