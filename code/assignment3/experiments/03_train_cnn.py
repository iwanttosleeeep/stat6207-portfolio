#!/usr/bin/env python3
"""
03_train_cnn.py
===============
Advanced Task -- repeat the classification *without* any pre-trained weights:
a relatively small CNN (Conv -> BN -> ReLU -> Pool ... -> GAP -> FC) trained
entirely from random initialisation, following the architecture ideas of
Lecture 6 ("build deep, not wide"; batch-norm; dropout; max-pooling).

Two depths are provided so the "depth" knob can be studied:
    cnn_small : 3 conv stages
    cnn_base  : 4 conv stages

Regularisation for the tiny 160-image training set: horizontal-flip + affine +
colour-jitter augmentation, dropout, weight decay, label smoothing and early
stopping.  Input normalisation uses the training-set mean/std (saved).
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from common import (MODELS, PREDICTIONS, RESULTS, SEED, FaceDataset,
                    cnn_transforms, compute_mean_std, compute_metrics,
                    describe_device, get_device, load_or_make_split,
                    loader_generator, manifest_index, save_json, seed_worker,
                    set_seed)


# --------------------------------------------------------------------------- #
def conv_bn(cin, cout, k=3, p=1):
    return [nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class SmallCNN(nn.Module):
    """Conv-BN-ReLU-Pool -> ... -> GlobalAvgPool -> Dropout -> Linear."""

    def __init__(self, depth="small", width=32, n_classes=2, dropout=0.5):
        super().__init__()
        stages = 3 if depth == "small" else 4
        chans = [3] + [width * (2 ** i) for i in range(stages)]
        blocks = []
        for i in range(stages):
            blocks += conv_bn(chans[i], chans[i + 1])
            blocks += conv_bn(chans[i + 1], chans[i + 1])
            blocks += [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(chans[-1], n_classes))
        self.out_dim = chans[-1]

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_cnn(arch):
    if arch == "cnn_small":
        return SmallCNN(depth="small", width=32)
    if arch == "cnn_base":
        return SmallCNN(depth="base", width=32)
    raise ValueError(arch)


# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.extend(logits.argmax(1).detach().cpu().tolist())
            trues.extend(y.detach().cpu().tolist())
    m = compute_metrics(trues, preds)
    m["loss"] = total_loss / len(loader.dataset)
    return m


def train(arch, train_loader, val_loader, device, cfg):
    model = build_cnn(arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    history, best = [], {"f1": -1.0, "loss": 1e9, "epoch": -1, "state": None}
    patience_ctr, t0 = 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        te = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer=opt)
        va = run_epoch(model, val_loader, criterion, device)
        sched.step()
        history.append({"epoch": ep, "train_loss": tr["loss"],
                        "train_acc": tr["accuracy"], "val_loss": va["loss"],
                        "val_acc": va["accuracy"], "val_f1": va["f1"],
                        "sec": time.time() - te})
        print(f"  [{arch}] ep{ep:02d} tr_loss={tr['loss']:.4f} "
              f"tr_acc={tr['accuracy']:.3f} | va_loss={va['loss']:.4f} "
              f"va_acc={va['accuracy']:.3f} va_f1={va['f1']:.3f} "
              f"({history[-1]['sec']:.1f}s)")
        better = (va["f1"] > best["f1"]) or (
            abs(va["f1"] - best["f1"]) < 1e-9 and va["loss"] < best["loss"])
        if better:
            best = {"f1": va["f1"], "loss": va["loss"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= cfg["patience"]:
            print(f"  [{arch}] early stop at epoch {ep}")
            break
    total_time = time.time() - t0

    model.load_state_dict(best["state"])
    model.eval()
    prob, vtrue, vloss = [], [], 0.0
    with torch.no_grad():
        for x, y, _ in val_loader:
            out = model(x.to(device))
            vloss += criterion(out, y.to(device)).item() * x.size(0)
            p = torch.softmax(out, dim=1)[:, 1]
            prob.extend(p.cpu().tolist())
            vtrue.extend(y.tolist())
    vpred = [1 if p >= 0.5 else 0 for p in prob]
    final = compute_metrics(vtrue, vpred)
    final["loss"] = vloss / len(val_loader.dataset)

    ckpt = os.path.join(MODELS, f"{arch}.pt")
    torch.save({"arch": arch, "state_dict": best["state"],
                "hyperparams": cfg, "epoch": best["epoch"],
                "n_params": n_params}, ckpt)
    result = {
        "model": arch, "arch": arch, "seed": SEED,
        "device": describe_device(),
        "n_params": int(n_params),
        "hyperparams": cfg, "history": history,
        "best_epoch": best["epoch"], "best_val_f1": best["f1"],
        "train_time_sec": total_time, "final_val": final,
        "checkpoint": os.path.relpath(ckpt, RESULTS),
    }
    return result, vtrue, vpred, prob


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["cnn_small", "cnn_base"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"[env] {describe_device()}")

    index = manifest_index()
    split = load_or_make_split()
    mean, std = compute_mean_std(split["train"], index)
    save_json({"mean": mean, "std": std},
              os.path.join(RESULTS, "cnn_norm_stats.json"))
    print(f"[norm] mean={[round(m,4) for m in mean]} std={[round(s,4) for s in std]}")

    train_ds = FaceDataset(split["train"], index, cnn_transforms(mean, std, True))
    val_ds = FaceDataset(split["val"], index, cnn_transforms(mean, std, False))
    g = loader_generator()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        worker_init_fn=seed_worker, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    cfg = {
        "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-3,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "patience": args.patience, "label_smoothing": 0.05,
        "dropout": 0.5, "scheduler": "CosineAnnealingLR",
        "augmentation": "hflip+affine(12deg,0.06,0.88-1.12)+colorjitter(.25,.25,.15)",
        "normalization": f"train mean={[round(m,4) for m in mean]} "
                         f"std={[round(s,4) for s in std]}",
        "input_size": 160, "init": "random (no pre-trained weights)",
        "selection_rule": "max val F1, tie-break lower val loss",
    }

    all_results = []
    for arch in args.archs:
        print(f"\n=== {arch} ===")
        res, vtrue, vpred, prob = train(arch, train_loader, val_loader, device, cfg)
        all_results.append(res)
        save_json([{"file": split["val"][i], "true": int(vtrue[i]),
                    "pred": int(vpred[i]), "prob_female": float(prob[i]),
                    "correct": int(vtrue[i] == vpred[i])}
                   for i in range(len(vtrue))],
                  os.path.join(PREDICTIONS, f"{arch}_val.json"))
        print(f"  -> val_acc={res['final_val']['accuracy']:.3f} "
              f"val_f1={res['final_val']['f1']:.3f} best_ep={res['best_epoch']}")

    save_json(all_results, os.path.join(RESULTS, "cnn_results.json"))
    print(f"\n[out] {os.path.join(RESULTS, 'cnn_results.json')}")


if __name__ == "__main__":
    main()
