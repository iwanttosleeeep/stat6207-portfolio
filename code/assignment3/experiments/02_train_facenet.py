#!/usr/bin/env python3
"""
02_train_facenet.py
===================
Basic Task 3/5 -- fine-tune a pre-trained FaceNet (Inception-ResNet-v1,
VGGFace2 weights via facenet-pytorch) as a binary gender classifier with a
small classification head, under three freezing configurations.

Freeze configurations (module groups are top-level children of
InceptionResnetV1):
    F0  train head only                       (entire backbone frozen)
    F1  + final feature stage (repeat_3, block8, embedding)
    F2  + one more upper stage (mixed_7a, repeat_2)

Everything is recorded: exact frozen/trainable module groups, trainable
parameter counts, hyper-parameters, per-epoch history, timing, and the best
checkpoint + validation predictions.

Usage
-----
    python 02_train_facenet.py --configs F0 F1 F2
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import (MODELS, PREDICTIONS, RESULTS, SEED, LOGS, CLASS_NAMES,
                    FaceDataset, compute_metrics, describe_device,
                    facenet_transforms, loader_generator, load_or_make_split,
                    manifest_index, save_json, seed_worker, set_seed,
                    get_device)

# Inception-ResNet-v1 top-level children grouped into logical stages.
GROUP_MODULES = {
    "stem": ["conv2d_1a", "conv2d_2a", "conv2d_2b", "maxpool_3a",
             "conv2d_3b", "conv2d_4a", "conv2d_4b"],
    "stage1": ["repeat_1"],
    "reduction_a": ["mixed_6a"],
    "stage2": ["repeat_2"],
    "reduction_b": ["mixed_7a"],
    "stage3": ["repeat_3", "block8"],
    "embed": ["avgpool_1a", "dropout", "last_linear", "last_bn"],
}
HEAD_MODULE = "logits"

# groups (besides the head) that remain trainable in each config
CONFIG_TRAIN_GROUPS = {
    "F0": [],
    "F1": ["stage3", "embed"],
    "F2": ["stage2", "reduction_b", "stage3", "embed"],
}


# --------------------------------------------------------------------------- #
def build_model(device):
    from facenet_pytorch import InceptionResnetV1
    # classify=True -> forward returns logits; we replace the 8631-way logits
    # with a 2-way head.  (Feature extractor is VGGFace2 Inception-ResNet-v1.)
    model = InceptionResnetV1(pretrained="vggface2", classify=True)
    model.logits = nn.Linear(512, 2)
    return model.to(device)


def apply_freezing(model, train_groups):
    """Freeze everything except head + `train_groups`; return bookkeeping."""
    train_module_names = set([HEAD_MODULE])
    for g in train_groups:
        train_module_names.update(GROUP_MODULES[g])

    frozen_modules, trainable_modules = [], []
    for name, _ in model.named_children():
        trainable = name in train_module_names
        for p in model._modules[name].parameters():
            p.requires_grad = trainable
        (trainable_modules if trainable else frozen_modules).append(name)

    def count(mods):
        return sum(p.numel() for n, p in model.named_parameters()
                   if n.split(".")[0] in mods)

    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    info = {
        "train_groups": train_groups,
        "frozen_modules": frozen_modules,
        "trainable_modules": trainable_modules,
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
        "total_params": int(total_params),
        "trainable_fraction": float(trainable_params / total_params),
    }
    return info, set(frozen_modules)


def set_modes(model, frozen_modules):
    """Train mode, but frozen top-level modules in eval (freeze BN stats)."""
    model.train()
    for name in frozen_modules:
        getattr(model, name).eval()


def make_optimizer(model, cfg):
    head, backbone = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if n.split(".")[0] == HEAD_MODULE else backbone).append(p)
    groups = [{"params": head, "lr": cfg["lr_head"]}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg["lr_backbone"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None,
              frozen_modules=None):
    train = optimizer is not None
    if train:
        set_modes(model, frozen_modules)
    else:
        model.eval()
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
    return m, preds, trues


def train_config(config_name, model, train_loader, val_loader, device, cfg):
    info, frozen = apply_freezing(model, CONFIG_TRAIN_GROUPS[config_name])
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"], eta_min=cfg["lr_backbone"] * 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    history = []
    best = {"f1": -1.0, "loss": 1e9, "epoch": -1, "state": None}
    patience_ctr = 0
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        te = time.time()
        tr, _, _ = run_epoch(model, train_loader, criterion, device,
                             optimizer=opt, frozen_modules=frozen)
        va, vpred, vtrue = run_epoch(model, val_loader, criterion, device,
                                     frozen_modules=frozen)
        sched.step()
        row = {"epoch": ep, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
               "val_loss": va["loss"], "val_acc": va["accuracy"],
               "val_f1": va["f1"], "sec": time.time() - te}
        history.append(row)
        print(f"  [{config_name}] ep{ep:02d} "
              f"tr_loss={tr['loss']:.4f} tr_acc={tr['accuracy']:.3f} | "
              f"va_loss={va['loss']:.4f} va_acc={va['accuracy']:.3f} "
              f"va_f1={va['f1']:.3f} ({row['sec']:.1f}s)")

        # selection rule: highest val F1, ties broken by lower val loss
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
            print(f"  [{config_name}] early stop at epoch {ep}")
            break
    total_time = time.time() - t0

    # reload best weights & compute final val metrics + probabilities
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

    ckpt = os.path.join(MODELS, f"facenet_{config_name}.pt")
    torch.save({"config": config_name, "state_dict": best["state"],
                "freeze_info": info, "hyperparams": cfg, "epoch": best["epoch"]},
               ckpt)

    result = {
        "model": "FaceNet-InceptionResNetV1",
        "config": config_name,
        "seed": SEED,
        "device": describe_device(),
        "freeze_info": info,
        "hyperparams": cfg,
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_f1": best["f1"],
        "train_time_sec": total_time,
        "final_val": final,
        "checkpoint": os.path.relpath(ckpt, RESULTS),
    }
    return result, vtrue, vpred, prob


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["F0", "F1", "F2"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"[env] {describe_device()}")

    index = manifest_index()
    split = load_or_make_split()
    train_tf = facenet_transforms(train=True)
    val_tf = facenet_transforms(train=False)
    train_ds = FaceDataset(split["train"], index, train_tf)
    val_ds = FaceDataset(split["val"], index, val_tf)
    g = loader_generator()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        worker_init_fn=seed_worker, generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    base_cfg = {
        "optimizer": "AdamW",
        "lr_head": 1e-3,
        "lr_backbone": 2e-4,
        "weight_decay": 1e-4,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "label_smoothing": 0.05,
        "scheduler": "CosineAnnealingLR",
        "augmentation": "hflip+affine(10deg,0.05,0.9-1.1)+colorjitter(0.2)",
        "normalization": "mean=0.5,std=0.5 (facenet fixed_image_standardization)",
        "input_size": 160,
        "selection_rule": "max val F1, tie-break lower val loss",
    }

    all_results = []
    for cfg_name in args.configs:
        print(f"\n=== FaceNet {cfg_name} ===")
        model = build_model(device)
        res, vtrue, vpred, prob = train_config(
            cfg_name, model, train_loader, val_loader, device, base_cfg)
        all_results.append(res)
        pred_rows = [{"file": split["val"][i], "true": int(vtrue[i]),
                      "pred": int(vpred[i]), "prob_female": float(prob[i]),
                      "correct": int(vtrue[i] == vpred[i])}
                     for i in range(len(vtrue))]
        save_json(pred_rows,
                  os.path.join(PREDICTIONS, f"facenet_{cfg_name}_val.json"))
        print(f"  -> val_acc={res['final_val']['accuracy']:.3f} "
              f"val_f1={res['final_val']['f1']:.3f} best_ep={res['best_epoch']}")

    save_json(all_results, os.path.join(RESULTS, "facenet_results.json"))
    print(f"\n[out] {os.path.join(RESULTS, 'facenet_results.json')}")


if __name__ == "__main__":
    main()
