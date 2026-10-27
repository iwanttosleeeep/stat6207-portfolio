#!/usr/bin/env python3
"""
common.py
=========
Shared utilities for the STAT6207 Assignment 3 training experiments.

Nothing in this module writes to the prepared dataset in
`face_gender_dataset/`; it only reads images + manifest and writes all
artefacts under `experiments/results/`.

Conventions
-----------
* label 0 = male, label 1 = female (matches FairFace gender and the
  `faces/{male,female}` folder order).
* Random seed is fixed (SEED = 42) everywhere.
* By default training uses 80% / validation 20%, stratified by class.
* FaceNet inputs are normalized exactly like facenet-pytorch's
  `fixed_image_standardization` -> [-1, 1] (mean 0.5, std 0.5 on [0,1]).
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # Assignment 3/
DATASET = os.path.join(ROOT, "face_gender_dataset")
FACES = {0: os.path.join(DATASET, "faces", "male"),
         1: os.path.join(DATASET, "faces", "female")}
MANIFEST = os.path.join(DATASET, "dataset_manifest.csv")
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(RESULTS, "models")
PLOTS = os.path.join(RESULTS, "plots")
LOGS = os.path.join(RESULTS, "logs")
PREDICTIONS = os.path.join(RESULTS, "predictions")
SPLIT_PATH = os.path.join(RESULTS, "split.json")

for _d in (RESULTS, MODELS, PLOTS, LOGS, PREDICTIONS):
    os.makedirs(_d, exist_ok=True)

SEED = 42
IMG_SIZE = 160
VAL_FRAC = 0.20
CLASS_NAMES = ["male", "female"]

# --------------------------------------------------------------------------- #
# Reproducibility & device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """MPS -> CUDA -> CPU, in that order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device() -> dict:
    dev = get_device()
    info = {"device": str(dev), "torch": torch.__version__}
    if dev.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    if dev.type == "mps":
        info["gpu"] = "Apple MPS"
    return info


def seed_worker(worker_id: int) -> None:  # for DataLoader worker_init_fn
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def loader_generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Manifest / dataset
# --------------------------------------------------------------------------- #
def load_manifest():
    """Return the manifest as a list of dicts, sorted by filename."""
    import csv
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["label"], r["image_filename"]))
    return rows


def manifest_index():
    """filename -> manifest row."""
    return {r["image_filename"]: r for r in load_manifest()}


def face_path(row) -> str:
    return os.path.join(FACES[0 if row["label"] == "male" else 1],
                        row["image_filename"])


def all_samples():
    """List of (filename, label_int, path) in deterministic order."""
    out = []
    for r in load_manifest():
        label = 0 if r["label"] == "male" else 1
        out.append((r["image_filename"], label, face_path(r)))
    return out


# --------------------------------------------------------------------------- #
# Split (reproducible, created once, reused by every experiment)
# --------------------------------------------------------------------------- #
def make_stratified_split(val_frac: float = VAL_FRAC, seed: int = SEED) -> dict:
    from sklearn.model_selection import StratifiedShuffleSplit
    samples = all_samples()
    names = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    split = {
        "seed": seed,
        "val_frac": val_frac,
        "n_total": len(samples),
        "train": [names[i] for i in sorted(train_idx)],
        "val": [names[i] for i in sorted(val_idx)],
    }
    return split


def load_or_make_split(path: str = SPLIT_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    split = make_stratified_split()
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    return split


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def facenet_transforms(train: bool):
    """[-1,1] normalization + light augmentation (horizontal flip is safe for
    gender; small affine / colour jitter add robustness on the tiny set)."""
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                           scale=(0.90, 1.10)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def cnn_transforms(mean, std, train: bool):
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=12, translate=(0.06, 0.06),
                           scale=(0.88, 1.12)),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, names, index, transform):
        self.names = list(names)
        self.index = index
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        row = self.index[name]
        label = 0 if row["label"] == "male" else 1
        img = Image.open(face_path(row)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, i


def compute_mean_std(names, index):
    """Per-channel mean/std of the training pixels (for from-scratch CNN)."""
    import torchvision.transforms.functional as F
    sum_ = np.zeros(3)
    sq = np.zeros(3)
    n = 0
    for name in names:
        img = Image.open(face_path(index[name])).convert("RGB")
        a = np.asarray(img, dtype=np.float64) / 255.0
        sum_ += a.reshape(-1, 3).sum(0)
        sq += (a.reshape(-1, 3) ** 2).sum(0)
        n += a.shape[0] * a.shape[1]
    mean = sum_ / n
    std = np.sqrt(sq / n - mean ** 2)
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1,
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1,
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": cm,
        "n": int(len(y_true)),
    }


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Stopwatch:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
