#!/usr/bin/env python3
"""
01_make_split.py
================
Create the single, reproducible, stratified 80/20 train/validation split used
by *every* experiment, then audit it for duplicate / identity leakage.

Checks
------
* class balance per split
* exact-duplicate leakage: shared `face_sha256` between train and val
* near-duplicate leakage: minimum Hamming distance of the 64-bit `dhash64`
  between every val image and the closest train image
* domain metadata (age / detection confidence) distribution per split

Writes
------
results/split.json        the split (filenames)
results/split_audit.json  leakage + composition audit
"""
import os

from common import (PLOTS, RESULTS, SEED, VAL_FRAC, load_or_make_split,
                    load_manifest, save_json)

import numpy as np


def hamming(a, b):
    return int(a ^ b).bit_count()


def main():
    rows = {r["image_filename"]: r for r in load_manifest()}
    split = load_or_make_split()
    train, val = split["train"], split["val"]

    def dist(names):
        c = {0: 0, 1: 0, "age": {}}
        for n in names:
            lab = 0 if rows[n]["label"] == "male" else 1
            c[lab] += 1
            age = rows[n]["fairface_age_label"]
            c["age"][age] = c["age"].get(age, 0) + 1
        return c

    train_d, val_d = dist(train), dist(val)

    # ---- exact-duplicate leakage ------------------------------------------
    train_sha = {rows[n]["face_sha256"] for n in train}
    val_sha = {rows[n]["face_sha256"] for n in val}
    exact_overlap = sorted(train_sha & val_sha)

    # ---- near-duplicate (dHash) leakage -----------------------------------
    train_dh = [(n, int(rows[n]["dhash64"])) for n in train]
    val_dh = [(n, int(rows[n]["dhash64"])) for n in val]
    min_per_val = []
    for vn, vh in val_dh:
        best = min(hamming(vh, th) for _, th in train_dh)
        min_per_val.append((vn, best))
    min_d = min(d for _, d in min_per_val)
    close_pairs = sorted([p for p in min_per_val if p[1] <= 6],
                         key=lambda p: p[1])

    # ---- metadata distribution --------------------------------------------
    def meta(names):
        probs = [float(rows[n]["mtcnn_prob"]) for n in names]
        return {"mtcnn_prob_min": float(np.min(probs)),
                "mtcnn_prob_mean": float(np.mean(probs))}

    audit = {
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "n_train": len(train),
        "n_val": len(val),
        "train_class_counts": {"male": train_d[0], "female": train_d[1]},
        "val_class_counts": {"male": val_d[0], "female": val_d[1]},
        "train_age_hist": train_d["age"],
        "val_age_hist": val_d["age"],
        "train_meta": meta(train),
        "val_meta": meta(val),
        "exact_duplicate_overlap": exact_overlap,
        "min_val_to_train_dhash_hamming": int(min_d),
        "near_duplicate_pairs_hamming_le_6": close_pairs,
        "leakage_verdict": ("clean" if (not exact_overlap and min_d > 6)
                            else "LEAKAGE DETECTED"),
    }
    save_json(audit, os.path.join(RESULTS, "split_audit.json"))

    print(f"Split: {len(train)} train / {len(val)} val  (seed={SEED})")
    print(f"  train male/female = {train_d[0]}/{train_d[1]}")
    print(f"  val   male/female = {val_d[0]}/{val_d[1]}")
    print(f"  exact duplicate overlap : {len(exact_overlap)}")
    print(f"  min val->train dHash HD : {min_d}  (<=6 would be leakage)")
    print(f"  verdict                 : {audit['leakage_verdict']}")
    print(f"[out] {os.path.join(RESULTS, 'split_audit.json')}")


if __name__ == "__main__":
    main()
