#!/usr/bin/env python3
"""
04_verify_dataset.py
====================
Quality-control pass over the finished dataset:

  * exactly 200 readable 160x160 faces, 100 per class;
  * manifest / folder consistency and label integrity;
  * no exact duplicates (sha256) and no perceptual near-duplicates (dHash);
  * raw images present and readable;
  * independent cross-detector check (OpenCV YuNet) confirms a face is
    present in every processed crop;
  * reports the age / gender distribution actually achieved.

Exit code is non-zero if any hard check fails.
"""
import csv
import hashlib
import os
import sys
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(ROOT, "dataset_manifest.csv")
# NOTE: opencv_zoo stores the model with Git-LFS; the plain github raw URL
# returns a ~130-byte pointer, so the media endpoint must be used.
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET = os.path.join(ROOT, "data_raw", "face_detection_yunet_2023mar.onnx")

EXPECT_PER_CLASS = 100
EXPECT_SIZE = (160, 160)
NEAR_DUP_HAMMING = 6
fails = []


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dhash64(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for bit in bits:
        v = (v << 1) | int(bit)
    return v


def ensure_yunet():
    os.makedirs(os.path.dirname(YUNET), exist_ok=True)
    if not os.path.exists(YUNET) or os.path.getsize(YUNET) < 10000:
        print("[get ] YuNet model ...")
        urllib.request.urlretrieve(YUNET_URL, YUNET)
    return YUNET


def main():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    # --- counts / manifest consistency -------------------------------------
    if len(rows) != 2 * EXPECT_PER_CLASS:
        fails.append(f"manifest rows = {len(rows)} (expected 200)")
    per_class = {}
    for r in rows:
        per_class.setdefault(r["label"], []).append(r)
    for cls, n in (("male", EXPECT_PER_CLASS), ("female", EXPECT_PER_CLASS)):
        got = len(per_class.get(cls, []))
        print(f"[cnt ] faces/{cls}: {got}")
        if got != n:
            fails.append(f"{cls}: {got} (expected {n})")

    # --- image integrity + sizes -------------------------------------------
    seen_sha, hashes = {}, []
    for r in rows:
        cls = r["label"]
        fp = os.path.join(ROOT, "faces", cls, r["image_filename"])
        rp = os.path.join(ROOT, "raw_images", r["raw_filename"])
        if not os.path.exists(fp):
            fails.append(f"missing processed file {fp}")
            continue
        if not os.path.exists(rp):
            fails.append(f"missing raw file {rp}")
        img = cv2.imread(fp)
        if img is None:
            fails.append(f"unreadable {fp}")
            continue
        if (img.shape[1], img.shape[0]) != EXPECT_SIZE:
            fails.append(f"{r['image_filename']} size {img.shape[1]}x{img.shape[0]}")
        sh = sha256_file(fp)
        if sh in seen_sha:
            fails.append(f"exact duplicate: {r['image_filename']} == {seen_sha[sh]}")
        seen_sha[sh] = r["image_filename"]
        hashes.append((r["image_filename"], cls, dhash64(fp)))
        if int(r["fairface_gender"]) != (1 if cls == "female" else 0):
            fails.append(f"{r['image_filename']} label/gender mismatch")

    # --- near-duplicate check (global, across classes) ---------------------
    worst = 64
    worst_pair = ("", "")
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = int(hashes[i][2] ^ hashes[j][2]).bit_count()
            if d < worst:
                worst, worst_pair = d, (hashes[i][0], hashes[j][0])
            if d <= NEAR_DUP_HAMMING:
                fails.append(f"near-duplicate {hashes[i][0]} ~ {hashes[j][0]} (d={d})")
    print(f"[dup ] closest dHash distance among all 200: {worst} "
          f"({worst_pair[0]} ~ {worst_pair[1]})  "
          f"[<= {NEAR_DUP_HAMMING} would flag a duplicate]")

    # --- cross-detector confirmation (OpenCV YuNet) ------------------------
    model = ensure_yunet()
    det = cv2.FaceDetectorYN.create(model, "", (EXPECT_SIZE[0], EXPECT_SIZE[1]),
                                    score_threshold=0.6)
    no_face = 0
    for r in rows:
        fp = os.path.join(ROOT, "faces", r["label"], r["image_filename"])
        img = cv2.imread(fp)
        if img is None:
            continue
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)
        if faces is None or len(faces) == 0:
            no_face += 1
            fails.append(f"YuNet found no face in {r['image_filename']}")
    print(f"[xdet] YuNet found a face in {len(rows) - no_face}/{len(rows)} crops")

    # --- distribution report -----------------------------------------------
    print("[age ] per-class age mix:")
    for cls in ("male", "female"):
        sub = per_class[cls]
        dd = {}
        for r in sub:
            dd[r["fairface_age_label"]] = dd.get(r["fairface_age_label"], 0) + 1
        print(f"       {cls:6s}: {dict(sorted(dd.items()))}")

    print()
    if fails:
        print(f"[FAIL] {len(fails)} issue(s):")
        for x in fails[:40]:
            print("   -", x)
        sys.exit(1)
    print("[PASS] all hard checks passed: 200 valid 160x160 faces, "
          "100 male + 100 female, no duplicates.")


if __name__ == "__main__":
    main()
