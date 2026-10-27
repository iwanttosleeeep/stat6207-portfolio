#!/usr/bin/env python3
"""
02_prepare_dataset.py
=====================
Build the STAT6207 Assignment 3 training set from the labelled FairFace
"1.25 padding" validation split (see 01_download_fairface.py).

Pipeline (Lectures 5-6)
-----------------------
raw FairFace image (with scene context)
  -> MTCNN face detection (facenet-pytorch)
  -> keep the single intended face (centre-closest, high confidence)
  -> QC (confidence, size, in-frame, eye spacing, valid-pixel fraction)
  -> 5-point landmark similarity-transform alignment
  -> resize to 160x160  (FaceNet / CNN standard input)
  -> exact + perceptual near-duplicate rejection
  -> save raw_images/<id>.<ext>  and  faces/{male,female}/<id>.jpg

Labels
------
We never infer gender from appearance.  Every kept image keeps FairFace's
own "gender" label (0=Male, 1=Female) and its own race label (0=East Asian,
matching the Hong Kong / East-Asian evaluation domain) and age label.

Outputs
-------
../raw_images/            original downloaded (1.25-padded) images
../faces/male/            100 processed male-labelled faces
../faces/female/          100 processed female-labelled faces
../dataset_manifest.csv   full provenance + detection metadata
"""
import csv
import hashlib
import io
import os
import random
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from facenet_pytorch import MTCNN

torch.set_num_threads(max(1, os.cpu_count() or 1))
cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_RAW = os.path.join(ROOT, "data_raw")
RAW_DIR = os.path.join(ROOT, "raw_images")
FACE_DIR = {"male": os.path.join(ROOT, "faces", "male"),
            "female": os.path.join(ROOT, "faces", "female")}
MANIFEST = os.path.join(ROOT, "dataset_manifest.csv")
PARQUET = os.path.join(DATA_RAW, "fairface_125_validation.parquet")

SOURCE_URL = ("https://huggingface.co/datasets/HuggingFaceM4/FairFace/"
              "resolve/main/1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet")
SOURCE_DATASET = "FairFace (HuggingFaceM4/FairFace)"
SOURCE_CONFIG = "1.25/validation"

# ---- FairFace label dictionaries (original dataset labels) -----------------
AGE_LABELS = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
              5: "40-49", 6: "50-59", 7: "60-69", 8: "more than 70"}
GENDER_LABELS = {0: "Male", 1: "Female"}
RACE_LABELS = {0: "East Asian", 1: "Indian", 2: "Black", 3: "White",
               4: "Middle Eastern", 5: "Latino_Hispanic", 6: "Southeast Asian"}

EAST_ASIAN = 0
MALE, FEMALE = 0, 1
N_PER_CLASS = 100
SEED = 42

# Target age distribution per class (FairFace age bins).  Age bin 0 (0-2
# infants) is intentionally excluded from the candidate pool.  The mix keeps
# the set adult-weighted (matches the HK-personality evaluation domain) while
# still spanning ages 3 to 70+.
AGE_TARGET = {1: 4, 2: 8, 3: 29, 4: 24, 5: 20, 6: 10, 7: 3, 8: 2}
assert sum(AGE_TARGET.values()) == N_PER_CLASS

# ---- QC thresholds ---------------------------------------------------------
PROB_MIN = 0.90          # MTCNN detection confidence
MIN_BOX_FRAC = 0.02      # detected face box >= 2% of image area
MAX_OVERFLOW = 0.02      # box may not stick out more than 2% of the image
MIN_EYE_DIST = 20.0      # pixels, original resolution
MIN_VALID_FRAC = 0.90    # >=90% of the aligned 160x160 must come from the image

IMG_SIZE = 160
# canonical 5-point template (ArcFace/FaceNet), scaled to IMG_SIZE
_BASE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                  [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
REFERENCE = _BASE * (IMG_SIZE / 112.0)

# dup detection
DHASH_MAX_HAMMING = 6    # 64-bit dHash; <=6 => near duplicate


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dhash64(gray):
    g = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return val


def hamming(a, b):
    return int(a ^ b).bit_count()


def build_mtcnn():
    return MTCNN(keep_all=True, device="cpu", min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], post_process=False)


def select_face(boxes, probs, lms, w, h):
    """Return (box, prob, landmarks) of the intended face, or None."""
    if boxes is None:
        return None
    cx, cy = w / 2.0, h / 2.0
    best = None
    for b, p, l in zip(boxes, probs, lms):
        if p is None or float(p) < PROB_MIN:
            continue
        bx1, by1, bx2, by2 = [float(v) for v in b]
        # in-frame check
        if bx1 < -MAX_OVERFLOW * w or by1 < -MAX_OVERFLOW * h:
            continue
        if bx2 > w * (1 + MAX_OVERFLOW) or by2 > h * (1 + MAX_OVERFLOW):
            continue
        if (bx2 - bx1) * (by2 - by1) < MIN_BOX_FRAC * w * h:
            continue
        lm = np.asarray(l, dtype=np.float32)
        if np.linalg.norm(lm[0] - lm[1]) < MIN_EYE_DIST:
            continue
        d = ((bx1 + bx2) / 2 - cx) ** 2 + ((by1 + by2) / 2 - cy) ** 2
        if best is None or d < best[0]:
            best = (d, np.array([bx1, by1, bx2, by2]), float(p), lm)
    if best is None:
        return None
    return best[1], best[2], best[3]


def align_face(rgb, landmarks):
    src = np.asarray(landmarks, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, REFERENCE, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(rgb, M, (IMG_SIZE, IMG_SIZE),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
    mw = cv2.warpAffine(mask, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_frac = float((mw > 0).mean())
    if valid_frac < MIN_VALID_FRAC:
        return None
    return aligned, valid_frac


def process_row(row, mtcnn):
    """Decode one FairFace row, detect/align. Returns dict or (None, reason)."""
    raw_bytes = row["image"]["bytes"]
    pil = Image.open(io.BytesIO(raw_bytes))
    fmt = (pil.format or "jpeg").lower()
    ext = "jpg" if fmt in ("jpeg", "jpg") else fmt
    rgb_pil = pil.convert("RGB")
    rgb = np.array(rgb_pil)
    h, w = rgb.shape[:2]

    boxes, probs, lms = mtcnn.detect(rgb_pil, landmarks=True)
    chosen = select_face(boxes, probs, lms, w, h)
    if chosen is None:
        return None, "no_face"
    box, prob, landmarks = chosen

    aligned = align_face(rgb, landmarks)
    if aligned is None:
        return None, "align_fail"
    face, valid_frac = aligned

    n_faces = 0 if boxes is None else len(boxes)
    return {
        "raw_bytes": raw_bytes, "ext": ext, "raw_size": (w, h),
        "face": face, "prob": prob, "box": box, "valid_frac": valid_frac,
        "n_faces": n_faces, "dhash": dhash64(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)),
        "raw_sha": sha256_bytes(raw_bytes),
    }, "ok"


def main():
    for d in [RAW_DIR, FACE_DIR["male"], FACE_DIR["female"]]:
        os.makedirs(d, exist_ok=True)
    # clean previous faces so re-runs are deterministic
    for d in [RAW_DIR, FACE_DIR["male"], FACE_DIR["female"]]:
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                os.remove(os.path.join(d, fn))

    print("[load] FairFace 1.25 validation parquet ...")
    table = pq.read_table(PARQUET)
    df = table.to_pandas()
    ea = df[df["race"] == EAST_ASIAN].reset_index(drop=True)
    print(f"[data] East Asian rows: {len(ea)}")

    # candidate pools keyed by (gender, age_bin), deterministic order
    pools = {}
    for g in (MALE, FEMALE):
        for a in AGE_LABELS:
            subset = ea[(ea["gender"] == g) & (ea["age"] == a)]
            idxs = list(subset.index)
            random.Random(SEED + g * 100 + a).shuffle(idxs)
            pools[(g, a)] = idxs

    mtcnn = build_mtcnn()
    accepted = {MALE: [], FEMALE: []}     # manifest row dicts
    used_dhashes = []                     # global near-dup guard
    used_raw_sha = set()
    used_raw_idx = set()
    exhausted = {g: set() for g in (MALE, FEMALE)}

    def try_accept(g, a, target, attempts_report):
        """Pull from pool (g,a) until `target` accepted or pool exhausted."""
        got = 0
        for idx in pools[(g, a)]:
            if got >= target:
                break
            if idx in used_raw_idx:
                continue
            attempts_report["tried"] += 1
            row = ea.loc[idx]
            res, reason = process_row(row, mtcnn)
            if res is None:
                attempts_report[reason] = attempts_report.get(reason, 0) + 1
                continue
            if res["raw_sha"] in used_raw_sha:
                attempts_report["dup_exact"] = attempts_report.get("dup_exact", 0) + 1
                continue
            if any(hamming(res["dhash"], hh) <= DHASH_MAX_HAMMING for hh in used_dhashes):
                attempts_report["dup_near"] = attempts_report.get("dup_near", 0) + 1
                continue
            used_raw_idx.add(idx)
            used_raw_sha.add(res["raw_sha"])
            used_dhashes.append(res["dhash"])
            rec = {
                "class": "male" if g == MALE else "female",
                "source_row_index": int(idx),
                "fairface_gender": int(row["gender"]),
                "fairface_gender_label": GENDER_LABELS[int(row["gender"])],
                "fairface_race": int(row["race"]),
                "fairface_race_label": RACE_LABELS[int(row["race"])],
                "fairface_age": int(row["age"]),
                "fairface_age_label": AGE_LABELS[int(row["age"])],
                **_res_fields(res),
            }
            accepted[g].append(rec)
            got += 1
        return got

    print("[run ] selecting + detecting faces ...")
    report = {"tried": 0}
    for g in (MALE, FEMALE):
        for a in sorted(AGE_TARGET):
            need = AGE_TARGET[a]
            got = try_accept(g, a, need, report)
            flag = "" if got == need else f"  (short by {need - got})"
            print(f"       {'male' if g==MALE else 'female':6s} age {AGE_LABELS[a]:>11s}"
                  f"  target {need:3d}  got {got:3d}{flag}")
            if got < need:
                exhausted[g].add(a)

    # fallback: fill any shortfall from remaining candidates of that gender
    for g in (MALE, FEMALE):
        while len(accepted[g]) < N_PER_CLASS:
            need = N_PER_CLASS - len(accepted[g])
            # remaining candidates of this gender, adult-first
            remaining = [i for i in ea.index[ea["gender"] == g]
                         if i not in used_raw_idx]
            remaining.sort(key=lambda i: (abs(int(ea.loc[i, "age"]) - 4), i))
            got = 0
            for idx in remaining:
                if got >= need:
                    break
                report["tried"] += 1
                row = ea.loc[idx]
                res, reason = process_row(row, mtcnn)
                if res is None:
                    report[reason] = report.get(reason, 0) + 1
                    continue
                if res["raw_sha"] in used_raw_sha:
                    continue
                if any(hamming(res["dhash"], hh) <= DHASH_MAX_HAMMING for hh in used_dhashes):
                    continue
                used_raw_idx.add(idx); used_raw_sha.add(res["raw_sha"])
                used_dhashes.append(res["dhash"])
                rec = {
                    "class": "male" if g == MALE else "female",
                    "source_row_index": int(idx),
                    "fairface_gender": int(row["gender"]),
                    "fairface_gender_label": GENDER_LABELS[int(row["gender"])],
                    "fairface_race": int(row["race"]),
                    "fairface_race_label": RACE_LABELS[int(row["race"])],
                    "fairface_age": int(row["age"]),
                    "fairface_age_label": AGE_LABELS[int(row["age"])],
                    **_res_fields(res),
                }
                accepted[g].append(rec); got += 1

    # ---- save + manifest ---------------------------------------------------
    rows = []
    for g in (MALE, FEMALE):
        cls = "male" if g == MALE else "female"
        for n, rec in enumerate(accepted[g], start=1):
            stem = f"{cls}_{n:04d}"
            raw_name = f"{stem}.{rec['_ext']}"
            face_name = f"{stem}.jpg"
            with open(os.path.join(RAW_DIR, raw_name), "wb") as f:
                f.write(rec["_raw_bytes"])
            face = rec["_face"]
            face_path = os.path.join(FACE_DIR[cls], face_name)
            cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(_manifest_row(stem, cls, raw_name, face_name, face_path, rec))

    cols = list(rows[0].keys())
    with open(MANIFEST, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)

    rejects = {k: v for k, v in report.items() if k != "tried"}
    print()
    print(f"[done] male={len(accepted[MALE])}  female={len(accepted[FEMALE])}")
    print(f"[stat] images processed: {report['tried']}  rejected: {rejects}")
    print(f"[out ] {MANIFEST}")


# helper plumbing so process_row results can be stashed on the record
def _res_fields(res):
    return {
        "_raw_bytes": res["raw_bytes"], "_ext": res["ext"], "_face": res["face"],
        "raw_width": res["raw_size"][0], "raw_height": res["raw_size"][1],
        "n_faces_detected": res["n_faces"], "mtcnn_prob": round(res["prob"], 4),
        "box_x1": round(float(res["box"][0]), 1), "box_y1": round(float(res["box"][1]), 1),
        "box_x2": round(float(res["box"][2]), 1), "box_y2": round(float(res["box"][3]), 1),
        "align_valid_frac": round(res["valid_frac"], 4),
        "raw_sha256": res["raw_sha"], "dhash64": res["dhash"],
    }


def _manifest_row(stem, cls, raw_name, face_name, face_path, rec):
    with open(face_path, "rb") as f:
        face_bytes = f.read()
    return {
        "image_filename": face_name,
        "raw_filename": raw_name,
        "label": cls,
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_config": SOURCE_CONFIG,
        "source_row_index": rec["source_row_index"],
        "fairface_gender": rec["fairface_gender"],
        "fairface_gender_label": rec["fairface_gender_label"],
        "fairface_race": rec["fairface_race"],
        "fairface_race_label": rec["fairface_race_label"],
        "fairface_age": rec["fairface_age"],
        "fairface_age_label": rec["fairface_age_label"],
        "mtcnn_prob": rec["mtcnn_prob"],
        "n_faces_detected": rec["n_faces_detected"],
        "box_x1": rec["box_x1"], "box_y1": rec["box_y1"],
        "box_x2": rec["box_x2"], "box_y2": rec["box_y2"],
        "raw_width": rec["raw_width"], "raw_height": rec["raw_height"],
        "processed_width": IMG_SIZE, "processed_height": IMG_SIZE,
        "align_method": "5pt_similarity_affine",
        "align_valid_frac": rec["align_valid_frac"],
        "raw_sha256": rec["raw_sha256"],
        "face_sha256": sha256_bytes(face_bytes),
        "dhash64": rec["dhash64"],
    }


if __name__ == "__main__":
    main()
