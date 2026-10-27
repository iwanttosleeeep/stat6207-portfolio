#!/usr/bin/env python3
"""
05_inference.py
===============
Reusable, read-only inference pipeline for STAT6207 Assignment 3.

It loads the Phase-2 checkpoints, rebuilds the exact models, applies the SAME
preprocessing used during training (MTCNN detect + 5-point alignment -> 160x160,
then model-specific normalisation), and exports the Lecture-6 JSON format:

    {"aaa.jpg": "M", "bbb.jpg": "F"}          # filename -> M / F

Label mapping (identical to training): index 0 -> "M" (male), 1 -> "F" (female).

Selected models (from results/comparison.json, rule = highest val F1):
    FaceNet : F0   (results/models/facenet_F0.pt)   val F1 = 0.884
    CNN     : cnn_small (results/models/cnn_small.pt) val F1 = 0.714
`best_predictions.json` always uses FaceNet F0 (per the assignment spec).

Modes
-----
  # 1) verification without any external data (reproduces Phase-2 val preds)
  python 05_inference.py --verify-val

  # 2) run on the instructor's test zip (raw photos: detect + align)
  python 05_inference.py --zip /path/to/test.zip

  # 3) run on a directory of already-aligned faces
  python 05_inference.py --input-dir /path/to/faces --detect off

Safety
------
* Never writes inside face_gender_dataset/.  Zips are extracted to a temp dir.
* No ground-truth is read or invented; no accuracy/F1 is reported here.
* Any image that cannot be processed is reported explicitly (never silently
  dropped) in results/inference/inference_report.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common import (CLASS_NAMES, MODELS, RESULTS, get_device,
                    cnn_transforms, facenet_transforms)

INFER_DIR = os.path.join(RESULTS, "inference")
os.makedirs(INFER_DIR, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# training label order: 0 = male, 1 = female
LABEL_TO_STR = {0: "M", 1: "F"}

# canonical 5-point ArcFace/FaceNet template scaled to 160 (matches
# face_gender_dataset/scripts/02_prepare_dataset.py exactly)
_BASE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                  [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
REFERENCE = _BASE * (160 / 112.0)


# --------------------------------------------------------------------------- #
# Model selection (from saved experiment records)
# --------------------------------------------------------------------------- #
def select_models():
    path = os.path.join(RESULTS, "comparison.json")
    with open(path) as f:
        comp = json.load(f)

    def best(family):
        cands = [m for m in comp["models"] if family in m["family"]]
        return max(cands, key=lambda m: m["val"]["f1"])

    facenet = best("FaceNet")
    cnn = best("cnn")
    return {
        "facenet": {"name": facenet["name"],
                    "ckpt": os.path.join(RESULTS, facenet.get("checkpoint", "")
                                         or f"models/facenet_{facenet['name']}.pt"),
                    "val_f1": facenet["val"]["f1"]},
        "cnn": {"name": cnn["name"],
                "ckpt": os.path.join(RESULTS, cnn.get("checkpoint", "")
                                     or f"models/{cnn['name']}.pt"),
                "val_f1": cnn["val"]["f1"]},
    }


# --------------------------------------------------------------------------- #
# Face detection + alignment (consistent with dataset preparation)
# --------------------------------------------------------------------------- #
def build_aligner():
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device="cpu", min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7], post_process=False)


def detect_and_align(pil_rgb, mtcnn):
    """Return (aligned 160x160 PIL, meta) or (None, meta) if no face."""
    rgb = np.asarray(pil_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    boxes, probs, lms = mtcnn.detect(pil_rgb, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None, {"reason": "no_face_detected", "n_faces": 0}

    cx, cy = w / 2.0, h / 2.0
    cands = []
    for b, p, l in zip(boxes, probs, lms):
        if l is None:
            continue
        cands.append((float(p), np.asarray(b, dtype=np.float32),
                      np.asarray(l, dtype=np.float32)))
    if not cands:
        return None, {"reason": "no_landmarks", "n_faces": len(boxes)}

    # prefer confident, centre-closest face (as in training); else largest box
    conf = [c for c in cands if c[0] >= 0.90]
    if conf:
        chosen = min(conf, key=lambda c: ((c[1][0] + c[1][2]) / 2 - cx) ** 2 +
                     ((c[1][1] + c[1][3]) / 2 - cy) ** 2)
    else:
        chosen = max(cands, key=lambda c: (c[1][2] - c[1][0]) *
                     (c[1][3] - c[1][1]))
    prob, box, lm = chosen

    M, _ = cv2.estimateAffinePartial2D(lm, REFERENCE, method=cv2.LMEDS)
    if M is None:
        return None, {"reason": "align_failed", "n_faces": len(boxes)}
    aligned = cv2.warpAffine(rgb, M, (160, 160), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    meta = {"reason": "ok", "n_faces": int(len(boxes)),
            "mtcnn_prob": round(prob, 4), "method": "mtcnn_align"}
    return Image.fromarray(aligned), meta


def load_image_key(path):
    with Image.open(path) as im:
        return im.convert("RGB")


# --------------------------------------------------------------------------- #
# Model construction / loading
# --------------------------------------------------------------------------- #
def load_facenet(ckpt_path, device):
    from facenet_pytorch import InceptionResnetV1
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = InceptionResnetV1(pretrained=None, classify=True, num_classes=2)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def _load_cnn_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "03_train_cnn.py")
    spec = importlib.util.spec_from_file_location("train_cnn_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cnn(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mod = _load_cnn_module()
    model = mod.build_cnn(ck["arch"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_batch(model, tensors, device):
    x = torch.stack(tensors).to(device)
    prob = torch.softmax(model(x), dim=1).cpu().numpy()
    idx = prob.argmax(1)
    return idx.tolist(), prob[:, 1].tolist()


def preprocess(pil_img, kind, mean_std):
    if kind == "facenet":
        return facenet_transforms(train=False)(pil_img)
    return cnn_transforms(mean_std[0], mean_std[1], train=False)(pil_img)


# --------------------------------------------------------------------------- #
# Input enumeration
# --------------------------------------------------------------------------- #
def enumerate_inputs(args):
    """Return (root, [(key, abs_path), ...]).  Keys are unique basenames,
    falling back to relative paths if basenames collide."""
    if args.zip:
        root = tempfile.mkdtemp(prefix="a3_test_")
        with zipfile.ZipFile(args.zip) as z:
            for m in z.namelist():
                # zip-slip guard
                dest = os.path.realpath(os.path.join(root, m))
                if not dest.startswith(os.path.realpath(root)):
                    raise SystemExit(f"unsafe path in zip: {m}")
            z.extractall(root)
    elif args.input_dir:
        root = os.path.abspath(args.input_dir)
    else:
        raise SystemExit("provide --zip, --input-dir, or --verify-val")

    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(dp, fn))
    files.sort()

    basenames = [os.path.basename(p) for p in files]
    if len(set(basenames)) == len(basenames):
        return root, list(zip(basenames, files))
    # collision -> use path relative to root
    return root, [(os.path.relpath(p, root).replace(os.sep, "/"), p)
                  for p in files]


# --------------------------------------------------------------------------- #
# Core run
# --------------------------------------------------------------------------- #
def run(inputs, args, models, detect_mode):
    device = get_device()
    mtcnn = build_aligner() if detect_mode != "off" else None
    mean_std = None
    if any(not k.startswith("facenet") for k in models):
        with open(os.path.join(RESULTS, "cnn_norm_stats.json")) as f:
            ns = json.load(f)
        mean_std = (ns["mean"], ns["std"])

    outputs = {name: {} for name in models}          # name -> {file: M/F}
    records = []
    unprocessable = []

    for key, path in inputs:
        rec = {"file": key, "source": os.path.basename(path)}
        try:
            img = load_image_key(path)
        except Exception as e:                       # unreadable file
            rec.update({"status": "unreadable", "error": str(e)})
            unprocessable.append(rec)
            records.append(rec)
            continue

        if detect_mode == "off":
            crop, method = img, "as_is"
            meta = {"reason": "ok", "method": "no_detection"}
            off = False
        else:
            crop, meta = detect_and_align(img, mtcnn)
            method = meta.get("method", meta.get("reason"))
            off = crop is None
            if off and detect_mode == "auto":
                crop, method = img, "center_crop_fallback"
                off = False

        if off:
            rec.update({"status": "unprocessable", **meta})
            unprocessable.append(rec)
            records.append(rec)
            continue

        # batch both models on the same crop
        for name, (model, _ck) in models.items():
            kind = "facenet" if name.startswith("facenet") else "cnn"
            t = preprocess(crop, kind, mean_std)
            idx, p_f = predict_batch(model, [t], device)
            outputs[name][key] = LABEL_TO_STR[idx[0]]
            rec[f"{name}_prob_female"] = round(float(p_f[0]), 4)
        rec.update({"status": "ok", "method": method, **{
            k: v for k, v in meta.items() if k not in ("method",)}})
        records.append(rec)
        if method == "center_crop_fallback":
            rec["status"] = "ok_fallback"

    return outputs, records, unprocessable


# --------------------------------------------------------------------------- #
# Validation of outputs
# --------------------------------------------------------------------------- #
def validate_outputs(inputs, outputs):
    input_keys = [k for k, _ in inputs]
    report = {"n_inputs": len(input_keys), "checks": {}}
    ok = True
    for name, mapping in outputs.items():
        keys = list(mapping.keys())
        missing = sorted(set(input_keys) - set(keys))
        extra = sorted(set(keys) - set(input_keys))
        dups = len(keys) != len(set(keys))
        bad_labels = sorted({v for v in mapping.values() if v not in ("M", "F")})
        c = {"n_predictions": len(keys), "missing_filenames": missing,
             "unexpected_filenames": extra, "duplicate_keys": dups,
             "invalid_labels": bad_labels}
        c["passed"] = (not missing and not extra and not dups and not bad_labels)
        report["checks"][name] = c
        ok = ok and c["passed"]
    report["passed"] = ok
    return report


# --------------------------------------------------------------------------- #
def save_outputs(outputs, best_name, report, unprocessable, tag=""):
    suffix = f"_{tag}" if tag else ""
    for name, mapping in outputs.items():
        if name.startswith("facenet"):
            fn = "facenet_predictions.json"
        else:
            fn = "cnn_predictions.json"
        with open(os.path.join(INFER_DIR, fn), "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"[out] {fn}: {len(mapping)} entries")

        # `best_predictions.json` mirrors the selected best model (FaceNet F0)
        if name == best_name:
            with open(os.path.join(INFER_DIR, "best_predictions.json"), "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"[out] best_predictions.json: {len(mapping)} entries "
                  f"(copy of {name})")

    full_report = {
        "tag": tag,
        "label_map": LABEL_TO_STR,
        "best_predictions_model": best_name,
        "n_unprocessable": len(unprocessable),
        "unprocessable": unprocessable,
        "validation": report,
    }
    rp = os.path.join(INFER_DIR, f"inference_report{suffix}.json")
    with open(rp, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"[out] {os.path.basename(rp)}")
    return rp


# --------------------------------------------------------------------------- #
def cmd_verify_val(args):
    """Run inference on the internal validation split and compare with the
    predictions saved in Phase 2.  This validates the plumbing end-to-end."""
    from common import load_or_make_split, manifest_index, face_path
    split = load_or_make_split()
    index = manifest_index()
    inputs = [(n, face_path(index[n])) for n in split["val"]]

    sel = select_models()
    device = get_device()
    models = {}
    fmodel, _ = load_facenet(sel["facenet"]["ckpt"], device)
    models[f"facenet_{sel['facenet']['name']}"] = (fmodel, None)
    cmodel, _ = load_cnn(sel["cnn"]["ckpt"], device)
    models[sel["cnn"]["name"]] = (cmodel, None)

    # detect off: images are already aligned 160x160
    outputs, records, unproc = run(inputs, args, models, detect_mode="off")
    report = validate_outputs(inputs, outputs)

    # compare to Phase-2 saved validation predictions
    comparisons = {}
    for name, mapping in outputs.items():
        pf = os.path.join(RESULTS, "predictions", f"{name}_val.json")
        if os.path.exists(pf):
            saved = {r["file"]: LABEL_TO_STR[r["pred"]]
                     for r in json.load(open(pf))}
            match = sum(1 for k in mapping if saved.get(k) == mapping[k])
            comparisons[name] = {"matched": match, "total": len(mapping),
                                 "agreement": match / len(mapping)}
    print(f"[verify] validation outputs: {report['passed']}")
    for name, c in comparisons.items():
        print(f"[verify] {name}: {c['matched']}/{c['total']} "
              f"({c['agreement']:.3f}) agree with Phase-2 val predictions")
    save_outputs(outputs, f"facenet_{sel['facenet']['name']}", report,
                 unproc, tag="verify_val")


def main():
    ap = argparse.ArgumentParser(description="A3 reusable inference pipeline")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--zip", help="instructor test zip")
    src.add_argument("--input-dir", help="directory of images")
    src.add_argument("--verify-val", action="store_true",
                     help="run on internal val split (no external data)")
    ap.add_argument("--detect", choices=["auto", "on", "off"], default="auto",
                    help="face detect+align: auto (detect, fallback to as-is), "
                         "on (detect only), off (already-aligned crops)")
    args = ap.parse_args()

    if args.verify_val:
        cmd_verify_val(args)
        return

    sel = select_models()
    print(f"[select] FaceNet={sel['facenet']['name']} "
          f"(val F1={sel['facenet']['val_f1']:.3f})  "
          f"CNN={sel['cnn']['name']} (val F1={sel['cnn']['val_f1']:.3f})")

    device = get_device()
    fmodel, _ = load_facenet(sel["facenet"]["ckpt"], device)
    cmodel, _ = load_cnn(sel["cnn"]["ckpt"], device)
    best_name = f"facenet_{sel['facenet']['name']}"
    models = {best_name: (fmodel, None), sel["cnn"]["name"]: (cmodel, None)}

    root, inputs = enumerate_inputs(args)
    print(f"[input] {len(inputs)} images from {root}")
    outputs, records, unproc = run(inputs, args, models, args.detect)

    report = validate_outputs(inputs, outputs)
    save_outputs(outputs, best_name, report, unproc)

    with open(os.path.join(INFER_DIR, "inference_records.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"[validate] passed={report['passed']}")
    for name, c in report["checks"].items():
        print(f"  {name}: n={c['n_predictions']} missing={len(c['missing_filenames'])} "
              f"extra={len(c['unexpected_filenames'])} dups={c['duplicate_keys']} "
              f"invalid_labels={c['invalid_labels']}")
    if unproc:
        print(f"[WARN] {len(unproc)} image(s) could not be processed "
              f"(see inference_report.json): "
              f"{[u['file'] for u in unproc][:20]}")
    else:
        print("[validate] every input produced exactly one prediction")

    if args.zip:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
