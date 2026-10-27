#!/usr/bin/env python3
"""
03_make_contact_sheet.py
========================
Build ../contact_sheet.png: a labelled visual overview of all 200 processed
face images (male block on top, female block below) for manual inspection.
"""
import csv
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(ROOT, "dataset_manifest.csv")
OUT = os.path.join(ROOT, "contact_sheet.png")

CELL = 160
LABEL_H = 22
COLS = 10
ROWS = 10
TITLE_H = 46
SEP = 12


def load_rows():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def draw_block(canvas, y0, rows, title):
    cv2.putText(canvas, title, (10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 2, cv2.LINE_AA)
    yy = y0 + TITLE_H
    for i, r in enumerate(rows):
        c = i % COLS
        row_i = i // COLS
        x = c * CELL
        y = yy + row_i * (CELL + LABEL_H)
        img_path = os.path.join(ROOT, "faces", r["label"], r["image_filename"])
        img = cv2.imread(img_path)
        if img is None:
            img = np.full((CELL, CELL, 3), 200, np.uint8)
        else:
            img = cv2.resize(img, (CELL, CELL))
        canvas[y:y + CELL, x:x + CELL] = img
        cap = f"{r['image_filename'][:12]} a{r['fairface_age_label']}"
        cv2.putText(canvas, cap, (x + 3, y + CELL + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
    return yy + ROWS * (CELL + LABEL_H)


def main():
    rows = load_rows()
    males = [r for r in rows if r["label"] == "male"]
    females = [r for r in rows if r["label"] == "female"]
    block_h = TITLE_H + ROWS * (CELL + LABEL_H)
    W = COLS * CELL
    H = block_h * 2 + SEP
    canvas = np.full((H, W, 3), 255, np.uint8)
    y1 = draw_block(canvas, 0, males, "MALE-labelled (n=%d)" % len(males))
    y2 = draw_block(canvas, y1 + SEP, females,
                    "FEMALE-labelled (n=%d)" % len(females))
    cv2.line(canvas, (0, y1 + SEP // 2), (W, y1 + SEP // 2), (0, 0, 0), 2)
    cv2.imwrite(OUT, canvas)
    print(f"[out ] {OUT}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
