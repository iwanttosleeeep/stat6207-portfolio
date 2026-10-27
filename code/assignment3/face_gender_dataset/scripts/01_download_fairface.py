#!/usr/bin/env python3
"""
01_download_fairface.py
=======================
Download the labelled source face dataset used to build the STAT6207
Assignment 3 gender-recognition training set.

Source
------
FairFace (Karkkainen & Joo, WACV 2021)
  paper : https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf
  repo  : https://github.com/joojs/fairface
  mirror: https://huggingface.co/datasets/HuggingFaceM4/FairFace  (CC-BY-4.0)

We use the *validation* split of the "1.25 padding" configuration because:
  * it carries original race / gender / age labels (no labels are inferred by us);
  * it is real-world "in the wild" photography (YFCC100M / Flickr), not studio
    headshots, so lighting / pose / background / quality vary naturally;
  * the 1.25 padding keeps scene context around each face, which lets us
    demonstrate genuine MTCNN face detection + cropping;
  * the split is small enough (~237 MB) to be reproducible on a laptop.

The downloaded parquet is stored unchanged in ../data_raw/.
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(DATA_RAW, "fairface_125_validation.parquet")

URL = (
    "https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/main/"
    "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet"
)
EXPECTED_BYTES = 236824583  # size observed at download time (informational)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(DATA_RAW, exist_ok=True)
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        print(f"[skip] already present: {OUT}")
    else:
        print(f"[get ] {URL}")
        tmp = OUT + ".part"
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                got += len(block)
                if total:
                    sys.stdout.write(f"\r       {got/1e6:7.1f} / {total/1e6:.1f} MB")
                    sys.stdout.flush()
        print()
        os.replace(tmp, OUT)
    print(f"[ok  ] {OUT}  ({os.path.getsize(OUT):,} bytes)")
    print(f"[sha ] {sha256(OUT)}")


if __name__ == "__main__":
    main()
