"""Turn the raw RSNA Pneumonia Detection Challenge data into small, balanced .npy splits.

Steps (mirrors the paper's Section 3):
  1. Keep only 'Lung Opacity' (pneumonia, label 1) and 'Normal' (healthy, label 0);
     drop 'No Lung Opacity / Not Normal' (diseased but no pneumonia).
  2. Balance the classes by subsampling the larger one.
  3. Resize 1024x1024 -> 128x128, store as uint8.
  4. Split 70/20/10 train/val/test (stratified, one row per patient, so no patient leakage).
  5. Compute mean/std on the TRAIN split only.
  6. Save ground-truth boxes (scaled to 128x128) for Person B's IoU evaluation.

Usage:
  python -m src.preprocess --raw_dir /kaggle/input/rsna-pneumonia-detection-challenge
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.common import IMG_SIZE

ORIG_SIZE = 1024  # RSNA images are 1024x1024


def find_file(raw_dir, name):
    hits = list(Path(raw_dir).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {raw_dir}")
    return hits[0]


def load_image(dcm_path, size):
    arr = pydicom.dcmread(dcm_path).pixel_array
    img = Image.fromarray(arr).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="folder holding the RSNA csvs and stage_2_train_images/")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only N images per class")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(find_file(raw, "stage_2_train_labels.csv"))
    info = pd.read_csv(find_file(raw, "stage_2_detailed_class_info.csv"))
    img_dir = find_file(raw, "stage_2_train_images")

    # one class per patient (the csv repeats a patient once per box)
    df = info.drop_duplicates("patientId")
    print("Original class counts (unique patients):\n", df["class"].value_counts().to_string())

    df = df[df["class"].isin(["Lung Opacity", "Normal"])].copy()
    df["label"] = (df["class"] == "Lung Opacity").astype(int)

    pos, neg = df[df.label == 1], df[df.label == 0]
    n = min(len(pos), len(neg))
    if args.limit:
        n = min(n, args.limit)
    df = pd.concat([pos.sample(n=n, random_state=args.seed), neg.sample(n=n, random_state=args.seed)])
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"Balanced to {n} pneumonia + {n} healthy = {len(df)} images")

    # 70 / 20 / 10 split
    trval, test = train_test_split(df, test_size=0.10, stratify=df.label, random_state=args.seed)
    train, val = train_test_split(trval, test_size=2 / 9, stratify=trval.label, random_state=args.seed)
    splits = {"train": train, "val": val, "test": test}

    stats = None
    for name, part in splits.items():
        X = np.stack([load_image(img_dir / f"{pid}.dcm", args.size) for pid in tqdm(part.patientId, desc=name)])
        y = part.label.to_numpy(dtype=np.int64)
        np.save(out / f"{name}_X.npy", X)
        np.save(out / f"{name}_y.npy", y)
        np.save(out / f"{name}_ids.npy", part.patientId.to_numpy(dtype="U36"))
        print(f"{name}: X{X.shape}  pneumonia={int(y.sum())}  healthy={int((1 - y).sum())}")
        if name == "train":
            x01 = X.astype(np.float32) / 255.0
            stats = {"mean": float(x01.mean()), "std": float(x01.std())}

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f)
    print("Train mean/std:", stats)

    # ground-truth boxes (x, y, w, h) scaled to the resized image, only for kept pneumonia patients
    s = args.size / ORIG_SIZE
    kept = set(df[df.label == 1].patientId)
    boxes = {}
    for r in labels[(labels.Target == 1) & labels.patientId.isin(kept)].itertuples():
        boxes.setdefault(r.patientId, []).append([r.x * s, r.y * s, r.width * s, r.height * s])
    with open(out / "boxes.json", "w") as f:
        json.dump(boxes, f)
    print(f"Saved boxes for {len(boxes)} pneumonia patients")


if __name__ == "__main__":
    main()
