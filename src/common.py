"""Shared helpers. Person B: import these so preprocessing matches training exactly."""
import json
from pathlib import Path

import numpy as np

IMG_SIZE = 128
PROC_DIR = Path("data/processed")
MODEL_PATH = Path("models/cnn.pt")


def load_split(name, proc_dir=PROC_DIR):
    """Return (X uint8 [N,128,128], y int64 [N]) for name in {train,val,test}."""
    proc_dir = Path(proc_dir)
    X = np.load(proc_dir / f"{name}_X.npy")
    y = np.load(proc_dir / f"{name}_y.npy")
    return X, y


def load_ids(name, proc_dir=PROC_DIR):
    return np.load(Path(proc_dir) / f"{name}_ids.npy")


def load_boxes(proc_dir=PROC_DIR):
    """Ground-truth boxes: {patientId: [[x, y, w, h], ...]} already scaled to IMG_SIZE."""
    with open(Path(proc_dir) / "boxes.json") as f:
        return json.load(f)


def load_stats(proc_dir=PROC_DIR):
    with open(Path(proc_dir) / "stats.json") as f:
        s = json.load(f)
    return s["mean"], s["std"]


def normalize(x_uint8, mean, std):
    """uint8 image(s) -> float32 normalized with TRAIN mean/std (computed on the 0-1 scale)."""
    return ((x_uint8.astype(np.float32) / 255.0) - mean) / std
