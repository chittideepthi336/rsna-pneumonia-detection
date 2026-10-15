"""Localization diagnostics (Person B): compare CAM boxes against simple baselines and variants.

Usage:  python -m src.diagnose
"""
import numpy as np
import torch

from src.common import MODEL_PATH, load_boxes
from src.eval_iou import positives
from src.localize import extract_boxes, image_iou
from src.model import load_model


def union_box(cam, t):
    """One box around ALL pixels >= t."""
    ys, xs = np.where(cam >= t)
    if len(xs) == 0:
        return []
    return [[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]]


def score(cams, ids, gt, fn):
    s = np.array([image_iou(fn(c), gt.get(str(i), [])) for c, i in zip(cams, ids)])
    return s.mean(), (s > 0.5).mean(), (s > 0.3).mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_model(MODEL_PATH, device)
    gt = load_boxes()

    _, vids, _, _ = positives("val", model, mean, std, device)
    _, ids, probs, cams = positives("test", model, mean, std, device)

    # mean GT box from val (a "no-model" baseline)
    allb = np.array([b for i in vids for b in gt.get(str(i), [])], dtype=float)
    mb = [int(round(v)) for v in allb.mean(0)]
    print("mean GT box (x,y,w,h):", mb, " GT boxes per image:", len(allb) / len(vids))
    print(f"{'method':32s} meanIoU  >0.5   >0.3")
    rows = [("mean-GT-box baseline (no CAM)", lambda c: [mb])]
    for t in (0.6, 0.5):
        rows.append((f"CAM components t={t} (current)", lambda c, t=t: extract_boxes(c, t)))
    for t in (0.5, 0.4, 0.3, 0.2):
        rows.append((f"CAM union box t={t}", lambda c, t=t: union_box(c, t)))
    for name, fn in rows:
        m, a, b = score(cams, ids, gt, fn)
        print(f"{name:32s} {m:.3f}   {a:.3f}  {b:.3f}")

    # pointing game: is the CAM peak inside any GT box?
    hits = []
    for c, i in zip(cams, ids):
        y, x = np.unravel_index(c.argmax(), c.shape)
        hits.append(any(bx <= x < bx + bw and by <= y < by + bh for bx, by, bw, bh in gt.get(str(i), [])))
    print(f"pointing game (CAM peak inside GT box): {np.mean(hits):.3f}")


if __name__ == "__main__":
    main()