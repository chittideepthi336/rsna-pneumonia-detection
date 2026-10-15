"""Paper-aligned localization variants (Person B).

The paper compares thresholds 0.2 / 0.45 / 0.7, clusters with DFS (= connected components),
keeps boxes "within 2 standard deviations of all predictions" (wording is vague; we read it as
box AREA within mean +/- 2 std over all predicted boxes), and gives IoU 0 to images with no box.

Prints, for each threshold and filter:
  box-IoU  : our per-GT-box best-match IoU (same as eval_iou)
  mask-IoU : pixel IoU between union of predicted boxes and union of GT boxes
  'oracle'   = every GT-positive image goes through CAM
  'pipeline' = images the classifier misses (P<=0.5) count as 0

Usage:  python -m src.paper_compare
"""
import numpy as np
import torch
from scipy import ndimage

from src.common import MODEL_PATH, load_boxes
from src.eval_iou import positives
from src.localize import image_iou
from src.model import load_model


def components(cam, t):
    labels, _ = ndimage.label(cam >= t, structure=np.ones((3, 3)))
    out = []
    for ys, xs in ndimage.find_objects(labels):
        out.append([int(xs.start), int(ys.start), int(xs.stop - xs.start), int(ys.stop - ys.start)])
    return out


def mask_iou(pred, gt, size=128):
    a, b = np.zeros((size, size), bool), np.zeros((size, size), bool)
    for x, y, w, h in pred:
        a[y:y + h, x:x + w] = True
    for box in gt:
        x, y, w, h = (int(round(v)) for v in box)
        b[max(y, 0):y + h, max(x, 0):x + w] = True
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_model(MODEL_PATH, device)
    gt = load_boxes()
    _, ids, probs, cams = positives("test", model, mean, std, device)
    tp = probs > 0.5
    print(f"test positives: {len(ids)}  (classifier hits: {int(tp.sum())})")
    print(f"{'thr':5s} {'filter':20s} | box-IoU oracle  pipeline | mask-IoU oracle  pipeline")
    for t in (0.2, 0.45, 0.7):
        allb = [components(c, t) for c in cams]
        areas = np.array([w * h for bs in allb for _, _, w, h in bs], float)
        mu, sd = (areas.mean(), areas.std()) if len(areas) else (0.0, 0.0)
        for name, ok in (("all components", lambda b: True),
                         ("2-sigma area filter", lambda b: abs(b[2] * b[3] - mu) <= 2 * sd)):
            bs_, ms_ = [], []
            for bs, i in zip(allb, ids):
                kept = [b for b in bs if ok(b)]
                g = gt.get(str(i), [])
                bs_.append(image_iou(kept, g))
                ms_.append(mask_iou(kept, g))
            bs_, ms_ = np.array(bs_), np.array(ms_)
            print(f"{t:<5} {name:20s} | {bs_.mean():.3f}          {np.where(tp, bs_, 0).mean():.3f}"
                  f"    | {ms_.mean():.3f}          {np.where(tp, ms_, 0).mean():.3f}")
    print("paper (CNN+CAM, test): 0.1508   paper 'all correctly classified': 0.379   paper R-CNN: 0.1266")


if __name__ == "__main__":
    main()