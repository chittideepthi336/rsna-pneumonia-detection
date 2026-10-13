"""Localization evaluation (Person B).

Sweeps the CAM threshold on the VAL split, then reports IoU on the TEST split at the best value.
Reports two views:
  * TP only        : positives the classifier got right (localization quality)
  * all positives  : every pneumonia image, missed ones count as IoU 0 (end-to-end)

Saves results/iou.json and results/localization_examples.png (best / worst cases).

Usage:  python -m src.eval_iou
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.cam import cam_for_uint8
from src.common import MODEL_PATH, load_boxes, load_ids, load_split
from src.localize import evaluate, extract_boxes, image_iou
from src.model import load_model

THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def positives(split, model, mean, std, device):
    X, y = load_split(split)
    ids = load_ids(split)
    m = y == 1
    X, ids = X[m], ids[m]
    probs, cams = cam_for_uint8(model, X, mean, std, cls=1, device=device)
    return X, ids, probs, cams


def summarize(scores, label):
    s = np.asarray(scores)
    return {"view": label, "n": int(len(s)), "mean_iou": float(s.mean()),
            "pct_iou_gt_0.5": float((s > 0.5).mean()), "pct_iou_gt_0.3": float((s > 0.3).mean())}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_model(MODEL_PATH, device)
    boxes_gt = load_boxes()
    Path("results").mkdir(exist_ok=True)

    # ---- pick threshold on val (all positives) ----
    _, vids, _, vcams = positives("val", model, mean, std, device)
    sweep = []
    for t in THRESHOLDS:
        r = evaluate(vcams, vids, boxes_gt, thresh=t)
        sweep.append({k: r[k] for k in ("thresh", "mean_iou", "pct_iou_gt_0.5", "pct_iou_gt_0.3")})
        print(f"val  thresh {t}: mean IoU {r['mean_iou']:.3f}  >0.5: {r['pct_iou_gt_0.5']:.3f}")
    best = max(sweep, key=lambda r: r["mean_iou"])["thresh"]
    print("best threshold:", best)

    # ---- test ----
    X, ids, probs, cams = positives("test", model, mean, std, device)
    r = evaluate(cams, ids, boxes_gt, thresh=best)
    scores = np.array(r["per_image"])
    tp = probs > 0.5
    scores_missed_zero = np.where(tp, scores, 0.0)
    out = {
        "best_thresh": best,
        "val_sweep": sweep,
        "test_tp_only": summarize(scores[tp], "TP only"),
        "test_all_positives": summarize(scores_missed_zero, "all positives (FN = 0)"),
    }
    with open("results/iou.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["test_tp_only"], indent=2))
    print(json.dumps(out["test_all_positives"], indent=2))

    # ---- examples: 4 best + 4 worst among TPs ----
    tp_idx = np.where(tp)[0]
    order = tp_idx[np.argsort(scores[tp_idx])]
    picks = list(order[-4:][::-1]) + list(order[:4])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(X[i], cmap="gray")
        ax.imshow(cams[i], cmap="jet", alpha=0.35)
        for (x, y, w, h) in boxes_gt.get(str(ids[i]), []):
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="lime", lw=2))
        for (x, y, w, h) in extract_boxes(cams[i], best):
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="red", lw=2))
        ax.set_title(f"IoU {scores[i]:.2f}", fontsize=10); ax.axis("off")
    fig.suptitle("Top row: best, bottom row: worst  (green = ground truth, red = predicted)")
    plt.tight_layout(); plt.savefig("results/localization_examples.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()