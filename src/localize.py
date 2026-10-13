"""CAM heatmap -> bounding boxes, plus IoU scoring (Person B).

Boxes are [x, y, w, h] in 128x128 pixel coordinates (same format as ground truth in boxes.json).

Pipeline: threshold the CAM -> connected components (8-neighbour, same idea as DFS flood fill)
          -> bounding box per component -> drop tiny ones -> keep the strongest few.
"""
import numpy as np
from scipy import ndimage


def extract_boxes(cam, thresh=0.6, min_area=100, max_boxes=2):
    """cam: [H,W] in 0-1. thresh is a fraction of the CAM max (cam is already max-normalized).

    min_area: minimum box area in px (128x128 image = 16384 px).
    max_boxes: keep only the boxes with the highest total CAM mass.
    Returns list of [x, y, w, h].
    """
    mask = cam >= thresh
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))   # 8-connectivity
    found = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = sl
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if w * h < min_area:
            continue
        mass = cam[sl][labels[sl] == i].sum()
        found.append((mass, [int(xs.start), int(ys.start), int(w), int(h)]))
    found.sort(key=lambda t: -t[0])
    return [b for _, b in found[:max_boxes]]


def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def image_iou(pred_boxes, gt_boxes):
    """Per-image score: for each GT box take its best-matching predicted box, then average.
    No predicted boxes -> 0."""
    if not gt_boxes:
        return 0.0
    if not pred_boxes:
        return 0.0
    return float(np.mean([max(iou(p, g) for p in pred_boxes) for g in gt_boxes]))


def evaluate(cams, ids, boxes_gt, thresh=0.6, min_area=100, max_boxes=2):
    """cams [N,128,128], ids [N] patient ids (positives only), boxes_gt {id: [[x,y,w,h],...]}.

    Returns dict with mean IoU and % of images with IoU > 0.5 / > 0.3, plus per-image IoUs.
    """
    scores = []
    for cam, pid in zip(cams, ids):
        gt = boxes_gt.get(str(pid), [])
        pred = extract_boxes(cam, thresh, min_area, max_boxes)
        scores.append(image_iou(pred, gt))
    s = np.array(scores)
    return {
        "thresh": thresh, "n": int(len(s)),
        "mean_iou": float(s.mean()),
        "pct_iou_gt_0.5": float((s > 0.5).mean()),
        "pct_iou_gt_0.3": float((s > 0.3).mean()),
        "per_image": s.tolist(),
    }


if __name__ == "__main__":
    cam = np.zeros((128, 128)); cam[30:70, 40:90] = 1.0
    b = extract_boxes(cam)
    print("boxes", b, "iou vs GT", iou(b[0], [40, 30, 50, 40]))