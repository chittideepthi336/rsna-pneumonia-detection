"""Gradio demo (Person B): upload a chest X-ray -> label, heatmap overlay, predicted box.

Run from repo root:  python app.py
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from matplotlib import cm
from PIL import Image, ImageDraw

from src.cam import compute_cam
from src.common import IMG_SIZE, MODEL_PATH, normalize
from src.localize import extract_boxes
from src.model import load_model

SHOW = 384  # display size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, MEAN, STD = load_model(MODEL_PATH, DEVICE)


def best_thresh(default=0.6):
    p = Path("results/iou.json")
    if p.exists():
        return json.load(open(p)).get("best_thresh", default)
    return default


THRESH = best_thresh()


def predict(img, thresh):
    if img is None:
        return "Upload an image.", None, None
    g = Image.fromarray(img).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x_u8 = np.array(g, dtype=np.uint8)
    x = torch.from_numpy(normalize(x_u8[None], MEAN, STD)).unsqueeze(1).to(DEVICE)
    prob, cam = compute_cam(MODEL, x, cls=1)
    prob, cam = float(prob[0]), cam[0]
    positive = prob > 0.5
    label = f"{'Pneumonia' if positive else 'Healthy'}  (P(pneumonia) = {prob:.2f})"

    base = g.convert("RGB").resize((SHOW, SHOW), Image.BILINEAR)
    if not positive:
        return label, base, base

    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(heat).resize((SHOW, SHOW), Image.BILINEAR)
    overlay = Image.blend(base, heat, 0.4)

    boxed = base.copy()
    d = ImageDraw.Draw(boxed)
    s = SHOW / IMG_SIZE
    for (bx, by, bw, bh) in extract_boxes(cam, thresh):
        d.rectangle([bx * s, by * s, (bx + bw) * s, (by + bh) * s], outline=(255, 0, 0), width=3)
    return label, overlay, boxed


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="numpy", label="Chest X-ray (PNG/JPG)"),
            gr.Slider(0.3, 0.9, value=THRESH, step=0.05, label="CAM threshold")],
    outputs=[gr.Textbox(label="Prediction"), gr.Image(label="CAM heatmap"), gr.Image(label="Predicted box")],
    title="Pneumonia detection and localization (weakly supervised CAM)",
    description="CNN with global average pooling. The box comes from thresholding the class activation map. "
                "Research demo, not for clinical use.",
)

if __name__ == "__main__":
    demo.launch()