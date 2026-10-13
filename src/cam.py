"""Class Activation Maps for the GAP-CNN (Person B).

CAM_c(x, y) = sum_k  w_{c,k} * f_k(x, y)
  f_k : last conv feature maps  [B, 512, 16, 16]
  w_c : FC weights for class c  [512]   (class 1 = pneumonia)

Usage:  python -m src.cam      (shape smoke test on an untrained model)
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.common import IMG_SIZE, normalize


@torch.no_grad()
def compute_cam(model, x, cls=1, size=IMG_SIZE):
    """x: float tensor [B,1,128,128] (already normalized).

    Returns (probs [B], cams [B,size,size] in 0-1, per-image min-max normalized).
    cls=None -> use each image's predicted class.
    """
    model.eval()
    logits, feats = model(x, return_features=True)          # feats [B,512,16,16]
    probs = logits.softmax(1)[:, 1]                          # P(pneumonia)
    if cls is None:
        w = model.fc.weight[logits.argmax(1)]                # [B,512]
        cam = torch.einsum("bk,bkhw->bhw", w, feats)
    else:
        cam = torch.einsum("k,bkhw->bhw", model.fc.weight[cls], feats)
    cam = F.interpolate(cam.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    lo, hi = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    cam = (cam - lo) / (hi - lo + 1e-8)
    return probs.cpu().numpy(), cam.cpu().numpy()


def cam_for_uint8(model, imgs_uint8, mean, std, cls=1, device="cpu", bs=64):
    """imgs_uint8: [N,128,128] uint8. Returns (probs [N], cams [N,128,128])."""
    P, C = [], []
    for i in range(0, len(imgs_uint8), bs):
        xb = normalize(imgs_uint8[i:i + bs], mean, std)
        xb = torch.from_numpy(xb).unsqueeze(1).to(device)
        p, c = compute_cam(model, xb, cls=cls)
        P.append(p); C.append(c)
    return np.concatenate(P), np.concatenate(C)


if __name__ == "__main__":
    from src.model import PneumoniaCNN
    m = PneumoniaCNN()
    p, c = compute_cam(m, torch.randn(2, 1, IMG_SIZE, IMG_SIZE))
    print("probs", p.shape, "cams", c.shape, "range", c.min(), c.max())