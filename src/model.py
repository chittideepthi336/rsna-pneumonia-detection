"""10-layer CNN with Global Average Pooling + one FC layer (the CAM-compatible design from the paper).

Person B contract:
  * input  : float32 tensor [B, 1, 128, 128], normalized with src.common.normalize
  * model(x, return_features=True) -> (logits [B, 2], feature_maps [B, 512, 16, 16])
  * model.fc.weight[1] (shape [512]) are the pneumonia-class weights used for CAM:
        cam = sum_k  w_k * f_k(x)
  * class index 1 = pneumonia, 0 = healthy
"""
import torch
import torch.nn as nn

# 10 conv layers (all 3x3, zero padding to keep size) with max-pool after blocks 2, 4 and 7.
CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]


class PneumoniaCNN(nn.Module):
    def __init__(self, n_classes=2, use_bn=True):
        super().__init__()
        layers, c_in = [], 1
        for v in CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.Conv2d(c_in, v, kernel_size=3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                c_in = v
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, n_classes)
        self.use_bn = use_bn

    def forward(self, x, return_features=False):
        f = self.features(x)
        logits = self.fc(self.gap(f).flatten(1))
        return (logits, f) if return_features else logits


def save_model(model, path, mean, std):
    torch.save({"state_dict": model.state_dict(), "use_bn": model.use_bn, "mean": mean, "std": std}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PneumoniaCNN(use_bn=ckpt["use_bn"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]
