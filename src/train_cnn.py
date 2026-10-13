"""Train the GAP-CNN classifier (Adam, lr 1e-4, 20 epochs as in the paper).

Saves: models/cnn.pt (best validation accuracy), results/cnn.json, results/curves.png,
       results/confusion_matrix.png

Usage:  python -m src.train_cnn
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.common import MODEL_PATH, load_split, load_stats
from src.model import PneumoniaCNN, save_model


def make_loader(X, y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def prep(xb, mean, std, device):
    return ((xb.to(device).float() / 255.0) - mean) / std


@torch.no_grad()
def predict(model, loader, mean, std, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(prep(xb, mean, std, device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_bn", action="store_true", help="paper-exact (no BatchNorm); trains slower")
    ap.add_argument("--model_out", default=str(MODEL_PATH))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    mean, std = load_stats(args.data_dir)
    Xtr, ytr = load_split("train", args.data_dir)
    Xva, yva = load_split("val", args.data_dir)
    Xte, yte = load_split("test", args.data_dir)
    tr = make_loader(Xtr, ytr, args.bs, True)
    tr_eval = make_loader(Xtr, ytr, args.bs, False)
    va = make_loader(Xva, yva, args.bs, False)
    te = make_loader(Xte, yte, args.bs, False)

    model = PneumoniaCNN(use_bn=not args.no_bn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    hist = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        t = time.time()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in tr:
            xb, yb = prep(xb, mean, std, device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        val_acc = float((predict(model, va, mean, std, device) == yva).mean())
        hist["train_loss"].append(total_loss / n)
        hist["train_acc"].append(correct / n)
        hist["val_acc"].append(val_acc)
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            save_model(model, args.model_out, mean, std)
            flag = "  * saved"
        print(f"epoch {ep:2d}/{args.epochs}  loss {total_loss / n:.4f}  train {correct / n:.4f}  val {val_acc:.4f}  ({time.time() - t:.0f}s){flag}")

    # ---- final evaluation with the best-validation checkpoint ----
    ckpt = torch.load(args.model_out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    train_acc = float((predict(model, tr_eval, mean, std, device) == ytr).mean())
    pred = predict(model, te, mean, std, device)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    res = {
        "train_acc": train_acc,
        "val_acc": best_val,
        "test_acc": float((pred == yte).mean()),
        "precision": float(tp / max(tp + fp, 1)),
        "recall_sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.bs, "batchnorm": not args.no_bn,
    }
    res["f1"] = 2 * res["precision"] * res["recall_sensitivity"] / max(res["precision"] + res["recall_sensitivity"], 1e-9)
    with open("results/cnn.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # ---- plots ----
    ep_axis = range(1, args.epochs + 1)
    plt.figure(figsize=(5, 3.4))
    plt.plot(ep_axis, hist["train_acc"], label="train")
    plt.plot(ep_axis, hist["val_acc"], label="validation")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Training vs validation accuracy")
    plt.legend(); plt.tight_layout(); plt.savefig("results/curves.png", dpi=150); plt.close()

    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(3.6, 3.2))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Healthy", "Pneumonia"]); plt.yticks([0, 1], ["Healthy", "Pneumonia"])
    plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Test confusion matrix")
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
