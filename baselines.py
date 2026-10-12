"""Baselines from the paper: Logistic Regression, SVM, Random Forest on flattened pixels.

Images are block-averaged (default 4x4 -> 32x32 = 1024 features) so the SVM finishes in minutes.
Use --pool 1 for full 128x128 (16,384 features) if you have time; document whichever you use.

Usage:  python -m src.baselines
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common import load_split, load_stats, normalize


def flatten(X, pool, mean, std):
    X = normalize(X, mean, std)
    if pool > 1:
        n, h, w = X.shape
        X = X.reshape(n, h // pool, pool, w // pool, pool).mean(axis=(2, 4))
    return X.reshape(len(X), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()

    mean, std = load_stats(args.data_dir)
    Xtr, ytr = load_split("train", args.data_dir)
    Xte, yte = load_split("test", args.data_dir)
    Xtr, Xte = flatten(Xtr, args.pool, mean, std), flatten(Xte, args.pool, mean, std)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    print("Feature matrix:", Xtr.shape)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=args.seed),
        "SVM (RBF)": SVC(kernel="rbf", random_state=args.seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=args.seed),
    }
    results = {}
    for name, clf in models.items():
        t = time.time()
        clf.fit(Xtr, ytr)
        results[name] = {
            "train_acc": float(accuracy_score(ytr, clf.predict(Xtr))),
            "test_acc": float(accuracy_score(yte, clf.predict(Xte))),
            "pool": args.pool,
        }
        print(f"{name:20s} train={results[name]['train_acc']:.4f} test={results[name]['test_acc']:.4f} ({time.time() - t:.0f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
