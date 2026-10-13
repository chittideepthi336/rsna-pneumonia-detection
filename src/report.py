"""Merge results/baselines.json and results/cnn.json into results/results_table.md (paste into write-up)."""
import json
from pathlib import Path

base = json.loads(Path("results/baselines.json").read_text())
cnn = json.loads(Path("results/cnn.json").read_text())

rows = [(k, v["train_acc"], v["test_acc"]) for k, v in base.items()] + [("CNN (GAP, 10 conv)", cnn["train_acc"], cnn["test_acc"])]
lines = ["| Classifier | Train acc | Test acc |", "|---|---|---|"]
lines += [f"| {n} | {a:.2%} | {b:.2%} |" for n, a, b in rows]
c = cnn["confusion"]
lines += [
    "",
    "| Confusion matrix (test) | Predicted pneumonia | Predicted healthy |",
    "|---|---|---|",
    f"| Actual pneumonia | {c['TP']} | {c['FN']} |",
    f"| Actual healthy | {c['FP']} | {c['TN']} |",
    "",
    f"Precision {cnn['precision']:.2%} | Sensitivity (recall) {cnn['recall_sensitivity']:.2%} | "
    f"Specificity {cnn['specificity']:.2%} | F1 {cnn['f1']:.2%}",
]
Path("results/results_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
