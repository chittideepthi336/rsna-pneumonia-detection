# Pneumonia Diagnosis, Detection and Localization (Weakly Supervised)

UE24CS352A Machine Learning mini-project, Problem 34.
Based on *Weakly Supervised Pneumonia Localization* (Huang, Monam, Cortes, CS229 2018).

Pipeline: chest X-ray -> CNN classifier (pneumonia vs healthy) -> if pneumonia, Class Activation Map (CAM)
-> heatmap -> clustering -> bounding box. Only image-level labels are used for training;
bounding boxes are used only to **evaluate** localization (IoU).

## Team
| Member | Part |
|---|---|
| Person A (Deepthi) | Data preprocessing, baselines, CNN classifier, classification results |
| Person B | CAM, clustering, bounding boxes, IoU evaluation, demo app |

## Setup
```bash
git clone <this repo> && cd pneumonia-localization
pip install -r requirements.txt
```
Data: accept the rules and download the competition data from
<https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data>
(we need `stage_2_train_images/`, `stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`).
On a Kaggle notebook, add the competition as an input and use `/kaggle/input/rsna-pneumonia-detection-challenge`.

## Part A: classification (run in this order)
```bash
python -m src.preprocess --raw_dir <path-to-rsna-data>   # -> data/processed/*.npy (128x128, balanced, 70/20/10)
python -m src.baselines                                  # LR / SVM / RF  -> results/baselines.json
python -m src.train_cnn                                  # CNN (GPU recommended) -> models/cnn.pt, results/cnn.json, plots
python -m src.report                                     # -> results/results_table.md
```
Data decisions (as in the paper): keep `Lung Opacity` (pneumonia) and `Normal` (healthy), drop
`No Lung Opacity / Not Normal`, balance the classes, resize 1024->128, normalize with train mean/std, split 70/20/10 by patient.

Model: 10 conv layers (3x3, zero padding) -> Global Average Pooling -> one FC layer (2 classes).
Adam, lr 1e-4, 20 epochs. We add BatchNorm after each conv for stable training (`--no_bn` for the paper-exact version).

### Interface for Part B
`src/model.py`: `model(x, return_features=True)` returns `(logits, feature_maps[B,512,16,16])`;
`model.fc.weight[1]` are the pneumonia weights for CAM. `src.common` has `load_split`, `load_boxes`
(ground-truth boxes scaled to 128x128), `load_stats` and `normalize`. Trained weights: `models/cnn.pt`
(load with `src.model.load_model`).

## Part B: localization and demo - 

Only image-level labels are used for training. Ground-truth boxes are used only to evaluate localization.

**Method** (`src/cam.py`, `src/localize.py`)
1. Classify the X-ray with the CNN. Only images predicted pneumonia are localized.
2. Compute the Class Activation Map: `CAM = sum_k w_k * f_k(x)`, using the pneumonia-class FC weights and the last conv feature maps (16x16), bilinearly upsampled to 128x128 and min-max normalized.
3. Threshold the CAM (default 0.6 of the max, chosen on the validation split).
4. Cluster the surviving pixels with connected components (8-neighbourhood, equivalent to a DFS flood fill).
5. Take one bounding box per cluster, drop boxes under 100 px area, keep the 2 strongest.
6. Score with IoU against the RSNA ground-truth boxes (scaled 1024 -> 128). Images with no predicted box score 0.

**Run it** (needs `models/cnn.pt` and `data/processed/`, see Part A)
```
python -m src.eval_iou          # threshold sweep on val, IoU on test -> results/iou.json, results/localization_examples.png
python -m src.diagnose          # baselines and variants (mean-GT-box, union box, pointing game)
python -m src.paper_compare     # the paper's thresholds (0.2 / 0.45 / 0.7) and a pixel-mask IoU
python app.py                   # Gradio demo at http://127.0.0.1:7860
```
Demo: upload a chest X-ray (PNG/JPG) to see the predicted label, the CAM heatmap overlay and the predicted box. The slider changes the CAM threshold.

## Results

**Classification:** see `results_table.md` (Part A).

**Localization** (test split, 601 pneumonia images, classifier detects 544)

| Setting | Mean IoU | IoU > 0.5 | IoU > 0.3 |
|---|---|---|---|
| CAM, classifier hits only (n=544) | 0.139 | 0.7% | 5.1% |
| CAM, all positives, misses count as 0 (n=601) | 0.126 | 0.7% | 4.7% |
| Baseline: always predict the mean GT box | 0.062 | 0.0% | 0.7% |

- Pointing game (CAM peak inside a GT box): 0.34.
- The paper reports 0.1508 mean IoU for CNN+CAM (and 0.1266 for its supervised R-CNN), so our localization is in the same range.
- Threshold sensitivity (paper's thresholds, pipeline mean IoU): 0.2 -> 0.083, 0.45 -> 0.126, 0.7 -> 0.123.

**Differences from the paper:** BatchNorm in the CNN, global instead of per-pixel normalization, min-area filtering instead of the paper's vague "2 standard deviations" box rule, no U-Net lung-segmentation input, and no R-CNN benchmark (scoped out).

**Limitations:** the CAM is only 16x16 while average GT boxes are about 28x41 px, the model overfits (train accuracy 1.0), missed pneumonia images get no box, and IoU against small boxes is a harsh metric.

