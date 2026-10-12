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

## Part B: localization and demo
_Person B fills this in: CAM, clustering, IoU, how to run the demo._

## Results
_Paste `results/results_table.md` and the IoU numbers here once available._
