# Explainable caries detection on panoramic radiographs

Faster R-CNN (ResNet-50 FPN v2) and a YOLOv8s baseline for caries detection on the DENTEX
panoramic radiographs, with external testing and a quantitative study of how faithful
Grad-CAM, HiResCAM, EigenCAM and RISE heatmaps are to the detector.

Master of Information Technology project (MIT 8313), Miva Open University.

## Setup
```bash
conda create -n caries python=3.11 && conda activate caries
pip install -r requirements.txt
```

## Data
- **DENTEX** (Hamamci et al., 2023): download from Hugging Face `ibrahimhamamci/DENTEX` into `data/dentex/`.
- **External, children's panoramic set** (Zhang et al., 2023, *Scientific Data*): figshare
  `10.6084/m9.figshare.c.6317013`, extract into `data/external/children_panoramic/raw/extracted/`.

## Reproduce every table and figure
| Step | Command | Output |
|---|---|---|
| 1. Build the DENTEX split (70/15/15 per image, seed 42; Caries + Deep Caries as one class) | `python scripts/prepare_dentex.py` | `data/dentex_yolo/`, `caries.yaml` |
| 2. YOLOv8s baseline, seeds 0 to 2 | `yolo detect train model=yolov8s.pt data=caries.yaml epochs=100 imgsz=640 batch=16 seed=S` | `runs/detect/` |
| 3. Faster R-CNN, seeds 0 to 2 | `python scripts/train_frcnn.py --name frcnn_sS --seed S --batch 1` | `runs/frcnn/` |
|    (steps 2 and 3 for seeds 1 and 2 as one resumable queue) | `zsh scripts/run_queue.sh [yolo:1 frcnn:2 ...]` | |
|    (or Faster R-CNN seeds on a free Colab T4) | [open `notebooks/colab_frcnn.ipynb` in Colab](https://colab.research.google.com/github/felix-chima/explainable-caries-detection/blob/main/notebooks/colab_frcnn.ipynb) | Drive `caries_runs/` |
| 4. Test predictions and scores (threshold = max F1 on validation) | `python scripts/evaluate.py {frcnn,yolo} <best.pt> --name NAME` | `runs/eval/` |
| 5. External set | `python scripts/prepare_external.py children` then `evaluate.py ... --data data/external/children_yolo --conf-from NAME` | `runs/eval/` |
| 6. Statistics and figures (bootstrap CIs, paired test, FROC, calibration) | `python scripts/analysis.py` | `runs/analysis/` |
| 7. Explainability faithfulness study | `python scripts/explain.py` | `runs/explain/` |
| 8. Reader study | `python scripts/reader_study.py make`, then `score` | `reader_study/` |

All detectors are scored by the same code (`scripts/metrics.py`): greedy IoU >= 0.5 matching,
COCO 101-point AP, lesion- and image-level sensitivity, precision, F1 and specificity.
