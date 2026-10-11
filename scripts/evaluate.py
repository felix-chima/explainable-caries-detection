"""Predict a DENTEX split with Faster R-CNN or the YOLO baseline and score both with metrics.py.

Raw predictions (conf >= 0.001) are cached to runs/eval/<name>_<split>.npz so later steps
(bootstrap CIs, other thresholds) rescore without rerunning the model.

  python scripts/evaluate.py frcnn runs/frcnn/frcnn_s0/weights/best.pt --name frcnn_s0
  python scripts/evaluate.py yolo runs/detect/full/weights/best.pt --name yolov8s
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from metrics import best_f1_conf, summary  # noqa: E402
import train_frcnn  # noqa: E402
from train_frcnn import ROOT, Dentex, build_model, collate, load_boxes, predict  # noqa: E402


def predict_frcnn(weights, split, device):
    model = build_model(800, 1333)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.to(device)
    dl = torch.utils.data.DataLoader(Dentex(split), 1, collate_fn=collate)
    return predict(model, dl, device)


def predict_yolo(weights, split, device):
    from ultralytics import YOLO

    model = YOLO(weights)
    data = train_frcnn.DATA
    imgs = sorted((data / "images" / split).glob("*.png"))
    preds, gts = [], []
    for p, r in zip(imgs, model.predict([str(p) for p in imgs], imgsz=640, conf=0.001, iou=0.7,
                                        max_det=300, device=device, stream=True, verbose=False)):
        preds.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        w, h = Image.open(p).size
        gts.append(load_boxes(data / "labels" / split / f"{p.stem}.txt", w, h).numpy())
    return preds, gts


def cached_predict(a, split, out):
    cache = out / f"{a.name}_{split}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(zip(z["boxes"], z["scores"])), list(z["gts"])
    fn = predict_frcnn if a.model == "frcnn" else predict_yolo
    preds, gts = fn(a.weights, split, torch.device(a.device))
    np.savez(cache, boxes=np.array([b for b, _ in preds], dtype=object),
             scores=np.array([s for _, s in preds], dtype=object), gts=np.array(gts, dtype=object))
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["frcnn", "yolo"])
    ap.add_argument("weights")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", default="auto",
                    help="operating threshold for P/R/F1/specificity; 'auto' = max-F1 threshold on val")
    ap.add_argument("--data", help="other YOLO-format dataset root, e.g. data/external/children_yolo")
    ap.add_argument("--conf-from", help="eval name whose DENTEX val predictions set the threshold "
                                        "(external sets: never tune on the external data)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    out = ROOT / "runs/eval"
    out.mkdir(parents=True, exist_ok=True)
    if a.data:
        train_frcnn.DATA = (ROOT / a.data).resolve()
    preds, gts = cached_predict(a, a.split, out)
    if a.conf != "auto":
        conf = float(a.conf)
    elif a.conf_from:
        z = np.load(out / f"{a.conf_from}_val.npz", allow_pickle=True)
        conf = best_f1_conf(list(zip(z["boxes"], z["scores"])), list(z["gts"]))
    else:
        conf = best_f1_conf(*cached_predict(a, "val", out))

    res = dict(name=a.name, model=a.model, split=a.split, conf=conf, images=len(gts),
               **summary(preds, gts, conf))
    (out / f"{a.name}_{a.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
