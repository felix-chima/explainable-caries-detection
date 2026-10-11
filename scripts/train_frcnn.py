"""Train the primary model: Faster R-CNN, ResNet-50 FPN v2 backbone (thesis Sec 3.6).

Follows Chapter 3: COCO-pretrained weights (transfer learning), Adam, per-image
min-max normalisation (Eq. 3.1), flip + brightness/contrast augmentation on the
training split only, early stopping on validation mAP50 with the best state kept,
fixed seed. Uses the same DENTEX split as the YOLO baseline (data/dentex_yolo).

  python scripts/train_frcnn.py --name frcnn_s0 --seed 0
  python scripts/train_frcnn.py --name frcnn_s0 --resume      # after an interruption
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, str(Path(__file__).parent))
from metrics import average_precision  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dentex_yolo"


def load_boxes(path, w, h):
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    b = torch.tensor([[float(v) for v in r[1:5]] for r in rows], dtype=torch.float32).reshape(-1, 4)
    cx, cy, bw, bh = b.unbind(1)
    return torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)


class Dentex(torch.utils.data.Dataset):
    def __init__(self, split, train=False, limit=None):
        self.split, self.train = split, train
        self.imgs = sorted((DATA / "images" / split).glob("*.png"))[:limit]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert("L")
        w, h = img.size
        x = TF.to_tensor(img)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # Eq. 3.1
        boxes = load_boxes(DATA / "labels" / self.split / f"{p.stem}.txt", w, h)
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.8:
                x = TF.adjust_brightness(x, random.uniform(0.8, 1.2))
                x = TF.adjust_contrast(x, random.uniform(0.8, 1.2)).clamp(0, 1)
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return x.expand(3, -1, -1), target


def collate(batch):
    return tuple(zip(*batch))


def build_model(min_size, max_size):
    m = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=min_size, max_size=max_size,
        box_score_thresh=0.001, box_detections_per_img=100,
    )
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for imgs, targets in loader:
        for out, t in zip(model([i.to(device) for i in imgs]), targets):
            preds.append((out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()))
            gts.append(t["boxes"].numpy())
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="frcnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8, help="early stopping, epochs without val mAP50 gain")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--max-size", type=int, default=1333)
    ap.add_argument("--limit", type=int, default=None, help="images per split, for smoke tests")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = ROOT / "runs/frcnn" / a.name
    (out / "weights").mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    g = torch.Generator().manual_seed(a.seed)
    train_dl = torch.utils.data.DataLoader(Dentex("train", True, a.limit), a.batch, shuffle=True,
                                           collate_fn=collate, num_workers=2, generator=g)
    val_dl = torch.utils.data.DataLoader(Dentex("val", False, a.limit), 2, collate_fn=collate, num_workers=2)

    model = build_model(a.min_size, a.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=3)
    start, best, bad = 1, -1.0, 0
    if a.resume:
        ck = torch.load(out / "weights/last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
    else:
        (out / "args.json").write_text(json.dumps(vars(a), indent=2))
        with open(out / "results.csv", "w") as f:
            f.write("epoch,train_loss,val_mAP50,lr,seconds\n")

    for epoch in range(start, a.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for step, (imgs, targets) in enumerate(train_dl, 1):
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            total, n = total + loss.item(), n + 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {total / n:.3f}", flush=True)
        val_map = average_precision(*predict(model, val_dl, device))
        sched.step(val_map)
        lr = opt.param_groups[0]["lr"]
        secs = time.time() - t0
        with open(out / "results.csv", "a") as f:
            csv.writer(f).writerow([epoch, f"{total / n:.4f}", f"{val_map:.4f}", lr, f"{secs:.0f}"])
        if val_map > best:
            best, bad = val_map, 0
            torch.save(model.state_dict(), out / "weights/best.pt")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=epoch, best=best, bad=bad, py_rng=random.getstate(),
                        torch_rng=torch.get_rng_state()), out / "weights/last.pt")
        print(f"epoch {epoch}/{a.epochs} loss {total / n:.3f} val mAP50 {val_map:.4f} "
              f"best {best:.4f} lr {lr:.0e} {secs:.0f}s", flush=True)
        if bad >= a.patience:
            print(f"early stopping: no val mAP50 gain for {a.patience} epochs", flush=True)
            break
    print(f"done. best val mAP50 {best:.4f} -> {out / 'weights/best.pt'}", flush=True)


if __name__ == "__main__":
    main()
