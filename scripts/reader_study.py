"""Reader study: clinicians vs the model on the same DENTEX test radiographs.

  python scripts/reader_study.py make     # build reader_study/ (images, instructions, key)
  python scripts/reader_study.py score    # after readers' YOLO labels are in reader_study/readers/<name>/

make: 50 test images, all 11 caries-free ones plus 39 random caries images (seed 0), renamed
R001..R050 in random order so readers cannot infer anything from file names. The key file
mapping them back is kept out of the folder sent to readers.

score: per reader and per model (at its validation-tuned threshold) on the same 50 images,
lesion-level sensitivity, precision and F1 (IoU >= 0.5) and image-level sensitivity and
specificity against the DENTEX ground truth; pairwise agreement (lesion F1 between two
readers' boxes, image-level Cohen's kappa) between readers and between each reader and the model.
"""
import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from metrics import counts_at  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "reader_study"
DATA = ROOT / "data/dentex_yolo"
MODELS = {"Faster R-CNN": "frcnn_s0", "YOLOv8s": "yolov8s"}

INSTRUCTIONS = """# Caries reader study: instructions

Thank you for taking part. This takes about 60 to 90 minutes.

## Task
For each of the 50 panoramic radiographs in `images/`, mark **every tooth with dental caries**
(any depth, including deep caries) with a rectangle. Draw the rectangle around the **whole
affected tooth** (crown and root), which is how the reference dataset was annotated.
If a radiograph has no caries, leave it without boxes.

Please work independently, do not discuss cases with other readers, and do not use any AI tool.
Use the screen and viewing conditions you would normally use.

## Tool: makesense.ai (free, runs in your browser, images are not uploaded to a server)
1. Open https://www.makesense.ai and click **Get Started**.
2. Drag in all 50 images from `images/` and choose **Object Detection**.
3. Create one label named exactly `caries`, then **Start project**.
4. Draw a rectangle around each carious tooth. Use the arrow keys to move between images.
5. When finished: **Actions > Export Annotations > "A .zip package containing files in YOLO format"**.
6. Send the zip back, renamed with your initials (for example `AB.zip`). Please also send:
   your role (dentist, specialist, student), years of experience, and the time taken.

Questions: contact the researcher.
"""


def make():
    imgs = sorted((DATA / "images/test").glob("*.png"))
    has = [bool((DATA / "labels/test" / f"{p.stem}.txt").read_text().strip()) for p in imgs]
    rng = np.random.default_rng(0)
    neg = [i for i, h in enumerate(has) if not h]
    pos = rng.choice([i for i, h in enumerate(has) if h], 50 - len(neg), replace=False).tolist()
    chosen = rng.permutation(neg + pos).tolist()
    (RS / "images").mkdir(parents=True, exist_ok=True)
    (RS / "readers").mkdir(exist_ok=True)
    key = []
    for n, i in enumerate(chosen, 1):
        rid = f"R{n:03d}"
        shutil.copy(imgs[i], RS / "images" / f"{rid}.png")
        key.append(dict(reader_id=rid, image=imgs[i].name, test_index=i, has_caries=has[i]))
    pd.DataFrame(key).to_csv(RS / "key_DO_NOT_SEND.csv", index=False)
    (RS / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    print(f"{len(chosen)} images ({len(neg)} caries-free) -> {RS / 'images'}. "
          f"Send images/ and INSTRUCTIONS.md only. Put returned labels in readers/<initials>/")


def yolo_boxes(path, w, h):
    rows = [list(map(float, l.split()[1:5])) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    b = np.array(rows).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)


def rates(preds, gts):
    les, img = counts_at(preds, gts, conf=0.5)
    p = les["TP"] / max(les["TP"] + les["FP"], 1)
    r = les["TP"] / max(les["TP"] + les["FN"], 1)
    return dict(lesion_sensitivity=r, lesion_precision=p, lesion_f1=2 * p * r / (p + r) if p + r else 0.0,
                image_sensitivity=img["TP"] / max(img["TP"] + img["FN"], 1),
                image_specificity=img["TN"] / max(img["TN"] + img["FP"], 1), **{f"les_{k}": v for k, v in les.items()})


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def score():
    key = pd.read_csv(RS / "key_DO_NOT_SEND.csv")
    sizes = [Image.open(DATA / "images/test" / im).size for im in key.image]
    gts = [yolo_boxes(DATA / "labels/test" / f"{Path(im).stem}.txt", w, h) for im, (w, h) in zip(key.image, sizes)]
    sets = {}
    for d in sorted(p for p in (RS / "readers").iterdir() if p.is_dir()):
        sets[d.name] = [(b, np.ones(len(b))) for b in
                        (yolo_boxes(next(d.rglob(f"{rid}.txt"), d / "none"), w, h) for rid, (w, h) in zip(key.reader_id, sizes))]
    for m, name in MODELS.items():
        z = np.load(ROOT / f"runs/eval/{name}_test.npz", allow_pickle=True)
        conf = json.loads((ROOT / f"runs/eval/{name}_test.json").read_text())["conf"]
        sets[m] = [(z["boxes"][i][z["scores"][i] >= conf], np.ones(int((z["scores"][i] >= conf).sum())))
                   for i in key.test_index]
    if len(sets) == len(MODELS):
        print("No reader labels yet in reader_study/readers/<initials>/")
    perf = pd.DataFrame([dict(reader=k, **rates(v, gts)) for k, v in sets.items()])
    agree = []
    for a, b in combinations(sets, 2):
        les, _ = counts_at(sets[a], [x[0] for x in sets[b]], conf=0.5)  # b's boxes as reference
        f1 = 2 * les["TP"] / max(2 * les["TP"] + les["FP"] + les["FN"], 1)  # symmetric in a and b
        ia, ib = [len(x[0]) > 0 for x in sets[a]], [len(x[0]) > 0 for x in sets[b]]
        agree.append(dict(pair=f"{a} vs {b}", lesion_f1_agreement=f1, image_kappa=kappa(ia, ib)))
    perf.to_csv(RS / "reader_performance.csv", index=False)
    pd.DataFrame(agree).to_csv(RS / "agreement.csv", index=False)
    print(perf.round(3).to_string(index=False), "\n")
    print(pd.DataFrame(agree).round(3).to_string(index=False))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
