"""Convert an external caries dataset to YOLO format, single class `caries`, all images as `test`.

children: Zhang et al. (2023), Sci Data, paediatric panoramic radiographs (figshare
  10.6084/m9.figshare.c.6317013). LabelMe rectangles; only `龋病` (caries) is kept. `深窝沟`
  (deep pits and fissures, pre-carious), pulpitis, periapical periodontitis, developmental
  anomaly and `其他` (other) are dropped. Train and Test are pooled because nothing is retrained.

  python scripts/prepare_external.py children
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = {"children": ROOT / "data/external/children_panoramic/raw/extracted/Pediatric dental disease detection dataset"}
CARIES = {"children": {"龋病"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SRC))
    a = ap.parse_args()
    out = ROOT / f"data/external/{a.dataset}_yolo"
    for d in ("images/test", "labels/test"):
        (out / d).mkdir(parents=True, exist_ok=True)
    seen, kept, n_img, n_neg = Counter(), 0, 0, 0
    for split in ("Train", "Test"):
        for js in sorted((SRC[a.dataset] / split / "label").glob("*.json")):
            d = json.loads(js.read_text())
            img = SRC[a.dataset] / split / "images" / f"{js.stem}.png"
            w, h = d["imageWidth"], d["imageHeight"]
            rows = []
            for s in d["shapes"]:
                seen[s["label"]] += 1
                if s["label"] not in CARIES[a.dataset] or s["shape_type"] != "rectangle":
                    continue
                (x1, y1), (x2, y2) = s["points"]
                x1, x2 = sorted((max(0, x1), min(w, x2)))
                y1, y2 = sorted((max(0, y1), min(h, y2)))
                rows.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            stem = f"{split.lower()}_{js.stem}"
            shutil.copy(img, out / "images/test" / f"{stem}.png")
            (out / "labels/test" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            kept, n_img, n_neg = kept + len(rows), n_img + 1, n_neg + (not rows)
    (ROOT / f"{a.dataset}.yaml").write_text(f"path: {out}\ntrain: images/test\nval: images/test\ntest: images/test\n"
                                           "names:\n  0: caries\n")
    print(f"{n_img} images, {kept} caries boxes, {n_img - n_neg} images with caries, {n_neg} without")
    print("all labels seen:", dict(seen))


if __name__ == "__main__":
    main()
