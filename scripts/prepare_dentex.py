"""Convert DENTEX disease annotations to a single-class YOLO caries dataset.

Reads the COCO-style JSON files shipped with DENTEX, keeps only boxes whose
disease label (category_id_3) is Caries (and, by default, Deep Caries), writes
one YOLO label file per image, and splits the images 70/15/15 into
train/val/test.

DENTEX is anonymised and has no patient IDs, so the split is per image. Images
with no caries boxes are kept as negatives (empty label files) so the model
also learns what a caries-free radiograph looks like.

Usage (from ~/caries, with the `caries` env active):
    python scripts/prepare_dentex.py                  # Caries + Deep Caries
    python scripts/prepare_dentex.py --caries-only    # Caries only
"""

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "dentex"
EXTRACTED = RAW / "extracted"
OUT = ROOT / "data" / "dentex_yolo"

CARIES, DEEP_CARIES = 1, 3  # ids in DENTEX categories_3


def extract_zips():
    for zip_path in sorted(RAW.glob("*.zip")):
        marker = EXTRACTED / f".{zip_path.stem}.done"
        if marker.exists():
            continue
        if zip_path.with_name(zip_path.name + ".aria2").exists():
            print(f"Skipping {zip_path.name}: download still in progress")
            continue
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(EXTRACTED)
        marker.touch()


def find_disease_sources():
    """Yield (annotation_json, image_dir) pairs that carry disease labels."""
    jsons = list(EXTRACTED.rglob("*.json")) + [RAW / "validation_triple.json"]
    for js in jsons:
        if not js.exists():
            continue
        with open(js) as f:
            data = json.load(f)
        if "categories_3" not in data:
            continue  # quadrant / enumeration-only subsets have no disease labels
        names = {img["file_name"] for img in data["images"]}
        # Image folder: look beside the JSON first, then anywhere. Subsets reuse
        # file names (train_0.png etc.), so prefer the disease subset's folder.
        sample = next(iter(names))
        hits = list(js.parent.rglob(sample)) or list(EXTRACTED.rglob(sample))
        hits.sort(key=lambda p: "disease" not in str(p))
        if not hits:
            print(f"  ! No images found for {js.name}, skipping")
            continue
        yield js, data, hits[0].parent


def to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    return ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caries-only", action="store_true",
                    help="exclude Deep Caries boxes (default: merge them into caries)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    keep = {CARIES} if args.caries_only else {CARIES, DEEP_CARIES}

    extract_zips()

    samples = []  # (image_path, [yolo lines])
    for js, data, img_dir in find_disease_sources():
        boxes = {}
        for ann in data["annotations"]:
            if ann["category_id_3"] in keep:
                boxes.setdefault(ann["image_id"], []).append(ann["bbox"])
        n_pos = 0
        for img in data["images"]:
            path = img_dir / img["file_name"]
            if not path.exists():
                continue
            lines = ["0 %.6f %.6f %.6f %.6f" % to_yolo(b, img["width"], img["height"])
                     for b in boxes.get(img["id"], [])]
            n_pos += bool(lines)
            samples.append((path, lines))
        print(f"{js.name}: {len(data['images'])} images, {n_pos} with caries")

    if not samples:
        raise SystemExit("No labelled images found - has the download finished?")

    random.Random(args.seed).shuffle(samples)
    n = len(samples)
    n_train, n_val = round(n * 0.70), round(n * 0.15)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    if OUT.exists():
        shutil.rmtree(OUT)
    for split, items in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for path, lines in items:
            # Prefix with source folder so names from different subsets never clash.
            stem = f"{path.parent.parent.name}_{path.stem}"
            shutil.copy2(path, OUT / "images" / split / f"{stem}{path.suffix}")
            (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        n_boxes = sum(len(l) for _, l in items)
        n_pos = sum(bool(l) for _, l in items)
        print(f"{split:5s}: {len(items)} images ({n_pos} with caries), {n_boxes} boxes")

    (ROOT / "caries.yaml").write_text(f"# caries.yaml\npath: {OUT}\ntrain: images/train\nval: images/val\n"
                                     "test: images/test\nnames:\n  0: caries\n")
    manifest = ROOT / "splits" / "dentex_split.csv"
    rows = sorted(f"{p.stem},{split}" for split in splits for p in (OUT / "images" / split).iterdir())
    if manifest.exists() and not args.caries_only:
        same = manifest.read_text().split() == ["image,split"] + rows
        print("split matches splits/dentex_split.csv" if same else
              "WARNING: split differs from splits/dentex_split.csv; results are not comparable")
    print(f"\nDataset written to {OUT}, config caries.yaml")


if __name__ == "__main__":
    main()
