"""
VocaVision — YOLO Dataset Preparation v2
=========================================
Fixed approach:
  - Downloads ALL food + clothing classes in ONE fiftyone call (no overwrite clash)
  - Uses a single shared dataset directory so images aren't clobbered
  - Converts directly using display label names that fiftyone stores

Usage:
    python training/prepare_yolo_dataset.py

Output:
    dataset/
    ├── data.yaml
    ├── train/images/ + train/labels/
    └── val/images/   + val/labels/
"""

import os
import sys
import shutil
import yaml
import random
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.yolo_classes import ALL_CLASSES, OI_ALL_MAP, OI_FOOD_CLASSES, OI_CLOTHING_CLASSES

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path(__file__).parent.parent / "dataset"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"
VAL_IMAGES   = DATASET_DIR / "val"   / "images"
VAL_LABELS   = DATASET_DIR / "val"   / "labels"
DATA_YAML    = DATASET_DIR / "data.yaml"

MAX_SAMPLES = 6000   # total images across all classes (food + clothing)
VAL_SPLIT   = 0.15


def make_dirs():
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directories ready under {DATASET_DIR}")


def download_all():
    """Download food + clothing in ONE combined fiftyone call."""
    try:
        import fiftyone.zoo as foz
    except ImportError:
        print("ERROR: pip install fiftyone")
        sys.exit(1)

    all_classes = OI_FOOD_CLASSES + OI_CLOTHING_CLASSES
    print(f"\n[1/3] Downloading from Open Images V7 ...")
    print(f"      Classes ({len(all_classes)}): {all_classes}")
    print(f"      Max samples: {MAX_SAMPLES}")

    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=all_classes,
        max_samples=MAX_SAMPLES,
        dataset_name="vocavision_all_tmp",
        overwrite=True,
    )
    print(f"      Loaded {len(ds)} samples")
    return ds


def convert_to_yolo(fo_dataset):
    """Convert fiftyone dataset -> YOLO txt labels, train/val split."""
    all_items  = list(fo_dataset)
    random.shuffle(all_items)
    n_val      = int(len(all_items) * VAL_SPLIT)
    val_idxs   = set(range(n_val))

    written_train = written_val = skipped = 0
    label_counts  = {}

    for idx, sample in enumerate(all_items):
        img_path = sample.filepath
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        H, W = img.shape[:2]

        lines = []
        for det in (sample.ground_truth.detections if sample.ground_truth else []):
            lbl = det.label
            if lbl not in OI_ALL_MAP:
                continue
            our_lbl  = OI_ALL_MAP[lbl]
            class_id = ALL_CLASSES.index(our_lbl)
            x1, y1, bw, bh = det.bounding_box   # fiftyone: relative [0,1]
            xc = x1 + bw / 2
            yc = y1 + bh / 2
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            label_counts[our_lbl] = label_counts.get(our_lbl, 0) + 1

        if not lines:
            skipped += 1
            continue

        is_val   = idx in val_idxs
        img_dst  = (VAL_IMAGES  if is_val else TRAIN_IMAGES) / os.path.basename(img_path)
        lbl_dst  = (VAL_LABELS  if is_val else TRAIN_LABELS) / (Path(img_path).stem + ".txt")

        shutil.copy2(img_path, img_dst)
        lbl_dst.write_text("\n".join(lines))

        if is_val:
            written_val   += 1
        else:
            written_train += 1

    print(f"\n      Train: {written_train}  |  Val: {written_val}  |  Skipped: {skipped}")
    print("\n      Per-class box counts:")
    for cls in ALL_CLASSES:
        cnt = label_counts.get(cls, 0)
        bar = "#" * min(30, cnt // 10)
        print(f"        {cls:15s} {cnt:5d}  {bar}")

    return written_train, written_val


def write_data_yaml():
    cfg = {
        "path" : str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val"  : "val/images",
        "nc"   : len(ALL_CLASSES),
        "names": ALL_CLASSES,
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n[3/3] data.yaml written -> {DATA_YAML}")


def main():
    random.seed(42)
    make_dirs()

    fo_ds = download_all()

    print("\n[2/3] Converting to YOLO label format ...")
    write_data_yaml()

    n_train, n_val = convert_to_yolo(fo_ds)

    print("\n" + "="*60)
    print("  Dataset ready!  Now run:")
    print("  python training/train_yolo.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
