"""
VocaVision — YOLO Dataset Preparation
======================================
Downloads food + clothing images from Open Images V7 via fiftyone,
converts them to YOLO format, and creates the dataset/data.yaml file.

Usage:
    python training/prepare_yolo_dataset.py

Output:
    dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/   (jpg files)
    │   └── labels/   (txt files — YOLO format)
    └── val/
        ├── images/
        └── labels/

YOLO label format per line:
    class_id  x_center  y_center  width  height   (all 0-1 normalised)
"""

import os
import sys
import shutil
import yaml
import random
import cv2
import numpy as np
from pathlib import Path

# ── Add project root so we can import yolo_classes ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.yolo_classes import (
    ALL_CLASSES, OI_ALL_OI_CLASSES, OI_ALL_MAP,
    OI_FOOD_CLASSES, OI_CLOTHING_CLASSES
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path(__file__).parent.parent / "dataset"
TRAIN_IMAGES  = DATASET_DIR / "train" / "images"
TRAIN_LABELS  = DATASET_DIR / "train" / "labels"
VAL_IMAGES    = DATASET_DIR / "val"   / "images"
VAL_LABELS    = DATASET_DIR / "val"   / "labels"
DATA_YAML     = DATASET_DIR / "data.yaml"

# Max images to download per Open Images class
# (keep low for faster download; increase to 800–1200 for better accuracy)
MAX_PER_CLASS = 300
VAL_SPLIT     = 0.15   # 15% of images → validation set


def make_dirs():
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Dataset directories ready under {DATASET_DIR}")


def download_open_images():
    """
    Uses fiftyone to download bounding-box annotated images from Open Images V7.
    Requires: pip install fiftyone
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("ERROR: fiftyone not installed. Run: pip install fiftyone")
        sys.exit(1)

    print(f"\n[1/3] Downloading from Open Images V7...")
    print(f"      Classes : {OI_ALL_OI_CLASSES}")
    print(f"      Max/cls : {MAX_PER_CLASS}")

    # Download food
    print("\n  Downloading FOOD classes...")
    food_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=OI_FOOD_CLASSES,
        max_samples=MAX_PER_CLASS * len(OI_FOOD_CLASSES),
        dataset_name="vocavision_food_tmp",
        overwrite=True,
    )

    # Download clothing
    print("\n  Downloading CLOTHING classes...")
    cloth_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=OI_CLOTHING_CLASSES,
        max_samples=MAX_PER_CLASS * len(OI_CLOTHING_CLASSES),
        dataset_name="vocavision_clothing_tmp",
        overwrite=True,
    )

    return food_ds, cloth_ds


def convert_to_yolo(fo_dataset, split_images, split_labels,
                    other_images=None, other_labels=None):
    """
    Converts a fiftyone dataset to YOLO txt label format.
    Returns number of images written to train, val splits.
    """
    import fiftyone as fo

    all_items = list(fo_dataset)
    random.shuffle(all_items)
    n_val   = int(len(all_items) * VAL_SPLIT)
    val_set = set(range(n_val))

    written = 0
    for idx, sample in enumerate(all_items):
        img_path = sample.filepath
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        lines = []
        for det in (sample.ground_truth.detections if sample.ground_truth else []):
            oi_label = det.label
            if oi_label not in OI_ALL_MAP:
                continue
            our_label  = OI_ALL_MAP[oi_label]
            class_id   = ALL_CLASSES.index(our_label)
            x1, y1, bw, bh = det.bounding_box   # fiftyone uses relative [0,1]
            x_c = x1 + bw / 2
            y_c = y1 + bh / 2
            lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        is_val  = (idx in val_set) and (other_images is not None)
        img_dst = (other_images  if is_val else split_images) / os.path.basename(img_path)
        lbl_dst = (other_labels  if is_val else split_labels) / (Path(img_path).stem + ".txt")

        shutil.copy2(img_path, img_dst)
        lbl_dst.write_text("\n".join(lines))
        written += 1

    return written


def write_data_yaml():
    cfg = {
        "path"  : str(DATASET_DIR.resolve()),
        "train" : "train/images",
        "val"   : "val/images",
        "nc"    : len(ALL_CLASSES),
        "names" : ALL_CLASSES,
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n[3/3] data.yaml written → {DATA_YAML}")


def main():
    random.seed(42)
    make_dirs()

    food_ds, cloth_ds = download_open_images()

    print("\n[2/3] Converting to YOLO label format...")
    n_food  = convert_to_yolo(food_ds,  TRAIN_IMAGES, TRAIN_LABELS,
                               VAL_IMAGES, VAL_LABELS)
    n_cloth = convert_to_yolo(cloth_ds, TRAIN_IMAGES, TRAIN_LABELS,
                               VAL_IMAGES, VAL_LABELS)

    print(f"      Food images   : {n_food}")
    print(f"      Cloth images  : {n_cloth}")

    write_data_yaml()

    print("\n============================================================")
    print("  Dataset ready! Now run:")
    print("  python training/train_yolo.py")
    print("============================================================\n")


if __name__ == "__main__":
    main()
