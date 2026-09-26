"""
VocaVision — YOLOv8 Training Script
=====================================
Fine-tunes YOLOv8s on the combined food + clothing dataset prepared by
prepare_yolo_dataset.py.

Hardware: NVIDIA RTX 3050 6GB (CUDA device 0)
Expected training time: ~25–40 minutes for 50 epochs on YOLOv8s

Usage:
    python training/train_yolo.py

Output:
    models/vocavision_yolo.pt   ← best weights (ready for inference)
    runs/detect/vocavision/     ← YOLO training logs, confusion matrix, curves
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML  = Path(__file__).parent.parent / "dataset" / "data.yaml"
MODEL_OUT     = Path(__file__).parent.parent / "models" / "vocavision_yolo.pt"
RUNS_DIR      = Path(__file__).parent.parent / "runs"

# Training hyperparameters — tuned for RTX 3050 6GB
CONFIG = dict(
    data    = str(DATASET_YAML),
    epochs  = 60,              # 60 epochs — good balance of speed + accuracy
    imgsz   = 640,             # Standard YOLO input size
    batch   = 16,              # Fits in 6 GB VRAM on YOLOv8s
    device  = 0,               # CUDA GPU 0 (RTX 3050)
    workers = 4,               # DataLoader workers
    project = str(RUNS_DIR / "detect"),
    name    = "vocavision",
    exist_ok= True,

    # ── Augmentation (YOLO's built-in mosaic + albumentations) ───────────────
    mosaic   = 1.0,            # Mosaic augmentation (4-image tiles)
    mixup    = 0.15,           # MixUp augmentation
    degrees  = 15.0,           # Random rotation
    fliplr   = 0.5,            # Horizontal flip
    scale    = 0.5,            # Random scale
    translate= 0.1,            # Random translation
    hsv_h    = 0.015,          # HSV hue
    hsv_s    = 0.7,            # HSV saturation
    hsv_v    = 0.4,            # HSV value

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer= "AdamW",
    lr0      = 0.001,          # Initial learning rate
    lrf      = 0.01,           # Final lr = lr0 * lrf
    momentum = 0.937,
    weight_decay = 0.0005,
    warmup_epochs= 3,
    cos_lr   = True,           # Cosine LR scheduler

    # ── Save + Eval ───────────────────────────────────────────────────────────
    save     = True,
    save_period = 10,          # Save checkpoint every 10 epochs
    val      = True,
    plots    = True,           # Save confusion matrix, PR curves
    verbose  = True,
)


def verify_dataset():
    if not DATASET_YAML.exists():
        print(f"\nERROR: Dataset not found at {DATASET_YAML}")
        print("Run this first:  python training/prepare_yolo_dataset.py")
        sys.exit(1)
    import yaml
    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)
    nc    = cfg.get("nc", 0)
    names = cfg.get("names", [])
    train_imgs = Path(cfg["path"]) / cfg["train"]
    val_imgs   = Path(cfg["path"]) / cfg["val"]
    n_train = len(list(train_imgs.glob("*.jpg")) + list(train_imgs.glob("*.png")))
    n_val   = len(list(val_imgs.glob("*.jpg"))   + list(val_imgs.glob("*.png")))
    print(f"[OK] Dataset: {nc} classes, {n_train} train images, {n_val} val images")
    return nc


def train():
    from ultralytics import YOLO

    nc = verify_dataset()
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  VocaVision — YOLOv8s Custom Training")
    print(f"  Classes  : {nc} (food + clothing)")
    print(f"  Epochs   : {CONFIG['epochs']}")
    print(f"  Img size : {CONFIG['imgsz']}x{CONFIG['imgsz']}")
    print(f"  Batch    : {CONFIG['batch']}")
    print(f"  Device   : CUDA:{CONFIG['device']} (RTX 3050)")
    print("="*60 + "\n")

    # Load YOLOv8s pretrained on COCO — best starting point for real-world objects
    model = YOLO("yolov8s.pt")

    # Train!
    results = model.train(**CONFIG)

    # Copy best weights to models/
    best_pt = Path(RUNS_DIR) / "detect" / "vocavision" / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, MODEL_OUT)
        print(f"\n[OK] Best weights copied → {MODEL_OUT}")
    else:
        print(f"\n[WARN] best.pt not found at {best_pt}; check runs/ directory")

    # Print summary
    metrics = results.results_dict
    print("\n" + "="*60)
    print("  Training Complete!")
    print(f"  mAP50    : {metrics.get('metrics/mAP50(B)', 0):.3f}")
    print(f"  mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
    print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
    print(f"  Recall   : {metrics.get('metrics/recall(B)', 0):.3f}")
    print("="*60)
    print(f"\n  Model saved → {MODEL_OUT}")
    print("  Now run:  python app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    train()
