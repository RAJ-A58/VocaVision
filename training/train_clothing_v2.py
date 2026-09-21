"""
VocaVision — Clothing Classifier v2 (PyTorch, GPU)
====================================================
Architecture : MobileNetV2  (ImageNet pretrained)
Dataset      : Fashion MNIST  →  upsampled 28×28 → 224×224 RGB
Augmentation : Heavy (flip, rotation, brightness, contrast, zoom)
Training     : Two-phase  (frozen backbone → fine-tune all layers)
Mixed Prec.  : torch.cuda.amp  (faster on RTX 30xx)
Output       : models/clothing_classifier_v2.pt

Expected accuracy:
  Fashion MNIST test set : ~92–95 %
  Real-world photos      : ~70–80 %  (vs ~40 % with old CNN)

Run: python training/train_clothing_v2.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "..", "models", "clothing_classifier_v2.pt")
CURVES_OUT  = os.path.join(os.path.dirname(__file__), "..", "models", "clothing_v2_training_curves.png")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", ".fashion_mnist_cache")

IMG_SIZE    = 224
BATCH_SIZE  = 64       # RTX 3050 6GB handles 64 comfortably
PHASE1_EP   = 8        # frozen backbone
PHASE2_EP   = 15       # fine-tune all layers
LR1         = 1e-3
LR2         = 5e-5
NUM_CLASSES = 10

CLOTHING_CLASSES = [
    "T-shirt or top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data transforms ───────────────────────────────────────────────────────────
# Fashion MNIST is grayscale — replicate to 3 channels, upsample to 224×224
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),       # 1ch → 3ch RGB
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_dataloaders():
    print("[1/4] Loading Fashion MNIST...")
    train_ds = datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"    Train: {len(train_ds):,} images  |  Val: {len(val_ds):,} images")
    return train_dl, val_dl


def build_model():
    print("[2/4] Building MobileNetV2 model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Replace classifier head for 10 Fashion MNIST classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {total:,} total, {trainable:,} trainable")
    return model.to(DEVICE)


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_all(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def run_epoch(model, loader, criterion, optimizer, scaler, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with autocast():
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


def train():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VocaVision — Clothing Classifier v2  (PyTorch GPU Training)")
    print(f"Device  : {DEVICE}  {'✅ GPU!' if DEVICE.type == 'cuda' else '⚠️ CPU (slower)'}")
    print(f"Classes : {', '.join(CLOTHING_CLASSES)}")
    print(f"{'='*60}\n")

    train_dl, val_dl = build_dataloaders()
    model            = build_model()
    criterion        = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler           = GradScaler()

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0
    best_state   = None

    # ── Phase 1: Train head only (backbone frozen) ────────────────────────────
    print(f"\n[3/4] Phase 1 — Training classifier head ({PHASE1_EP} epochs)...")
    freeze_backbone(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EP)

    for ep in range(1, PHASE1_EP + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, scaler, training=True)
        va_loss, va_acc = run_epoch(model, val_dl,   criterion, optimizer, scaler, training=False)
        scheduler.step()

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  P1 Ep {ep:02d}/{PHASE1_EP}  "
              f"train={tr_acc:.1%}  val={va_acc:.1%}  "
              f"loss={va_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({time.time()-t0:.0f}s)")

    # ── Phase 2: Fine-tune all layers ─────────────────────────────────────────
    print(f"\n[4/4] Phase 2 — Fine-tuning all layers ({PHASE2_EP} epochs)...")
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EP, eta_min=1e-6)

    for ep in range(1, PHASE2_EP + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer, scaler, training=True)
        va_loss, va_acc = run_epoch(model, val_dl,   criterion, optimizer, scaler, training=False)
        scheduler.step()

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        marker = "  ← best" if va_acc == best_val_acc else ""
        print(f"  P2 Ep {ep:02d}/{PHASE2_EP}  "
              f"train={tr_acc:.1%}  val={va_acc:.1%}  "
              f"loss={va_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({time.time()-t0:.0f}s){marker}")

    # ── Save best model ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "classes":     CLOTHING_CLASSES,
        "img_size":    IMG_SIZE,
        "architecture": "mobilenet_v2",
    }, MODEL_OUT)

    print(f"\n✅ Best Validation Accuracy : {best_val_acc:.2%}")
    print(f"✅ Model saved → {MODEL_OUT}")

    # ── Training curves ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax1.plot(epochs, history["train_acc"], label="Train")
    ax1.plot(epochs, history["val_acc"],   label="Validation")
    ax1.axvline(PHASE1_EP + 0.5, color="gray", linestyle="--", alpha=0.5, label="Fine-tune start")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history["train_loss"], label="Train")
    ax2.plot(epochs, history["val_loss"],   label="Validation")
    ax2.axvline(PHASE1_EP + 0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(CURVES_OUT, dpi=120)
    print(f"Training curves saved → {CURVES_OUT}")


if __name__ == "__main__":
    train()
