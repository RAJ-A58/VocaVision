"""
Train a Food Classifier using Transfer Learning on MobileNetV2.

Dataset : Food-101 (10 selected classes via tensorflow_datasets)
          NOTE: First-time download is ~4.6 GB. It is cached after that.
Architecture: MobileNetV2 (ImageNet pretrained) + custom softmax head
Output  : models/food_classifier.keras

Training runs in two phases:
  Phase 1 — Freeze backbone, train only the new classification head (fast)
  Phase 2 — Unfreeze top 30 backbone layers for fine-tuning (slower, better accuracy)

Run:
    python training/train_food.py
"""
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────
# These 10 classes MUST match FOOD_CLASSES in vision_engine.py (sorted alphabetically)
SELECTED_CLASSES = sorted([
    "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",   "waffles"
])

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 5   # Phase 1: train head only
EPOCHS_FINE = 5   # Phase 2: fine-tune top backbone layers
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset():
    """
    Loads Food-101 via tfds, filters to SELECTED_CLASSES only, and remaps
    labels from the original 0-100 range to a compact 0-9 range.
    """
    builder = tfds.builder("food101")
    builder.download_and_prepare()
    info = builder.info

    all_classes     = info.features["label"].names           # 101 class names
    selected_idx    = sorted([all_classes.index(c) for c in SELECTED_CLASSES])
    selected_tensor = tf.constant(selected_idx, dtype=tf.int64)

    def filter_fn(ex):
        label = tf.cast(ex["label"], tf.int64)
        return tf.reduce_any(tf.equal(label, selected_tensor))

    def preprocess(ex):
        image = tf.image.resize(ex["image"], [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        # Remap original label index → compact 0-9 label
        orig  = tf.cast(ex["label"], tf.int64)
        new_label = tf.cast(tf.where(tf.equal(selected_tensor, orig))[0][0], tf.int32)
        return image, new_label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    ds_train = (builder.as_dataset(split="train")
                       .filter(filter_fn)
                       .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                       .map(augment,    num_parallel_calls=tf.data.AUTOTUNE)
                       .shuffle(2000)
                       .batch(BATCH_SIZE)
                       .prefetch(tf.data.AUTOTUNE))

    ds_val   = (builder.as_dataset(split="validation")
                       .filter(filter_fn)
                       .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                       .batch(BATCH_SIZE)
                       .prefetch(tf.data.AUTOTUNE))

    return ds_train, ds_val

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    """
    Builds a MobileNetV2-based transfer learning classifier.
    The backbone is frozen during Phase 1; top 30 layers are unfrozen in Phase 2.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # frozen for Phase 1

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base

# ── Training ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "food_classifier.keras")

    print("=" * 60)
    print("VocaVision — Food Classifier Training")
    print(f"Classes ({len(SELECTED_CLASSES)}): {', '.join(SELECTED_CLASSES)}")
    print("=" * 60)

    print("\n[1/4] Building dataset (Food-101 — may download ~4.6 GB on first run)...")
    ds_train, ds_val = build_dataset()

    print("\n[2/4] Building MobileNetV2 model...")
    model, base = build_model(len(SELECTED_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print(f"\n[3/4] Phase 1 — Training head only ({EPOCHS_HEAD} epochs)...")
    history1 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_HEAD)

    print(f"\n[4/4] Phase 2 — Fine-tuning top 30 backbone layers ({EPOCHS_FINE} epochs)...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history2 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_FINE)

    model.save(save_path)
    print(f"\n✅ Food model saved → {save_path}")

    _plot_history(history1, history2)


def _plot_history(h1, h2):
    acc   = h1.history["accuracy"]      + h2.history["accuracy"]
    val   = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss  = h1.history["loss"]          + h2.history["loss"]
    vloss = h1.history["val_loss"]      + h2.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    split  = len(h1.history["accuracy"])   # where fine-tuning starts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, acc, label="Train Acc")
    ax1.plot(epochs, val, label="Val Acc")
    ax1.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("Food Classifier — Accuracy"); ax1.legend()

    ax2.plot(epochs, loss, label="Train Loss")
    ax2.plot(epochs, vloss, label="Val Loss")
    ax2.axvline(split, color="gray", linestyle="--")
    ax2.set_title("Food Classifier — Loss"); ax2.legend()

    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "food_training_curves.png")
    plt.savefig(out)
    print(f"Training curves saved → {out}")


if __name__ == "__main__":
    train()
