"""
Train a Clothing Classifier using a custom CNN on Fashion MNIST.

Dataset : Fashion MNIST (10 categories, ~30 MB — auto-downloaded by Keras)
Architecture: 3-block CNN trained from scratch on 28x28 grayscale images
Output  : models/clothing_classifier.keras

Run:
    python training/train_clothing.py
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
# MUST match CLOTHING_CLASSES order in vision_engine.py
CLOTHING_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",  "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",    "Ankle boot"
]

EPOCHS     = 20
BATCH_SIZE = 64
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    """
    Loads Fashion MNIST via tf.keras.datasets and applies data augmentation
    to the training set to improve generalisation.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalise to [0, 1] and add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = x_train.astype(np.float32)[..., np.newaxis] / 255.0
    x_test  = x_test.astype(np.float32)[..., np.newaxis]  / 255.0

    # Augmentation pipeline for training data
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    ds_train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                              .shuffle(10000)
                              .batch(BATCH_SIZE)
                              .map(lambda x, y: (augment(x, training=True), y),
                                   num_parallel_calls=tf.data.AUTOTUNE)
                              .prefetch(tf.data.AUTOTUNE))

    ds_val = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                             .batch(BATCH_SIZE)
                             .prefetch(tf.data.AUTOTUNE))

    return ds_train, ds_val, x_test, y_test

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """
    A 3-block CNN designed for 28x28 grayscale images.
    Trained completely from scratch — no pretrained weights.

    Architecture summary:
      Block 1 : Conv(32) → BN → Conv(32) → MaxPool → Dropout
      Block 2 : Conv(64) → BN → Conv(64) → MaxPool → Dropout
      Block 3 : Conv(128) → BN → MaxPool → Dropout
      Head    : Flatten → Dense(256) → BN → Dropout → Softmax(10)
    """
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),

        # Classification head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="clothing_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ── Training ──────────────────────────────────────────────────────────────────

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "clothing_classifier.keras")

    print("=" * 60)
    print("VocaVision — Clothing Classifier Training")
    print(f"Dataset: Fashion MNIST | Classes: {len(CLOTHING_CLASSES)}")
    print("=" * 60)

    print("\n[1/3] Loading Fashion MNIST dataset...")
    ds_train, ds_val, x_test, y_test = build_dataset()

    print("\n[2/3] Building CNN model...")
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    print(f"\n[3/3] Training (up to {EPOCHS} epochs, early stopping enabled)...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Final Test Accuracy : {acc:.2%}")
    print(f"   Final Test Loss     : {loss:.4f}")

    model.save(save_path)
    print(f"✅ Clothing model saved → {save_path}")

    _plot_history(history)


def _plot_history(history):
    acc   = history.history["accuracy"]
    val   = history.history["val_accuracy"]
    loss  = history.history["loss"]
    vloss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, acc, label="Train Acc")
    ax1.plot(epochs, val, label="Val Acc")
    ax1.set_title("Clothing Classifier — Accuracy"); ax1.legend()

    ax2.plot(epochs, loss, label="Train Loss")
    ax2.plot(epochs, vloss, label="Val Loss")
    ax2.set_title("Clothing Classifier — Loss"); ax2.legend()

    plt.tight_layout()
    out = os.path.join(MODELS_DIR, "clothing_training_curves.png")
    plt.savefig(out)
    print(f"Training curves saved → {out}")


if __name__ == "__main__":
    train()
