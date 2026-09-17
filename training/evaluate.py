"""
Evaluate both trained classifiers and generate:
  - Classification reports (per-class precision, recall, F1)
  - Confusion matrix plots saved to models/

Run AFTER training both models:
    python training/evaluate.py
"""
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Make sure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FOOD_CLASSES = sorted([
    "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",   "waffles"
])

CLOTHING_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",  "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",    "Ankle boot"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _plot_cm(cm, class_names, title, save_path):
    """Plots a confusion matrix and saves it to disk."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = range(len(class_names))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel="True label", xlabel="Predicted label")
    plt.xticks(rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Confusion matrix saved → {save_path}")

# ── Evaluators ────────────────────────────────────────────────────────────────

def evaluate_clothing():
    print("\n" + "=" * 55)
    print("Clothing Classifier Evaluation  (Fashion MNIST test set)")
    print("=" * 55)

    model_path = os.path.join(MODELS_DIR, "clothing_classifier.keras")
    if not os.path.exists(model_path):
        print("  ❌ Model not found. Run: python training/train_clothing.py")
        return

    model = tf.keras.models.load_model(model_path)
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype(np.float32)[..., np.newaxis] / 255.0

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Accuracy : {acc:.2%}")
    print(f"  Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLOTHING_CLASSES))

    cm = confusion_matrix(y_test, y_pred)
    _plot_cm(cm, CLOTHING_CLASSES, "Clothing Classifier — Confusion Matrix",
             os.path.join(MODELS_DIR, "clothing_confusion_matrix.png"))


def evaluate_food():
    print("\n" + "=" * 55)
    print("Food Classifier Evaluation  (Food-101 validation split)")
    print("=" * 55)

    model_path = os.path.join(MODELS_DIR, "food_classifier.keras")
    if not os.path.exists(model_path):
        print("  ❌ Model not found. Run: python training/train_food.py")
        return

    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("  ❌ tensorflow-datasets not installed. Run: pip install tensorflow-datasets")
        return

    model = tf.keras.models.load_model(model_path)
    builder = tfds.builder("food101")
    builder.download_and_prepare()

    all_classes     = builder.info.features["label"].names
    selected_idx    = sorted([all_classes.index(c) for c in FOOD_CLASSES])
    selected_tensor = tf.constant(selected_idx, dtype=tf.int64)

    def filter_fn(ex):
        return tf.reduce_any(tf.equal(tf.cast(ex["label"], tf.int64), selected_tensor))

    def preprocess(ex):
        image = tf.image.resize(ex["image"], [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        orig  = tf.cast(ex["label"], tf.int64)
        new_label = tf.cast(tf.where(tf.equal(selected_tensor, orig))[0][0], tf.int32)
        return image, new_label

    ds_val = (builder.as_dataset(split="validation")
                     .filter(filter_fn)
                     .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(32)
                     .prefetch(tf.data.AUTOTUNE))

    all_true, all_pred = [], []
    for imgs, labels in ds_val:
        preds = model.predict(imgs, verbose=0)
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(np.argmax(preds, axis=1).tolist())

    acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\n  Validation Accuracy: {acc:.2%}")
    print("\n  Classification Report:")
    print(classification_report(all_true, all_pred, target_names=FOOD_CLASSES))

    cm = confusion_matrix(all_true, all_pred)
    _plot_cm(cm, FOOD_CLASSES, "Food Classifier — Confusion Matrix",
             os.path.join(MODELS_DIR, "food_confusion_matrix.png"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_clothing()
    evaluate_food()
    print("\n✅ Evaluation complete. Check models/ for confusion matrix images.")
