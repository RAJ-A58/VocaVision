"""
Vision Engine — Local Neural Network Inference (No external API required)

Loads two locally trained models:
  1. food_classifier.keras   — MobileNetV2 fine-tuned on Food-101 (10 classes)
  2. clothing_classifier.keras — CNN trained from scratch on Fashion MNIST (10 classes)

analyze_image() runs both models, picks the winner by confidence score,
and returns a natural-language description string for the TTS engine.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from utils.color_detector import describe_colors

# ── Class Labels ─────────────────────────────────────────────────────────────
# FOOD_CLASSES must be sorted alphabetically — matches label remapping in train_food.py
FOOD_CLASSES = sorted([
    "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",   "waffles"
])

# CLOTHING_CLASSES order must match Fashion MNIST label indices 0-9
CLOTHING_CLASSES = [
    "T-shirt or top", "Trouser", "Pullover", "Dress",  "Coat",
    "Sandal",         "Shirt",   "Sneaker",  "Bag",    "Ankle boot"
]

# ── Model Paths ───────────────────────────────────────────────────────────────
_MODELS_DIR      = os.path.join(os.path.dirname(__file__), "models")
_FOOD_MODEL_PATH = os.path.join(_MODELS_DIR, "food_classifier.keras")
_CLTH_MODEL_PATH = os.path.join(_MODELS_DIR, "clothing_classifier.keras")

# ── Model Loading ─────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Loads both trained classifiers from disk.
    Raises FileNotFoundError with a helpful message if either model is missing.

    Returns:
        (food_model, clothing_model) — both tf.keras.Model instances
    """
    if not os.path.exists(_FOOD_MODEL_PATH):
        raise FileNotFoundError(
            f"Food model not found at '{_FOOD_MODEL_PATH}'.\n"
            "  Train it first:  python training/train_food.py"
        )
    if not os.path.exists(_CLTH_MODEL_PATH):
        raise FileNotFoundError(
            f"Clothing model not found at '{_CLTH_MODEL_PATH}'.\n"
            "  Train it first:  python training/train_clothing.py"
        )

    print("[VocaVision] Loading food classifier...")
    food_model     = tf.keras.models.load_model(_FOOD_MODEL_PATH)

    print("[VocaVision] Loading clothing classifier...")
    clothing_model = tf.keras.models.load_model(_CLTH_MODEL_PATH)

    print("[VocaVision] Both models loaded successfully.\n")
    return food_model, clothing_model

# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_food(frame: np.ndarray) -> np.ndarray:
    """
    Prepares a BGR OpenCV frame for the MobileNetV2 food model.
    Output: float32 array of shape (1, 224, 224, 3), values in [0, 1].
    """
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def _preprocess_clothing(frame: np.ndarray) -> np.ndarray:
    """
    Prepares a BGR OpenCV frame for the Fashion MNIST clothing CNN.
    Converts to grayscale (matching training distribution) and resizes to 28x28.
    Output: float32 array of shape (1, 28, 28, 1), values in [0, 1].
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img  = cv2.resize(gray, (28, 28))
    img  = img.astype(np.float32) / 255.0
    return img.reshape(1, 28, 28, 1)

# ── Inference ─────────────────────────────────────────────────────────────────

def analyze_image(food_model: tf.keras.Model,
                  clothing_model: tf.keras.Model,
                  frame: np.ndarray) -> tuple:
    """
    Runs the frame through both classifiers.

    For clothing: also runs GrabCut segmentation and returns an annotated
    image with a bounding box, color-tinted mask, and text label.

    Args:
        food_model     : Loaded Keras food classifier
        clothing_model : Loaded Keras clothing classifier
        frame          : OpenCV BGR image (numpy array)

    Returns:
        (description: str, annotated_frame: np.ndarray)
        annotated_frame has clothing region highlighted for clothing predictions,
        or is the original frame for food predictions.
    """
    from utils.annotator import segment_and_annotate

    # Run both models (suppress per-batch progress bars)
    food_probs     = food_model.predict(_preprocess_food(frame),         verbose=0)[0]
    clothing_probs = clothing_model.predict(_preprocess_clothing(frame), verbose=0)[0]

    food_conf     = float(np.max(food_probs))
    clothing_conf = float(np.max(clothing_probs))

    food_class     = FOOD_CLASSES[int(np.argmax(food_probs))]
    clothing_class = CLOTHING_CLASSES[int(np.argmax(clothing_probs))]

    # ── Smarter decision: food must have a clear lead to win ──────────────────
    # Food wins ONLY IF it has >75% confidence, OR leads clothing by >30 points
    FOOD_MIN_THRESHOLD = 0.75
    FOOD_MARGIN        = 0.30

    food_wins = (food_conf >= FOOD_MIN_THRESHOLD) or \
                (food_conf - clothing_conf >= FOOD_MARGIN)

    if food_wins:
        label = food_class.replace("_", " ")
        description = f"I can see {label}. I am {food_conf:.0%} confident."
        # No segmentation for food — return original frame
        return description, frame.copy()
    else:
        color = describe_colors(frame)
        description = f"I can see a {color} {clothing_class}. I am {clothing_conf:.0%} confident."
        # Segment and annotate the clothing region
        annotated = segment_and_annotate(frame, clothing_class, color, clothing_conf)
        return description, annotated
