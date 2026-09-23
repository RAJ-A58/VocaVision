"""
Vision Engine v2 — Hybrid Inference (PyTorch GPU + TensorFlow fallback)

Clothing model:
  - v2 (preferred): PyTorch MobileNetV2, trained with GPU → models/clothing_classifier_v2.pt
  - v1 (fallback):  TF Keras custom CNN               → models/clothing_classifier.keras

Food model:
  - TF Keras MobileNetV2 fine-tuned on Food-101       → models/food_classifier.keras

Additional features:
  - Background removal via rembg (U²-Net) before inference
  - Test-Time Augmentation (TTA) on clothing for +2-3% accuracy
  - GrabCut segmentation + visual annotation on output image
"""
import os
import cv2
import numpy as np
from utils.color_detector import describe_colors

# ── Class Labels ─────────────────────────────────────────────────────────────
FOOD_CLASSES = sorted([
    "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",   "waffles"
])
CLOTHING_CLASSES = [
    "T-shirt or top", "Trouser", "Pullover", "Dress",  "Coat",
    "Sandal",         "Shirt",   "Sneaker",  "Bag",    "Ankle boot"
]

# ── Model Paths ───────────────────────────────────────────────────────────────
_DIR              = os.path.dirname(__file__)
_MODELS_DIR       = os.path.join(_DIR, "models")
_FOOD_TF_PATH     = os.path.join(_MODELS_DIR, "food_classifier.keras")
_CLTH_PT_PATH     = os.path.join(_MODELS_DIR, "clothing_classifier_v2.pt")   # PyTorch v2
_CLTH_TF_PATH     = os.path.join(_MODELS_DIR, "clothing_classifier.keras")   # TF v1 fallback

# ── Optional: background removal ─────────────────────────────────────────────
try:
    from rembg import remove as _rembg_remove, new_session as _rembg_new_session
    from PIL import Image as _PIL_Image
    _REMBG_AVAILABLE = True
except Exception:
    _REMBG_AVAILABLE = False

_REMBG_SESSION = None

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and _REMBG_AVAILABLE:
        try:
            # u2netp is a lightweight 4.5MB model (vs 1GB default)
            _REMBG_SESSION = _rembg_new_session("u2netp")
        except Exception:
            _REMBG_SESSION = False
    return _REMBG_SESSION if _REMBG_SESSION else None

# ── Optional: PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.models as tvm
    _TORCH_AVAILABLE = True
    _TORCH_DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH_AVAILABLE = False

# ── TensorFlow (food model + clothing fallback) ────────────────────────────────
import tensorflow as tf


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Loads food model (TF) and clothing model (PyTorch v2 if available, else TF v1).
    Returns (food_model, clothing_model, using_pytorch_clothing: bool)
    """
    # Food model — always TF
    if not os.path.exists(_FOOD_TF_PATH):
        raise FileNotFoundError(
            f"Food model not found at '{_FOOD_TF_PATH}'.\n"
            "  Train it first:  python training/train_food.py"
        )
    print("[VocaVision] Loading food classifier (TF)...")
    food_model = tf.keras.models.load_model(_FOOD_TF_PATH)

    # Clothing model — prefer PyTorch v2
    if _TORCH_AVAILABLE and os.path.exists(_CLTH_PT_PATH):
        print(f"[VocaVision] Loading clothing classifier v2 (PyTorch, device={_TORCH_DEVICE})...")
        checkpoint = torch.load(_CLTH_PT_PATH, map_location=_TORCH_DEVICE)
        clothing_model = tvm.mobilenet_v2(weights=None)
        in_features = clothing_model.classifier[1].in_features
        import torch.nn as nn
        clothing_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(CLOTHING_CLASSES)),
        )
        clothing_model.load_state_dict(checkpoint["model_state"])
        clothing_model.to(_TORCH_DEVICE).eval()
        using_pytorch = True
        print("[VocaVision] Both models loaded  |  Clothing: PyTorch v2 (GPU)  |  Food: TF")
    elif os.path.exists(_CLTH_TF_PATH):
        print("[VocaVision] Loading clothing classifier v1 (TF fallback)...")
        clothing_model = tf.keras.models.load_model(_CLTH_TF_PATH)
        using_pytorch  = False
        print("[VocaVision] Both models loaded  |  Clothing: TF v1 (CPU)  |  Food: TF")
    else:
        raise FileNotFoundError(
            "No clothing model found. Train one:\n"
            "  python training/train_clothing_v2.py  (recommended, GPU)\n"
            "  python training/train_clothing.py     (CPU fallback)"
        )

    if _REMBG_AVAILABLE:
        print("[VocaVision] Background removal (rembg): ENABLED [OK]")
    else:
        print("[VocaVision] Background removal (rembg): not installed (pip install rembg)")

    return food_model, clothing_model, using_pytorch


# ── Background Removal ────────────────────────────────────────────────────────

def _remove_background(frame: np.ndarray) -> np.ndarray:
    """
    Removes background using rembg (U²-Net). Returns BGR frame
    with background replaced by white pixels. Falls back to original
    frame if rembg is not available.
    """
    session = _get_rembg_session()
    if session is None:
        return frame
    try:
        pil_in  = _PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_out = _rembg_remove(pil_in, session=session)          # RGBA with transparent bg
        bg      = _PIL_Image.new("RGB", pil_out.size, (255, 255, 255))
        bg.paste(pil_out, mask=pil_out.split()[3])               # paste on white bg
        return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame


# ── Preprocessing ─────────────────────────────────────────────────────────────

_CLOTHING_PT_TF = T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]) if _TORCH_AVAILABLE else None

# TTA augments (slight variation, averaged for better accuracy)
_TTA_TRANSFORMS = [
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((236,236)),
               T.CenterCrop(224),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
] if _TORCH_AVAILABLE else []


def _preprocess_food(frame: np.ndarray) -> np.ndarray:
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def _preprocess_clothing(frame: np.ndarray) -> np.ndarray:
    """Legacy TF preprocessing (28×28 grayscale) — used only if PyTorch not available."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img  = cv2.resize(gray, (28, 28)).astype(np.float32) / 255.0
    return img.reshape(1, 28, 28, 1)


def _predict_clothing_pytorch(model, frame: np.ndarray, use_tta: bool = True) -> np.ndarray:
    """Runs PyTorch clothing model with optional Test-Time Augmentation."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if use_tta and _TTA_TRANSFORMS:
        preds = []
        for tf_aug in _TTA_TRANSFORMS:
            tensor = tf_aug(rgb).unsqueeze(0).to(_TORCH_DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                preds.append(F.softmax(logits, dim=1).cpu().numpy()[0])
        return np.mean(preds, axis=0)
    else:
        tensor = _CLOTHING_PT_TF(rgb).unsqueeze(0).to(_TORCH_DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            return F.softmax(logits, dim=1).cpu().numpy()[0]


# ── GrabCut mask ──────────────────────────────────────────────────────────────

def _grabcut_mask(frame: np.ndarray) -> np.ndarray:
    h, w   = frame.shape[:2]
    mask   = np.zeros((h, w), np.uint8)
    bgd    = np.zeros((1, 65), np.float64)
    fgd    = np.zeros((1, 65), np.float64)
    mx, my = int(w * 0.05), int(h * 0.05)
    rect   = (mx, my, w - 2*mx, h - 2*my)
    try:
        cv2.grabCut(frame, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        return np.ones((h, w), np.uint8)


# ── Main Inference ────────────────────────────────────────────────────────────

def analyze_image(food_model, clothing_model, frame: np.ndarray,
                  using_pytorch: bool = True) -> tuple:
    """
    Runs both classifiers on the frame.

    Pipeline:
      1. Remove background (if rembg installed) → cleaner input
      2. Run food classifier (TF)
      3. Run clothing classifier (PyTorch v2 with TTA, or TF v1)
      4. Smart decision: food wins only if very confident
      5. Clothing path: GrabCut segment → color detect on fg pixels → annotate

    Returns:
        (description: str, annotated_frame: np.ndarray)
    """
    from utils.annotator import segment_and_annotate

    # Step 1: Remove background for cleaner inference
    clean_frame = _remove_background(frame)

    # Step 2: Food prediction (run on original frame — Food-101 trained on natural scene/table context)
    food_probs = food_model.predict(_preprocess_food(frame), verbose=0)[0]
    food_conf  = float(np.max(food_probs))
    food_class = FOOD_CLASSES[int(np.argmax(food_probs))]

    # Step 3: Clothing prediction (PyTorch v2 with TTA, or TF v1)
    if using_pytorch and _TORCH_AVAILABLE and hasattr(clothing_model, 'parameters'):
        clothing_probs = _predict_clothing_pytorch(clothing_model, clean_frame, use_tta=True)
    else:
        clothing_probs = clothing_model.predict(_preprocess_clothing(clean_frame), verbose=0)[0]

    clothing_conf  = float(np.max(clothing_probs))
    clothing_class = CLOTHING_CLASSES[int(np.argmax(clothing_probs))]

    # Step 4: Smart domain decision (Food vs Clothing)
    #
    # Principles:
    # 1. On food images (pizza, sushi, etc.), Fashion-MNIST lacks an "Other" class
    #    and dumps probability into "Bag" (the generic round/textured catch-all).
    # 2. When food is recognized with solid confidence (>= 50%), it should not be
    #    trumped by a "Bag" false positive.
    # 3. On real clothing (T-shirts, trousers, coats), food confidence is low (< 40%).
    if food_conf >= 0.50 and clothing_class == "Bag":
        food_wins = True
    elif food_conf >= 0.55:
        food_wins = True
    elif food_conf > clothing_conf:
        food_wins = True
    else:
        food_wins = False

    if food_wins:
        label       = food_class.replace("_", " ")
        description = f"I can see {label}. I am {food_conf:.0%} confident."
        return description, frame.copy(), food_probs, clothing_probs
    else:
        # Step 5: Detect color on clothing pixels only, then annotate
        fg_mask     = _grabcut_mask(clean_frame)
        color       = describe_colors(clean_frame, fg_mask=fg_mask)
        description = f"I can see a {color} {clothing_class}. I am {clothing_conf:.0%} confident."
        annotated   = segment_and_annotate(frame, clothing_class, color, clothing_conf,
                                           fg_mask=fg_mask)
        return description, annotated, food_probs, clothing_probs

