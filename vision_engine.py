"""
VocaVision — Vision Engine v3 (YOLO-Powered)
=============================================
Priority:
  1. YOLO v3 (best) — Custom YOLOv8s trained on food + clothing   → models/vocavision_yolo.pt
  2. PyTorch v2 (fallback) — MobileNetV2 GPU clothing + TF food   → models/clothing_classifier_v2.pt
  3. TF v1 (legacy fallback) — original CNN                       → models/clothing_classifier.keras

YOLO mode:
  - Single model detects ALL objects in one forward pass (~15ms GPU)
  - Bounding box crops used for precise color analysis
  - Handles multiple objects in one image simultaneously

Fallback (non-YOLO) mode:
  - Hybrid PyTorch (clothing) + TF (food) classification
  - GrabCut segmentation for color detection
"""
import os
import cv2
import numpy as np
from utils.color_detector import describe_colors

# ── Class Labels (must match training order) ──────────────────────────────────
FOOD_CLASSES = sorted([
    "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",   "waffles"
])
CLOTHING_CLASSES = [
    "T-shirt or top", "Trouser", "Pullover", "Dress",      "Coat",
    "Sandal",         "Shirt",   "Sneaker",  "Bag",        "Ankle boot"
]

# YOLO class names (must match training/yolo_classes.py ALL_CLASSES)
YOLO_CLASSES = [
    "apple_pie", "fried_rice", "hamburger", "hot_dog",    "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",      "waffles",
    "t_shirt",   "trouser",    "pullover",  "dress",      "coat",
    "sandal",    "shirt",      "sneaker",   "bag",        "ankle_boot"
]
YOLO_FOOD_IDS     = set(range(0,  10))
YOLO_CLOTHING_IDS = set(range(10, 20))

# ── Model Paths ───────────────────────────────────────────────────────────────
_DIR          = os.path.dirname(__file__)
_MODELS_DIR   = os.path.join(_DIR, "models")
_YOLO_PATH    = os.path.join(_MODELS_DIR, "vocavision_yolo.pt")   # v3 — YOLO
_FOOD_TF_PATH = os.path.join(_MODELS_DIR, "food_classifier.keras")
_CLTH_PT_PATH = os.path.join(_MODELS_DIR, "clothing_classifier_v2.pt")
_CLTH_TF_PATH = os.path.join(_MODELS_DIR, "clothing_classifier.keras")

# ── Optional: rembg background removal ───────────────────────────────────────
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

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Returns (food_model, clothing_model, using_pytorch, yolo_model, using_yolo)
    where yolo_model is an ultralytics YOLO instance if vocavision_yolo.pt exists.
    """
    # ── Try loading YOLO (v3) ─────────────────────────────────────────────────
    yolo_model  = None
    using_yolo  = False

    if os.path.exists(_YOLO_PATH):
        try:
            from ultralytics import YOLO
            print("[VocaVision] Loading YOLO detector (v3)...")
            yolo_model = YOLO(_YOLO_PATH)
            using_yolo = True
            print(f"[VocaVision] YOLO model loaded | Classes: {len(YOLO_CLASSES)} | Device: GPU" )
        except Exception as e:
            print(f"[VocaVision] YOLO load failed: {e} — falling back to v2/v1")

    # ── Food model (TF — always loaded for fallback) ───────────────────────────
    print("[VocaVision] Loading food classifier (TF)...")
    food_model = tf.keras.models.load_model(_FOOD_TF_PATH)

    # ── Clothing model: PyTorch v2 or TF v1 ──────────────────────────────────
    using_pytorch  = False
    clothing_model = None

    if os.path.exists(_CLTH_PT_PATH) and _TORCH_AVAILABLE:
        try:
            print(f"[VocaVision] Loading clothing classifier v2 (PyTorch, device={_TORCH_DEVICE})...")
            m = tvm.mobilenet_v2(weights=None)
            m.classifier[1] = torch.nn.Linear(m.last_channel, len(CLOTHING_CLASSES))
            m.load_state_dict(torch.load(_CLTH_PT_PATH, map_location=_TORCH_DEVICE,
                                         weights_only=True))
            m.eval().to(_TORCH_DEVICE)
            clothing_model = m
            using_pytorch  = True
        except Exception as e:
            print(f"[VocaVision] PyTorch clothing load failed: {e}")

    if clothing_model is None and os.path.exists(_CLTH_TF_PATH):
        print("[VocaVision] Loading clothing classifier v1 (TF fallback)...")
        clothing_model = tf.keras.models.load_model(_CLTH_TF_PATH)

    if clothing_model is None:
        raise FileNotFoundError(
            "No clothing model found. Train one:\n"
            "  python training/train_clothing_v2.py  (fast, GPU)"
        )

    if using_yolo:
        print(f"[VocaVision] Engine: YOLO v3 [ACTIVE] + TF food fallback")
    else:
        mode = "PyTorch v2 (GPU)" if using_pytorch else "TF v1 (CPU)"
        print(f"[VocaVision] Engine: Hybrid | Clothing: {mode} | Food: TF")

    if _REMBG_AVAILABLE:
        print("[VocaVision] Background removal (rembg): ENABLED [OK]")
    else:
        print("[VocaVision] Background removal (rembg): not installed")

    return food_model, clothing_model, using_pytorch, yolo_model, using_yolo


# ── Background Removal ────────────────────────────────────────────────────────

def _remove_background(frame: np.ndarray) -> np.ndarray:
    session = _get_rembg_session()
    if session is None:
        return frame
    try:
        pil_in  = _PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_out = _rembg_remove(pil_in, session=session)
        bg      = _PIL_Image.new("RGB", pil_out.size, (255, 255, 255))
        bg.paste(pil_out, mask=pil_out.split()[3])
        return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame


# ── Preprocessing (for fallback classifiers) ──────────────────────────────────

def _preprocess_food(frame: np.ndarray) -> np.ndarray:
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    arr   = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _preprocess_clothing(frame: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    arr   = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=(0, -1))


# ── PyTorch TTA Clothing (fallback) ──────────────────────────────────────────

_CLOTHING_PT_TF = (T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]) if _TORCH_AVAILABLE else None)

_TTA_TRANSFORMS = [
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.RandomHorizontalFlip(p=1), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3),
               T.Resize((256,256)), T.CenterCrop(224), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.ColorJitter(brightness=0.3), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
] if _TORCH_AVAILABLE else []

def _predict_clothing_pytorch(model, frame: np.ndarray, use_tta: bool = True) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if use_tta and _TTA_TRANSFORMS:
        preds = []
        for tf_aug in _TTA_TRANSFORMS:
            tensor = tf_aug(rgb).unsqueeze(0).to(_TORCH_DEVICE)
            with torch.no_grad():
                preds.append(F.softmax(model(tensor), dim=1).cpu().numpy()[0])
        return np.mean(preds, axis=0)
    else:
        tensor = _CLOTHING_PT_TF(rgb).unsqueeze(0).to(_TORCH_DEVICE)
        with torch.no_grad():
            return F.softmax(model(tensor), dim=1).cpu().numpy()[0]


# ── GrabCut ──────────────────────────────────────────────────────────────────

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


# ── YOLO Inference ────────────────────────────────────────────────────────────

def _analyze_with_yolo(yolo_model, frame: np.ndarray) -> tuple:
    """
    Runs the custom YOLOv8 model on the frame.

    Returns:
        description    : str   — voice-ready text listing all detected objects
        annotated      : np.ndarray — frame with bounding boxes + labels drawn
        detections     : list  — [{label, conf, bbox, is_food, color}]
    """
    from utils.annotator import _draw_fullframe_label

    conf_threshold = 0.35

    results  = yolo_model(frame, conf=conf_threshold, verbose=False)[0]
    boxes    = results.boxes

    if boxes is None or len(boxes) == 0:
        return "I could not detect any known objects.", frame.copy(), []

    annotated   = frame.copy()
    detections  = []
    desc_parts  = []

    for box in boxes:
        class_id = int(box.cls[0])
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Clamp coords to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

        raw_label = YOLO_CLASSES[class_id]
        is_food   = class_id in YOLO_FOOD_IDS
        is_cloth  = class_id in YOLO_CLOTHING_IDS

        # ── Color detection for clothing ──────────────────────────────────────
        color = None
        if is_cloth and (x2 - x1) > 20 and (y2 - y1) > 20:
            crop  = frame[y1:y2, x1:x2]
            color = describe_colors(crop)

        # ── Build display label ───────────────────────────────────────────────
        display_label = raw_label.replace("_", " ").title()
        if color:
            display_label = f"{color} {display_label}"

        detections.append({
            "label"   : display_label,
            "raw"     : raw_label,
            "conf"    : conf,
            "bbox"    : (x1, y1, x2, y2),
            "is_food" : is_food,
            "color"   : color,
        })

        # ── Draw bounding box ─────────────────────────────────────────────────
        box_color = (0, 120, 255) if is_food else (50, 220, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Label background
        label_str = f"{display_label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx1, ly1 = x1, max(0, y1 - th - 8)
        cv2.rectangle(annotated, (lx1, ly1), (lx1 + tw + 6, y1), box_color, -1)
        cv2.putText(annotated, label_str, (lx1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Build voice description ───────────────────────────────────────────────
    food_items  = [d for d in detections if d["is_food"]]
    cloth_items = [d for d in detections if not d["is_food"]]

    if food_items:
        food_names = ", ".join(d["label"] for d in food_items)
        desc_parts.append(f"I can see {food_names}.")

    if cloth_items:
        cloth_descs = [f"a {d['label']}" for d in cloth_items]
        desc_parts.append("I can see " + ", ".join(cloth_descs) + ".")

    description = " ".join(desc_parts) if desc_parts else "I can see something, but I am not sure what it is."

    return description, annotated, detections


# ── Fallback Inference (v1/v2 non-YOLO) ──────────────────────────────────────

def _analyze_fallback(food_model, clothing_model, frame: np.ndarray,
                      using_pytorch: bool) -> tuple:
    """Classic hybrid classify approach (used when YOLO is not yet trained)."""
    from utils.annotator import segment_and_annotate

    clean_frame = _remove_background(frame)

    # Food prediction
    food_probs = food_model.predict(_preprocess_food(frame), verbose=0)[0]
    food_conf  = float(np.max(food_probs))
    food_class = FOOD_CLASSES[int(np.argmax(food_probs))]

    # Clothing prediction
    if using_pytorch and _TORCH_AVAILABLE and hasattr(clothing_model, "parameters"):
        clothing_probs = _predict_clothing_pytorch(clothing_model, clean_frame, use_tta=True)
    else:
        clothing_probs = clothing_model.predict(_preprocess_clothing(clean_frame), verbose=0)[0]

    clothing_conf  = float(np.max(clothing_probs))
    clothing_class = CLOTHING_CLASSES[int(np.argmax(clothing_probs))]

    # Smart domain decision
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
        fg_mask     = _grabcut_mask(clean_frame)
        color       = describe_colors(clean_frame, fg_mask=fg_mask)
        description = f"I can see a {color} {clothing_class}. I am {clothing_conf:.0%} confident."
        annotated   = segment_and_annotate(frame, clothing_class, color, clothing_conf,
                                           fg_mask=fg_mask)
        return description, annotated, food_probs, clothing_probs


# ── Main Entry Point ──────────────────────────────────────────────────────────

def analyze_image(food_model, clothing_model, frame: np.ndarray,
                  using_pytorch: bool = True,
                  yolo_model=None, using_yolo: bool = False) -> tuple:
    """
    Unified inference entry point.

    Args:
        food_model, clothing_model : loaded models
        frame                      : BGR numpy array
        using_pytorch              : whether clothing model is PyTorch
        yolo_model                 : ultralytics YOLO instance (or None)
        using_yolo                 : True when yolo_model is available

    Returns:
        (description: str, annotated_frame: np.ndarray,
         food_probs: np.ndarray, clothing_probs: np.ndarray)
    """
    if using_yolo and yolo_model is not None:
        description, annotated, detections = _analyze_with_yolo(yolo_model, frame)

        # Build dummy prob arrays for the confidence chart in app.py
        food_probs     = np.zeros(len(FOOD_CLASSES),     dtype=np.float32)
        clothing_probs = np.zeros(len(CLOTHING_CLASSES), dtype=np.float32)
        for det in detections:
            if det["is_food"]:
                idx = FOOD_CLASSES.index(det["raw"]) if det["raw"] in FOOD_CLASSES else -1
                if idx >= 0:
                    food_probs[idx] = max(food_probs[idx], det["conf"])
            else:
                yolo_to_clothing = {
                    "t_shirt": "T-shirt or top", "trouser": "Trouser",
                    "pullover": "Pullover", "dress": "Dress", "coat": "Coat",
                    "sandal": "Sandal", "shirt": "Shirt", "sneaker": "Sneaker",
                    "bag": "Bag", "ankle_boot": "Ankle boot"
                }
                standard = yolo_to_clothing.get(det["raw"])
                if standard and standard in CLOTHING_CLASSES:
                    idx = CLOTHING_CLASSES.index(standard)
                    clothing_probs[idx] = max(clothing_probs[idx], det["conf"])

        return description, annotated, food_probs, clothing_probs

    else:
        # Fallback to v2/v1 classifier pipeline
        return _analyze_fallback(food_model, clothing_model, frame, using_pytorch)
