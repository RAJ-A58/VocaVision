"""
VocaVision — Bounding Box Demo
==============================
Draws annotated bounding boxes on any image showing:
  [FOOD]  Pizza  94%
  [CLOTH] Blue Shirt  88%

Uses matplotlib for display (works even with headless OpenCV).

Usage:
    # Browse & select a photo using a file picker popup:
    python demo_bbox.py --upload

    # On an image file (direct path):
    python demo_bbox.py --image path/to/photo.jpg

    # On your webcam (press ENTER to capture, Ctrl+C to quit):
    python demo_bbox.py --webcam

    # Save only (no display window):
    python demo_bbox.py --upload --no-show

Output:
    - A window showing the annotated image (via matplotlib)
    - Saved file: <original_name>_bbox.jpg
"""

import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import vision_engine as _ve
from vision_engine import (
    load_models, analyze_image,
    FOOD_CLASSES, CLOTHING_CLASSES, YOLO_CLASSES,
    YOLO_FOOD_IDS, YOLO_CLOTHING_IDS,
)
from utils.color_detector import describe_colors

# Private constants — access via getattr (safe even if names change)
_YOLO_DEFAULT_THRESHOLD = getattr(_ve, "_YOLO_DEFAULT_THRESHOLD", 0.52)
_YOLO_CLASS_THRESHOLDS  = getattr(_ve, "_YOLO_CLASS_THRESHOLDS",  {18: 0.75, 17: 0.70, 15: 0.65, 19: 0.65})
_YOLO_DISPLAY_NAMES     = getattr(_ve, "_YOLO_DISPLAY_NAMES", {})

# ── Matplotlib for display (works with headless OpenCV) ───────────────────────
import matplotlib
# Try backends in order until one works (TkAgg is best on Windows)
for _backend in ("TkAgg", "Qt5Agg", "WXAgg", "Agg"):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_test
        _plt_test.figure()
        _plt_test.close("all")
        break
    except Exception:
        continue
import matplotlib.pyplot as plt

# ── Color palette ─────────────────────────────────────────────────────────────
_FOOD_BOX_COLOR  = (0, 180, 255)
_CLOTH_BOX_COLOR = (50, 220, 80)

_COLOR_NAME_TO_BGR = {
    "red":     (0,   0,   220),
    "orange":  (0,   140, 255),
    "yellow":  (0,   220, 220),
    "green":   (0,   180, 0),
    "blue":    (220, 80,  0),
    "navy":    (128, 20,  0),
    "purple":  (180, 0,   180),
    "pink":    (180, 80,  200),
    "white":   (220, 220, 220),
    "black":   (60,  60,  60),
    "gray":    (150, 150, 150),
    "brown":   (30,  80,  130),
    "beige":   (180, 210, 220),
    "neutral": (160, 160, 160),
}


def _color_to_bgr(color_name: str):
    first = (color_name or "neutral").split()[0].lower()
    return _COLOR_NAME_TO_BGR.get(first, (100, 255, 100))


# ── Display helper (replaces cv2.imshow) ─────────────────────────────────────

def show_image(title: str, bgr_frame: np.ndarray):
    """Display a BGR OpenCV image using matplotlib (no GUI OpenCV needed)."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 7), facecolor="#111111")
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(title, color="white", fontsize=13, pad=10)
    plt.tight_layout(pad=0.5)
    plt.show()


# ── Threshold helper ─────────────────────────────────────────────────────────

def _passes_threshold(class_id: int, conf: float) -> bool:
    return conf >= _YOLO_CLASS_THRESHOLDS.get(class_id, _YOLO_DEFAULT_THRESHOLD)


# ── YOLO bounding box drawing ─────────────────────────────────────────────────

def draw_yolo_boxes(frame: np.ndarray, yolo_model) -> tuple:
    """
    Runs YOLO on frame and draws rich bounding boxes.

    Box format:
      ┌──────────────────────────────┐
      │ [FOOD]  Pizza  94%           │   ← filled banner (orange)
      │                              │
      │  ✦ corner ticks (modern UI) │
      └──────────────────────────────┘

      ┌──────────────────────────────┐
      │ [CLOTH]  Blue Shirt  88%     │   ← filled banner (matches clothing color)
      └──────────────────────────────┘

    Bottom bar: "I can see a Blue Shirt and a Pizza."

    Returns (annotated_frame, detections, description)
    """
    results = yolo_model(frame, conf=_YOLO_DEFAULT_THRESHOLD, verbose=False)[0]
    boxes   = results.boxes

    annotated  = frame.copy()
    detections = []

    if boxes is None or len(boxes) == 0:
        _draw_msg(annotated, "No objects detected.")
        return annotated, [], "No objects detected."

    food_items  = []
    cloth_items = []

    for box in boxes:
        class_id = int(box.cls[0])
        conf     = float(box.conf[0])
        if not _passes_threshold(class_id, conf):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        raw_label = YOLO_CLASSES[class_id]
        is_food   = class_id in YOLO_FOOD_IDS
        is_cloth  = class_id in YOLO_CLOTHING_IDS

        # Color detection for clothing only
        color_name = None
        if is_cloth and (x2 - x1) > 20 and (y2 - y1) > 20:
            crop       = frame[y1:y2, x1:x2]
            color_name = describe_colors(crop)

        # Display name
        if is_food:
            display_name = raw_label.replace("_", " ").title()
            category_tag = "FOOD"
            box_bgr      = _FOOD_BOX_COLOR
        else:
            display_name = _YOLO_DISPLAY_NAMES.get(raw_label,
                           raw_label.replace("_", " ").title())
            category_tag = "CLOTH"
            box_bgr      = _color_to_bgr(color_name) if color_name else _CLOTH_BOX_COLOR

        # Label text
        if color_name:
            label_str = f"[{category_tag}]  {color_name.title()} {display_name}  {conf:.0%}"
        else:
            label_str = f"[{category_tag}]  {display_name}  {conf:.0%}"

        _draw_box_with_label(annotated, x1, y1, x2, y2, label_str, box_bgr)

        det = dict(label=label_str, raw=raw_label, conf=conf,
                   bbox=(x1, y1, x2, y2), is_food=is_food,
                   color=color_name, display_name=display_name,
                   category=category_tag)
        detections.append(det)
        (food_items if is_food else cloth_items).append(det)

    if not detections:
        _draw_msg(annotated, "Nothing detected above confidence threshold.")
        return annotated, [], "Nothing detected."

    description = _build_description(food_items, cloth_items)
    _draw_description_banner(annotated, description)
    return annotated, detections, description


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_box_with_label(img, x1, y1, x2, y2, label, color):
    """Thick box + corner ticks + filled label banner above box."""
    # Main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # Corner accent ticks (modern look)
    tick, lw = 20, 5
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * tick, cy), color, lw)
        cv2.line(img, (cx, cy), (cx, cy + dy * tick), color, lw)

    # Label banner
    font  = cv2.FONT_HERSHEY_DUPLEX
    fscale = 0.62
    thick  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, fscale, thick)
    pad = 6
    bx1 = x1
    by1 = max(0, y1 - th - pad * 2)
    by2 = y1

    cv2.rectangle(img, (bx1, by1), (bx1 + tw + pad * 2, by2), color, -1)
    cv2.putText(img, label, (bx1 + pad, by2 - pad),
                font, fscale, (255, 255, 255), thick, cv2.LINE_AA)


def _draw_description_banner(img, text):
    """Semi-transparent bottom bar with full description."""
    h, w  = img.shape[:2]
    bar_h = 50
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 140), 2, cv2.LINE_AA)


def _draw_msg(img, text):
    cv2.putText(img, text, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2, cv2.LINE_AA)


# ── Natural language description ──────────────────────────────────────────────

def _oxford(items):
    if len(items) == 1:  return items[0]
    if len(items) == 2:  return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _build_description(food_items, cloth_items) -> str:
    parts = []
    if food_items:
        names = [f"a {d['display_name']}" for d in food_items]
        parts.append("I can see " + _oxford(names) + ".")
    if cloth_items:
        descs = [f"a {(d['color'].title() + ' ') if d['color'] else ''}{d['display_name']}"
                 for d in cloth_items]
        connector = "I can also see" if parts else "I can see"
        parts.append(f"{connector} " + _oxford(descs) + ".")
    return " ".join(parts) if parts else "I could not identify anything."


# ── Fallback (no YOLO) ────────────────────────────────────────────────────────

def draw_fallback_boxes(frame, food_model, clothing_model, using_pytorch):
    from vision_engine import _analyze_fallback
    description, annotated, _, _ = _analyze_fallback(
        food_model, clothing_model, frame, using_pytorch
    )
    _draw_description_banner(annotated, description)
    return annotated, description


# ── Run on a single image file ────────────────────────────────────────────────

def run_on_image(path, food_model, clothing_model, using_pytorch,
                 yolo_model, using_yolo, show=True):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: '{path}'")
        return

    print(f"\n🔍 Analyzing: {path}")

    if using_yolo and yolo_model:
        annotated, detections, description = draw_yolo_boxes(frame, yolo_model)
        print(f"✅ Engine   : YOLO v3")
        print(f"📦 Objects  : {len(detections)} detected")
        for d in detections:
            print(f"   {d['label']}")
    else:
        print("⚠️  YOLO unavailable — using fallback classifier")
        annotated, description = draw_fallback_boxes(
            frame, food_model, clothing_model, using_pytorch)

    print(f"🔊 Result   : {description}")

    # Save
    base, _ = os.path.splitext(path)
    out_path = base + "_bbox.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"💾 Saved    → {out_path}")

    if show:
        show_image("VocaVision — Close window to exit", annotated)


# ── Webcam mode ───────────────────────────────────────────────────────────────

def run_on_webcam(food_model, clothing_model, using_pytorch, yolo_model, using_yolo):
    """
    Live webcam loop using only OpenCV for capture.
    Preview window rendered via matplotlib on SPACE press.
    Press Q in the terminal or close the matplotlib window to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera connection.")
        return

    print("\n📷 Webcam ready!")
    print("   ▶  Press  ENTER  in this terminal to capture & analyze")
    print("   ▶  Press  Ctrl+C  to quit\n")

    try:
        while True:
            input("⏎  Press ENTER to capture a frame (Ctrl+C to quit): ")

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                continue

            print("⏳ Analyzing...")

            if using_yolo and yolo_model:
                annotated, detections, description = draw_yolo_boxes(frame, yolo_model)
                print(f"✅ {len(detections)} detection(s): {description}")
                for d in detections:
                    print(f"   {d['label']}")
            else:
                annotated, description = draw_fallback_boxes(
                    frame, food_model, clothing_model, using_pytorch)
                print(f"✅ {description}")

            # Save
            cv2.imwrite("webcam_result_bbox.jpg", annotated)
            print("💾 Saved → webcam_result_bbox.jpg")

            # Show via matplotlib
            show_image(f"VocaVision — {description}", annotated)

    except KeyboardInterrupt:
        print("\n👋 Exiting webcam mode.")
    finally:
        cap.release()


# ── File browser (tkinter — built into Python, no install needed) ─────────────

def browse_for_image() -> str:
    """Opens a native Windows file-picker popup and returns the selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()          # Hide the empty root window
        root.lift()
        root.attributes("-topmost", True)   # Popup appears on top

        path = filedialog.askopenfilename(
            title="VocaVision — Select an Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return path  # Empty string if user cancels
    except Exception as e:
        print(f"[ERROR] Could not open file picker: {e}")
        print("       Try using:  python demo_bbox.py --image <path>")
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VocaVision — Bounding box demo with FOOD/CLOTH labels + color"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--upload", action="store_true",
                       help="Open a file browser to pick an image")
    group.add_argument("--image",  type=str,
                       help="Direct path to an image file")
    group.add_argument("--webcam", action="store_true",
                       help="Use live webcam")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip display window (just save the result)")
    args = parser.parse_args()

    print("=" * 55)
    print("  VocaVision — Bounding Box Demo")
    print("=" * 55)

    # ── Resolve image path ────────────────────────────────────────────────────
    image_path = None

    if args.upload:
        print("\n📂 Opening file browser...")
        image_path = browse_for_image()
        if not image_path:
            print("❌ No file selected. Exiting.")
            return
        print(f"✅ Selected: {image_path}")

    elif args.image:
        image_path = args.image

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n⏳ Loading models...")
    food_model, clothing_model, using_pytorch, yolo_model, using_yolo = load_models()

    if using_yolo:
        print("✅ Engine: YOLO v3 [ACTIVE]")
    else:
        print("⚠️  Engine: Fallback Classifier")

    # ── Run ───────────────────────────────────────────────────────────────────
    if image_path:
        run_on_image(
            image_path,
            food_model, clothing_model, using_pytorch,
            yolo_model, using_yolo,
            show=not args.no_show,
        )
    else:
        run_on_webcam(food_model, clothing_model, using_pytorch, yolo_model, using_yolo)


if __name__ == "__main__":
    main()
