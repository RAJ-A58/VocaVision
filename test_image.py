"""
Quick test script for VocaVision models.
Tests a single image file (or webcam snapshot) without needing the full app.

Usage:
    # Test with an image file:
    python test_image.py --image path/to/your/image.jpg

    # Test with a live webcam snapshot (press Space to capture):
    python test_image.py --webcam
"""
import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from vision_engine import load_models, analyze_image, FOOD_CLASSES, CLOTHING_CLASSES

def draw_result(frame, description, food_probs, clothing_probs):
    """Overlays prediction bars and description text on the image."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Semi-transparent dark bar at bottom
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 200), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    # Description text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = h - 170
    for line in _wrap_text(description, max_chars=55):
        cv2.putText(display, line, (10, y), font, 0.6, (0, 255, 100), 2)
        y += 25

    # Top-3 food predictions
    top3_food = np.argsort(food_probs)[::-1][:3]
    cv2.putText(display, "Top Food:", (10, y + 10), font, 0.5, (255, 200, 0), 1)
    yf = y + 28
    for idx in top3_food:
        label = f"  {FOOD_CLASSES[idx].replace('_',' '):<16} {food_probs[idx]:.1%}"
        cv2.putText(display, label, (10, yf), font, 0.45, (255, 200, 0), 1)
        yf += 18

    # Top-3 clothing predictions
    top3_cloth = np.argsort(clothing_probs)[::-1][:3]
    cv2.putText(display, "Top Clothing:", (w // 2, y + 10), font, 0.5, (100, 200, 255), 1)
    yc = y + 28
    for idx in top3_cloth:
        label = f"  {CLOTHING_CLASSES[idx]:<16} {clothing_probs[idx]:.1%}"
        cv2.putText(display, label, (w // 2, yc), font, 0.45, (100, 200, 255), 1)
        yc += 18

    return display

def _wrap_text(text, max_chars):
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def test_image_file(path, food_model, clothing_model):
    """Tests a single image file and displays annotated result."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: Could not read image at '{path}'")
        return

    print(f"\nAnalyzing: {path}")

    # Get raw probabilities for display
    from vision_engine import _preprocess_food, _preprocess_clothing
    food_probs     = food_model.predict(_preprocess_food(frame),         verbose=0)[0]
    clothing_probs = clothing_model.predict(_preprocess_clothing(frame), verbose=0)[0]
    description    = analyze_image(food_model, clothing_model, frame)

    print(f"\nResult: {description}")
    print(f"\nFood confidence scores:")
    for i, name in enumerate(FOOD_CLASSES):
        bar = "█" * int(food_probs[i] * 30)
        print(f"  {name:<16} {food_probs[i]:.1%}  {bar}")
    print(f"\nClothing confidence scores:")
    for i, name in enumerate(CLOTHING_CLASSES):
        bar = "█" * int(clothing_probs[i] * 30)
        print(f"  {name:<16} {clothing_probs[i]:.1%}  {bar}")

    annotated = draw_result(frame, description, food_probs, clothing_probs)
    cv2.imshow("VocaVision — Test Result (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save annotated result
    out_path = os.path.splitext(path)[0] + "_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved → {out_path}")

def test_webcam(food_model, clothing_model):
    """Opens webcam. Press SPACE to capture+analyze, Q to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\nWebcam test mode — Press SPACE to analyze | Q to quit")
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Instructions overlay
        cv2.putText(frame, "SPACE = analyze  |  Q = quit",
                    (10, 30), font, 0.7, (0, 255, 200), 2)
        cv2.imshow("VocaVision — Webcam Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            snapshot = frame.copy()
            print("\nAnalyzing snapshot...")
            from vision_engine import _preprocess_food, _preprocess_clothing
            food_probs     = food_model.predict(_preprocess_food(snapshot),         verbose=0)[0]
            clothing_probs = clothing_model.predict(_preprocess_clothing(snapshot), verbose=0)[0]
            description    = analyze_image(food_model, clothing_model, snapshot)

            print(f"Result: {description}")
            annotated = draw_result(snapshot, description, food_probs, clothing_probs)
            cv2.imshow("VocaVision — Test Result (press any key)", annotated)
            cv2.waitKey(0)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="VocaVision Model Tester")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to an image file to test")
    group.add_argument("--webcam", action="store_true", help="Capture from webcam")
    args = parser.parse_args()

    print("Loading models...")
    food_model, clothing_model = load_models()

    if args.image:
        test_image_file(args.image, food_model, clothing_model)
    else:
        test_webcam(food_model, clothing_model)

if __name__ == "__main__":
    main()
