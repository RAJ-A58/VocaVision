"""
VocaVision — Gradio Web Interface (v3 — YOLO-Powered)

Engine priority:
  1. YOLO v3  — vocavision_yolo.pt (custom trained, detects ALL objects + bounding boxes)
  2. Hybrid v2 — PyTorch MobileNetV2 (GPU) + TF food model (fallback while YOLO trains)
"""
import os
import sys
import tempfile

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from vision_engine import (
    CLOTHING_CLASSES, FOOD_CLASSES,
    analyze_image, load_models,
)
from utils.color_detector import describe_colors

# ── Load all models once at startup ──────────────────────────────────────────
print("Loading models...")
food_model, clothing_model, _USING_PYTORCH, _YOLO_MODEL, _USING_YOLO = load_models()
if _USING_YOLO:
    print("Models ready.  [Engine: YOLO v3 — ACTIVE]")
else:
    print("Models ready.  [Engine: Hybrid v2 — YOLO not trained yet]")


# ── Core prediction function ──────────────────────────────────────────────────

def predict(pil_image: Image.Image):
    if pil_image is None:
        return "Please upload an image.", None, None, None

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    description, annotated_bgr, food_probs, clothing_probs = analyze_image(
        food_model, clothing_model, frame,
        using_pytorch=_USING_PYTORCH,
        yolo_model=_YOLO_MODEL,
        using_yolo=_USING_YOLO,
    )

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    audio_path    = _generate_audio(description)
    fig           = _make_confidence_chart(food_probs, clothing_probs)

    return description, audio_path, fig, annotated_pil


def _generate_audio(text: str) -> str:
    tts = gTTS(text=text, lang="en", slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name


def _make_confidence_chart(food_probs, clothing_probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    def _bar(ax, probs, labels, title, highlight_color, bar_color):
        ax.set_facecolor("#16213e")
        colors = [highlight_color if p == max(probs) else bar_color for p in probs]
        bars   = ax.barh(labels, probs, color=colors, edgecolor="none", height=0.6)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.xaxis.set_visible(False)
        for bar, prob in zip(bars, probs):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", color="white", fontsize=9)

    food_labels = [c.replace("_", " ").title() for c in FOOD_CLASSES]
    _bar(ax1, food_probs,     food_labels,      "Food Classifier",     "#2ecc71", "#3498db")
    _bar(ax2, clothing_probs, CLOTHING_CLASSES, "Clothing Classifier", "#f39c12", "#e74c3c")

    plt.tight_layout(pad=2)
    return fig


# ── Gradio Interface ──────────────────────────────────────────────────────────

_ENGINE_TAG = "YOLO v3 (Custom Trained)" if _USING_YOLO else "Hybrid v2 (PyTorch GPU + TF)"

_DESCRIPTION = f"""
## VocaVision — AI-Based Assistive Recognition System

An assistive AI that identifies **clothing colors/types** and **food items** in a single image,  
then describes them through **voice output** to support independent daily activities.

**Active Engine:** `{_ENGINE_TAG}`

**How to use:**
1. Upload a photo of **food** or **clothing** (or both!)
2. The AI detects and identifies what it sees
3. Listen to the spoken description

**Powered by:**
- YOLO v3 — Custom YOLOv8s trained on 20 food + clothing classes
- K-Means Color Detector — Dominant color extraction from detected regions
"""

_FOOD_CLASSES_STR  = " • ".join([c.replace("_", " ").title() for c in FOOD_CLASSES])
_CLOTH_CLASSES_STR = " • ".join(CLOTHING_CLASSES)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Upload an image of food or clothing",
    ),
    outputs=[
        gr.Textbox(label="AI Description", lines=2),
        gr.Audio(label="Voice Output", type="filepath"),
        gr.Plot(label="Model Confidence Scores"),
        gr.Image(label="Detected & Annotated", type="pil"),
    ],
    title="VocaVision",
    description=_DESCRIPTION,
    article=f"""
---
**Recognisable Food:** {_FOOD_CLASSES_STR}  
**Recognisable Clothing:** {_CLOTH_CLASSES_STR}
""",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="blue",
        )
    )
