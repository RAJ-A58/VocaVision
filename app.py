"""
VocaVision — Gradio Web Interface for Hugging Face Spaces

Replaces the desktop webcam app with a browser-based image upload interface.
Uses gTTS (cloud TTS) instead of pyttsx3 since pyttsx3 doesn't work on servers.

The models are loaded once at startup for fast inference.
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

# ── Ensure project modules are importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from vision_engine import (
    CLOTHING_CLASSES, FOOD_CLASSES,
    _preprocess_clothing, _preprocess_food,
    analyze_image, load_models,
)
from utils.color_detector import describe_colors

# ── Load both models once at startup (not on every request) ───────────────────
print("Loading models...")
food_model, clothing_model = load_models()
print("Models ready.")

# ── Core prediction function ──────────────────────────────────────────────────

def predict(pil_image: Image.Image):
    """
    Takes a PIL image, runs both classifiers, generates:
      1. A text description
      2. A spoken audio file (MP3 via gTTS)
      3. A confidence bar chart figure
    """
    if pil_image is None:
        return "Please upload an image.", None, None

    # Convert PIL (RGB) → OpenCV (BGR) for our preprocessing functions
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Get raw probability arrays from both models
    food_probs     = food_model.predict(_preprocess_food(frame),         verbose=0)[0]
    clothing_probs = clothing_model.predict(_preprocess_clothing(frame), verbose=0)[0]

    # Get the natural-language description
    description = analyze_image(food_model, clothing_model, frame)

    # Generate spoken audio using gTTS
    audio_path = _generate_audio(description)

    # Build the confidence chart
    fig = _make_confidence_chart(food_probs, clothing_probs)

    return description, audio_path, fig


def _generate_audio(text: str) -> str:
    """Converts text to speech and saves as a temp MP3 file."""
    tts = gTTS(text=text, lang="en", slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name


def _make_confidence_chart(food_probs, clothing_probs):
    """Builds a side-by-side horizontal bar chart for both classifiers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    def _bar(ax, probs, labels, title, highlight_color, bar_color):
        ax.set_facecolor("#16213e")
        colors = [highlight_color if p == max(probs) else bar_color for p in probs]
        bars = ax.barh(labels, probs, color=colors, edgecolor="none", height=0.6)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.xaxis.set_visible(False)
        for bar, prob in zip(bars, probs):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", color="white", fontsize=9)

    food_labels  = [c.replace("_", " ").title() for c in FOOD_CLASSES]
    _bar(ax1, food_probs,     food_labels,     "🍕 Food Classifier",    "#2ecc71", "#3498db")
    _bar(ax2, clothing_probs, CLOTHING_CLASSES, "👕 Clothing Classifier", "#f39c12", "#e74c3c")

    plt.tight_layout(pad=2)
    return fig

# ── Gradio Interface ──────────────────────────────────────────────────────────

_DESCRIPTION = """
## 🎙️ VocaVision — AI-Based Assistive Recognition System

An assistive AI that identifies **clothing colors/patterns** and **food items**,  
then describes them through **voice output** to support independent daily activities.

**How to use:**  
1. Upload a photo of **food** or **clothing**  
2. The AI identifies what it sees  
3. Listen to the spoken description  

**Powered by:**  
- 🍕 **Food Classifier** — MobileNetV2 fine-tuned on Food-101 (10 classes)  
- 👕 **Clothing Classifier** — Custom CNN trained on Fashion MNIST (10 classes)  
- 🎨 **Color Detector** — K-Means clustering on dominant pixel colors  
"""

_FOOD_CLASSES_STR    = " • ".join([c.replace("_"," ").title() for c in FOOD_CLASSES])
_CLOTHES_CLASSES_STR = " • ".join(CLOTHING_CLASSES)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="📤 Upload an image of food or clothing",
    ),
    outputs=[
        gr.Textbox(
            label="📝 AI Description",
            lines=2,
        ),
        gr.Audio(
            label="🔊 Voice Output",
            type="filepath",
        ),
        gr.Plot(
            label="📊 Model Confidence Scores",
        ),
    ],
    title="🎙️ VocaVision",
    description=_DESCRIPTION,
    article=f"""
---
**Recognisable Food:** {_FOOD_CLASSES_STR}  
**Recognisable Clothing:** {_CLOTHES_CLASSES_STR}
""",
    flagging_mode="never",   # replaces allow_flagging in Gradio 6.0
)

if __name__ == "__main__":
    # theme moved from Interface() to launch() in Gradio 6.0
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="blue",
        )
    )
