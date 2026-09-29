"""
VocaVision — Gradio Web Interface (v5 — Live Video Feed)

Replaces static image upload with a real-time webcam stream.
The webcam feed is processed every `STREAM_EVERY` seconds; bounding
boxes and color labels are drawn on the annotated output frame that
updates live beside the camera feed.

Voice output is generated only when the detected objects change,
preventing the TTS from firing on every single frame.

Engine priority (unchanged from vision_engine.py):
  1. YOLO v3  — vocavision_yolo.pt
  2. Hybrid v2 — PyTorch MobileNetV2 + TF food fallback
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

# ── Load all models once at startup ──────────────────────────────────────────
print("Loading models...")
food_model, clothing_model, _USING_PYTORCH, _YOLO_MODEL, _USING_YOLO = load_models()
_ENGINE_TAG = "YOLO v3 (Custom Trained)" if _USING_YOLO else "Hybrid v2 (PyTorch + TF)"
print(f"Models ready.  [Engine: {_ENGINE_TAG}]")

# How often (seconds) a webcam frame is sent to the inference function.
# Lower = more responsive but higher CPU/GPU usage.
STREAM_EVERY = 1.5


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_frame(pil_image: Image.Image, last_desc: str):
    """
    Called by Gradio's .stream() for every captured webcam frame.

    Args:
        pil_image : latest webcam frame (PIL RGB)
        last_desc : the description produced on the previous frame
                    (stored in gr.State — used to throttle audio output)

    Returns:
        annotated_pil : BGR→PIL annotated image with bounding boxes
        description   : natural-language sentence about detected objects
        audio_path    : path to MP3 (or None if nothing changed)
        fig           : matplotlib confidence chart
        new_last_desc : updated state for the next call
    """
    if pil_image is None:
        return None, "Waiting for camera...", None, None, last_desc

    try:
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        description, annotated_bgr, food_probs, clothing_probs = analyze_image(
            food_model, clothing_model, frame,
            using_pytorch=_USING_PYTORCH,
            yolo_model=_YOLO_MODEL,
            using_yolo=_USING_YOLO,
        )

        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        fig           = _make_confidence_chart(food_probs, clothing_probs)

        # ── Audio throttle: only speak when detection changes ─────────────────
        audio_path = None
        if description != last_desc:
            audio_path = _generate_audio(description)

        return annotated_pil, description, audio_path, fig, description

    except Exception as e:
        print(f"[predict_frame ERROR] {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        # Return safe fallback values — keeps stream alive
        blank = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        return blank, f"Error: {type(e).__name__}", None, None, last_desc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_audio(text: str) -> str | None:
    """Generate a TTS MP3 for the given text. Returns None on any failure."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[TTS ERROR] {type(e).__name__}: {e}")
        return None


def _make_confidence_chart(food_probs, clothing_probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    def _bar(ax, probs, labels, title, highlight_color, bar_color):
        ax.set_facecolor("#16213e")
        max_val = max(probs) if max(probs) > 0 else 1.0
        colors  = [highlight_color if p == max_val else bar_color for p in probs]
        bars    = ax.barh(labels, probs, color=colors, edgecolor="none", height=0.6)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.xaxis.set_visible(False)
        for bar, prob in zip(bars, probs):
            if prob > 0:
                ax.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.1%}", va="center", color="white", fontsize=9)

    food_labels = [c.replace("_", " ").title() for c in FOOD_CLASSES]
    _bar(ax1, food_probs,     food_labels,      "Food Classifier",     "#2ecc71", "#3498db")
    _bar(ax2, clothing_probs, CLOTHING_CLASSES, "Clothing Classifier", "#f39c12", "#e74c3c")

    plt.tight_layout(pad=2)
    return fig


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_FOOD_CLASSES_STR  = " • ".join([c.replace("_", " ").title() for c in FOOD_CLASSES])
_CLOTH_CLASSES_STR = " • ".join(CLOTHING_CLASSES)

with gr.Blocks(title="VocaVision — Live Detection") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(f"""
# 🎙️👁️ VocaVision — Live Detection
**AI-Based Assistive Recognition System**

Point your camera at **clothing** or **food** — the AI detects and describes it in real time.  
Voice output fires automatically whenever the detected object changes.

**Active Engine:** `{_ENGINE_TAG}` &nbsp;|&nbsp; **Processing rate:** every `{STREAM_EVERY}s`
""")

    # ── State: tracks last description to throttle TTS ────────────────────────
    last_desc_state = gr.State(value="")

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        webcam_feed = gr.Image(
            sources=["webcam"],
            streaming=True,
            type="pil",
            label="📷 Live Camera",
            height=420,
        )
        annotated_out = gr.Image(
            type="pil",
            label="🔍 Detected & Annotated",
            height=420,
        )

    with gr.Row():
        with gr.Column(scale=2):
            description_out = gr.Textbox(
                label="🗣️ AI Description",
                lines=2,
                interactive=False,
                placeholder="Waiting for camera feed…",
            )
        with gr.Column(scale=1):
            audio_out = gr.Audio(
                label="🔊 Voice Output",
                type="filepath",
                autoplay=True,
            )

    chart_out = gr.Plot(label="📊 Model Confidence Scores")

    # ── Status / instructions row ─────────────────────────────────────────────
    gr.Markdown("""
> **How it works:** Allow camera access when prompted. Detection runs automatically every
> 1.5 seconds. Voice narration only fires when the detected item changes.
""")

    gr.Markdown(f"""
---
**Recognisable Food:** {_FOOD_CLASSES_STR}  
**Recognisable Clothing:** {_CLOTH_CLASSES_STR}
""")

    # ── Wire the streaming event ──────────────────────────────────────────────
    webcam_feed.stream(
        fn=predict_frame,
        inputs=[webcam_feed, last_desc_state],
        outputs=[annotated_out, description_out, audio_out, chart_out, last_desc_state],
        stream_every=STREAM_EVERY,
        time_limit=None,        # run indefinitely
        concurrency_limit=1,    # one inference at a time — avoids GPU contention
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue"),
    )
