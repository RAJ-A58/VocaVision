# VocaVision — Project Tasks & Progress Tracker

> **Vision-based Assistive System for Food, Clothing & Color Recognition with Offline Speech Output**

---

## 📌 Executive Summary
VocaVision is an AI-powered assistive system built to empower visually impaired individuals in daily activities. It identifies food items, recognizes clothing articles and their dominant colors/patterns, and announces natural-language descriptions through offline Text-to-Speech (TTS).

All neural networks and pipelines run **100% locally** without external API dependencies or active internet connection requirements.

---

## ✅ Completed Milestones (Till Now)

### 1. Neural Network Architecture & Model Zoo
- [x] **Primary Engine — YOLO v3 (`models/vocavision_yolo.pt`)**:
  - Custom-trained **YOLOv8s** fine-tuned on a merged 20-class dataset (10 food + 10 clothing categories).
  - Trained over 60 epochs, achieving **~65.9% mAP50** and **72.4% precision**.
  - Performs object localization and multi-object classification in a single GPU pass (~15 ms latency).
- [x] **Secondary Fallback — PyTorch MobileNetV2 v2 (`models/clothing_classifier_v2.pt`)**:
  - PyTorch transfer learning model for clothing classification with Test-Time Augmentation (TTA), achieving **~88–92% accuracy**.
- [x] **Food Classifier — MobileNetV2 (`models/food_classifier.keras`)**:
  - Fine-tuned on Food-101 (10 common daily food classes: Apple Pie, Fried Rice, Hamburger, Hot Dog, Ice Cream, Pizza, Ramen, Samosa, Sushi, Waffles).
- [x] **Legacy Fallback — TF CNN (`models/clothing_classifier.keras`)**:
  - 3-block convolutional neural network for baseline fallback.
- [x] **Dynamic Priority Loading (`vision_engine.py`)**:
  - Automatic fallback cascade: `YOLO v3` -> `PyTorch v2 + TF Food` -> `TF CNN Legacy`.
  - Automatic hardware acceleration routing (CUDA GPU / CPU).

### 2. Color & Pattern Analysis Engine
- [x] **K-Means Dominant Color Clustering (`utils/color_detector.py`)**:
  - Extracts dominant colors directly from detected bounding box crops.
  - Recognizes **14 distinct color profiles** (Chromatic: Red, Orange, Yellow, Green, Blue, Navy, Purple, Pink; Achromatic/Earth: White, Black, Gray, Brown, Beige, Neutral).
  - Background/studio backdrop filtering (eliminates near-white lighting glare and near-black shadow clusters).
- [x] **Background Segmentation (`rembg`)**:
  - Integrated u2netp lightweight background removal for precise garment boundary isolation.

### 3. Audio & Speech Synthesis
- [x] **Offline Text-to-Speech (`audio_output.py`)**:
  - Offline SAPI5 / espeak speech output via `pyttsx3`.
  - Threaded non-blocking asynchronous speech worker (`speak_async`), preventing frame drops or terminal lockups during speech.

### 4. Headless & Pure Terminal-Based Interface (Latest Update)
- [x] **Frontend Stripped**:
  - Removed reliance on heavy GUI windows (`cv2.imshow`, `matplotlib.pyplot`, `Gradio`).
  - Archived legacy Gradio and matplotlib tools safely into `legacy_gui/` (`app.py`, `demo_bbox.py`).
- [x] **Headless Camera Handler (`camera_handler.py`)**:
  - Added single-frame capture with hardware auto-exposure & white-balance warmup.
- [x] **Interactive & CLI Terminal Entry Point (`main.py`)**:
  - Interactive terminal menu (Webcam snap, Continuous monitor, Image analysis, Voice toggle).
  - Command-line flags for scripting (`--image`, `--webcam`, `--monitor`, `--save`, `--no-tts`).
- [x] **Headless Model Testing Script (`test_image.py`)**:
  - Updated to unpack all 5 engine objects and run direct headless terminal diagnostics.

### 5. Documentation & Presentation Assets
- [x] Updated repository [`README.md`](README.md) with terminal commands and architecture table.
- [x] Created [`VocaVision_Presentation.pptx`](VocaVision_Presentation.pptx) and [`VocaVision_Tech_Stack.pdf`](VocaVision_Tech_Stack.pdf).

---

## 📋 Open Tasks & Roadmap for Teammates

Here is the assigned task backlog for next development iterations:

| Priority | Task ID | Description | Target Area | Assignee / Status |
|:---:|:---:|---|---|:---:|
| 🔴 **P1** | **TASK-01** | **Lighting & Environment Robustness**: Implement automatic CLAHE / white-balancing on webcam frames before passing to the color detector to prevent warm-light color drift. | `utils/color_detector.py` | Open |
| 🔴 **P1** | **TASK-02** | **Audio Queue Throttling in Monitor Mode**: Ensure rapid changes in camera feed do not overlap overlapping speech threads (add a FIFO speech queue or cancellation token). | `audio_output.py` | Open |
| 🟡 **P2** | **TASK-03** | **Model Quantization (ONNX / OpenVINO / INT8)**: Export `vocavision_yolo.pt` to ONNX and INT8 to optimize CPU-only inference for low-power laptops and Raspberry Pi. | `models/`, `training/` | Open |
| 🟡 **P2** | **TASK-04** | **Spatial & Directional Heuristics**: Use bounding box `(x1, y1, x2, y2)` coordinates to announce object locations (e.g., *"On your left, a blue shirt; in the center, a plate of pizza"*). | `vision_engine.py` | Open |
| 🟡 **P2** | **TASK-05** | **Automated Unit & Regression Tests**: Create a `tests/` directory with `pytest` scripts testing `vision_engine.py` against dummy matrices and benchmark images. | `tests/` | Open |
| 🟢 **P3** | **TASK-06** | **Multi-Language Speech & Accents**: Add support for language selection (Spanish, Hindi, French, etc.) and speech speed presets via CLI args. | `audio_output.py`, `main.py` | Open |
| 🟢 **P3** | **TASK-07** | **Raspberry Pi / Edge Device Integration**: Test camera capture loop with CSI / USB cameras on Linux/ARM environments and document setup. | `camera_handler.py` | Open |

---

## 🚀 Quickstart for Teammates

### 1. Setup Environment
```bash
git pull origin main
pip install -r requirements.txt
```

### 2. Launch Interactive Terminal
```bash
python main.py
```

### 3. Run Direct Headless Commands
```bash
# Test an image file
python main.py --image path/to/sample.jpg

# Capture from webcam in terminal
python main.py --webcam

# Continuous camera monitor with voice alerts
python main.py --monitor --interval 2.0

# Silent mode (no audio)
python main.py --image path/to/sample.jpg --no-tts
```

---

## 👥 Contribution Guidelines
1. Create a feature branch for your task: `git checkout -b feature/TASK-ID-description`.
2. Keep the codebase **100% headless / terminal-compatible** (do not add `cv2.imshow` or blocking GUI loops to core code).
3. Update this file (`task.md`) when you complete or pick up a task.
4. Verify your changes with `python test_image.py --image <image_path>` before pushing.
