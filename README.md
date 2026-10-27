# VocaVision 

**AI-Based Color, Clothing and Food Recognition Assistive System**

A vision-based assistive system that identifies **clothing colors/patterns** and **food items** and describes them through **voice output** to support independent daily activities.

> **All recognition runs locally using custom-trained neural networks. No external API or internet connection required for inference.**

---

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Object Detector | YOLOv8s (custom trained) | Detects all food & clothing items + bounding boxes |
| Food Classifier | MobileNetV2 + TF (fallback) | Identifies 10 food categories |
| Clothing Classifier | PyTorch MobileNetV2 / TF CNN (fallback) | Identifies 10 clothing categories |
| Color Detector | OpenCV + K-Means Clustering | Detects dominant clothing colors (14 color names) |
| Voice Output | pyttsx3 (offline SAPI5 / espeak) | Speaks descriptions aloud |
| Interface | Terminal CLI & Interactive Menu | **100% Terminal-based** operation (webcam, monitor, image) |

---

## Engine Priority

```
1. YOLO v3 (best) ─── vocavision_yolo.pt  ─── custom YOLOv8s, 20 classes, ~15 ms/frame
2. PyTorch v2      ─── clothing_classifier_v2.pt + food_classifier.keras
3. TF v1 (legacy)  ─── clothing_classifier.keras + food_classifier.keras
```

The system automatically selects the highest-priority available engine on startup.

---

## Project Structure

```
VocaVision/
├── main.py                       # Main terminal entry point (interactive & CLI)
├── test_image.py                 # Quick headless terminal test tool
├── vision_engine.py              # Unified inference engine (YOLO + fallback)
├── audio_output.py               # pyttsx3 Text-to-Speech helpers
├── camera_handler.py             # Headless webcam frame grabber & stream generator
│
├── utils/
│   ├── color_detector.py         # K-Means dominant color detection (14 colors)
│   └── annotator.py              # GrabCut segmentation + bounding-box drawing
│
├── training/
│   ├── train_food.py             # Train food classifier (MobileNetV2)
│   ├── train_clothing.py         # Train clothing classifier v1 (CNN)
│   ├── train_clothing_v2.py      # Train clothing classifier v2 (PyTorch GPU)
│   ├── train_yolo.py             # Fine-tune YOLOv8s on food + clothing dataset
│   ├── prepare_yolo_dataset.py   # Build YOLO-format dataset from raw images
│   ├── yolo_classes.py           # Shared class list (20 classes)
│   └── evaluate.py               # Evaluate models & plot confusion matrices
│
├── models/                       # Trained model files
│   ├── vocavision_yolo.pt        # ← primary YOLO v3 model (active)
│   ├── clothing_classifier_v2.pt # ← PyTorch fallback
│   ├── clothing_classifier.keras # ← TF legacy fallback
│   └── food_classifier.keras     # ← TF food classifier
│
├── dataset/                      # YOLO-format dataset (train / val splits)
├── legacy_gui/                   # Archived GUI / Web code (Gradio & matplotlib viewers)
├── requirements.txt
└── .env.example
```

---

## Terminal Usage

### 1. Interactive Terminal Mode
Run without arguments to launch the interactive terminal menu:
```bash
python main.py
```
You can choose to snap & analyze a webcam frame, run continuous background camera monitoring, analyze image files, or toggle voice output.

### 2. Direct CLI Commands

**Live video stream with real-time bounding boxes & color labels:**
```bash
python main.py --live-view
```

**Stream from mobile phone camera over Wi-Fi (IP Webcam):**
```bash
python main.py --live-view --camera http://192.168.1.X:8080/video
```

**Analyze a single image file:**
```bash
python main.py --image path/to/image.jpg
```

**Snap a frame from webcam and analyze immediately:**
```bash
python main.py --webcam
```

**Run continuous terminal monitoring (alerts via voice when scene changes):**
```bash
python main.py --monitor --interval 2.0
```

**Save annotated bounding-box image to file:**
```bash
python main.py --image path/to/image.jpg --save output.jpg
```

**Mute voice output (text output only):**
```bash
python main.py --image path/to/image.jpg --no-tts
```

---

## Model Details

| Model | Dataset | Architecture | Accuracy / Metrics |
|-------|---------|-------------|--------------------|
| YOLO Detector | Custom (food + clothing) | YOLOv8s fine-tuned | ~65.9% mAP50, 72.4% precision |
| Clothing v2 | Fashion-MNIST (10 classes) | MobileNetV2 (PyTorch, TTA) | ~88–92% |
| Food Classifier | Food-101 (10 classes) | MobileNetV2 + fine-tuning | ~75–85% |
| Clothing v1 | Fashion-MNIST (10 classes) | 3-block CNN (TF) | ~88–90% |

---

## Color Detection

The K-Means color detector supports **14 color names**:

| Chromatic | Achromatic / Earth |
|-----------|--------------------|
| Red, Orange, Yellow, Green | White, Black, Gray |
| Blue, Navy, Purple, Pink | Brown, Beige, Neutral |

Background clusters are automatically filtered so the returned color reflects the actual item.

---

## Recognized Categories (20 Classes)

**Food (10):**
Apple Pie • Fried Rice • Hamburger • Hot Dog • Ice Cream • Pizza • Ramen • Samosa • Sushi • Waffles

**Clothing (10):**
T-shirt/top • Trouser • Pullover • Dress • Coat • Sandal • Shirt • Sneaker • Bag • Ankle Boot
