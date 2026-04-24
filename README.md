# 🚁 Real-Time UAV Object Detection System

<div align="center">

![UAV Detection](https://img.shields.io/badge/Model-YOLOv8m-blue?style=for-the-badge&logo=python)
![Dataset](https://img.shields.io/badge/Dataset-VisDrone--2019-green?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge&logo=pytorch)
![App](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Deploy](https://img.shields.io/badge/Deploy-HuggingFace-yellow?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**A deep learning-based real-time object detection system for UAV/drone aerial imagery using YOLOv8**

[🌐 Live Demo](https://huggingface.co/spaces/Suraj1229/uav-object-detection) · [📊 Dataset](https://github.com/VisDrone/VisDrone-Dataset) · [📄 Report Bug](https://github.com/Suraj1229/uav-object-detection/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Model](#-model)
- [Training](#-training)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

The **Real-Time UAV Object Detection System** is an end-to-end deep learning project that detects and classifies objects from drone/UAV aerial imagery in real-time. Built using **YOLOv8m** trained on the **VisDrone 2019** dataset, the system can identify 10 different object categories including pedestrians, vehicles, and more.

The project includes a full-stack web application deployed on **Hugging Face Spaces** with a military-style tactical interface, supporting image detection, video analysis, and live camera detection.

---

## 🌐 Demo

> **Live App:** [https://huggingface.co/spaces/Suraj1229/uav-object-detection](https://huggingface.co/spaces/Suraj1229/uav-object-detection)

| Image Detection | Video Analysis | Live Camera |
|:-:|:-:|:-:|
| Upload aerial image | Frame-by-frame video | Real-time camera feed |
| Instant bounding boxes | Live streaming detection | Auto-refresh capture |
| Stats & location map | Per-frame analytics | Session tracking |

---

## ✨ Features

### 🔍 Detection Capabilities
- **10 Object Classes** — Pedestrian, People, Bicycle, Car, Van, Truck, Tricycle, Awning-Tricycle, Bus, Motor
- **Adjustable Confidence** — Threshold slider (0.10 – 0.90)
- **IoU Control** — Non-maximum suppression tuning
- **Small Object Detection** — Optimized for aerial imagery

### 📸 Image Detection Tab
- Upload JPG/PNG aerial images
- Real-time YOLOv8 inference
- Bounding boxes with class labels and confidence
- Detection statistics with bar charts
- Confidence score distribution histogram
- Tactical object location map
- Downloadable CSV detection log

### 🎥 Video Detection Tab
- Upload MP4/AVI/MOV video files
- Frame-by-frame live streaming detection
- Real-time target count per frame chart
- Mission summary with peak/average statistics

### 📡 Live Camera Tab
- **Single Photo Mode** — Capture & detect instantly
- **Auto Live Mode** — Camera auto-refreshes every 1-5 seconds
- Live target history chart
- Scan history grid (last 6 frames)
- Session-wide statistics tracking
- Reset session button

### 🎨 UI/UX
- Military tactical dark theme
- Animated grid background
- Scanning line animation
- Live radar widget
- Orbitron / Share Tech Mono fonts
- Neon cyan/green color scheme
- Corner HUD brackets

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│              STREAMLIT WEB APP              │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Image   │ │  Video   │ │   Live     │  │
│  │Detection │ │Detection │ │  Camera    │  │
│  └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
└───────┼─────────────┼─────────────┼─────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────┐
│              DETECT MODULE                  │
│  UAVDetector → YOLO(best.pt) → Results      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              UTILS MODULE                   │
│  Stats → Charts → Map → Table → CSV         │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│           YOLOv8m MODEL (best.pt)           │
│  Input: 640×640 | Classes: 10 | Epochs: 100 │
└─────────────────────────────────────────────┘
```

---

## 📦 Dataset

### VisDrone 2019 — DET Task

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 6,471  | ~390,000    |
| Val   | 548    | ~29,000     |
| Test  | 1,610  | —           |

### Object Classes

| ID | Class | Description |
|----|-------|-------------|
| 0  | pedestrian | Single walking person |
| 1  | people | Group of people |
| 2  | bicycle | Bicycle with/without rider |
| 3  | car | Passenger car |
| 4  | van | Van/minivan |
| 5  | truck | Truck/lorry |
| 6  | tricycle | Three-wheeled vehicle |
| 7  | awning-tricycle | Covered tricycle |
| 8  | bus | Bus/coach |
| 9  | motor | Motorcycle/scooter |

### Data Preprocessing
- Converted VisDrone annotation format → YOLO format
- Normalized bounding box coordinates
- Filtered invalid boxes (w≤0 or h≤0)
- Removed ignore region class (category 0)
- Removed out-of-range class IDs (>9)

---

## 🤖 Model

### YOLOv8m Architecture

```
Input (640×640×3)
    │
    ▼
Backbone (CSPDarknet)
    │ Feature maps at 3 scales
    ▼
Neck (PANet + FPN)
    │ Multi-scale feature fusion
    ▼
Head (Decoupled Detection)
    │ Per-scale predictions
    ▼
Output: [class, confidence, bbox] × 10 classes
```

| Parameter | Value |
|-----------|-------|
| Model variant | YOLOv8m |
| Parameters | 25.9M |
| GFLOPs | 78.9 |
| Input size | 640×640 |
| Pretrained | COCO (ImageNet backbone) |

---

## 🔥 Training

### Configuration

```yaml
Model     : yolov8m.pt (pretrained COCO)
Dataset   : VisDrone 2019 DET
Epochs    : 100
Batch     : 16
Image size: 640
Optimizer : AdamW
lr0       : 0.001
lrf       : 0.01
Momentum  : 0.937
Weight dec: 0.0005
AMP       : True
```

### Data Augmentation

| Augmentation | Value |
|-------------|-------|
| Mosaic | 1.0 |
| MixUp | 0.2 |
| Copy-Paste | 0.3 |
| Rotation | ±15° |
| Flip LR | 0.5 |
| Flip UD | 0.5 |
| Scale | 0.9 |
| HSV-H | 0.015 |
| HSV-S | 0.7 |
| HSV-V | 0.4 |

### Training Environment

| Platform | Spec |
|----------|------|
| Google Colab | Tesla T4 (15GB) |
| Kaggle | Tesla T4 (15GB) |
| Python | 3.12 |
| PyTorch | 2.x |
| Ultralytics | 8.4.x |

---

## 📊 Results

### Validation Metrics

| Metric | Score |
|--------|-------|
| mAP@50 | 0.389 |
| mAP@50-95 | 0.223 |
| Precision | 0.504 |
| Recall | 0.403 |

### Per-Class Performance

| Class | AP@50 |
|-------|-------|
| car | 0.783 |
| bus | 0.560 |
| van | 0.447 |
| motor | 0.440 |
| pedestrian | 0.425 |
| truck | 0.384 |
| tricycle | 0.284 |
| people | 0.304 |
| bicycle | 0.123 |
| awning-tricycle | 0.141 |

> **Note:** Low mAP is expected for aerial UAV datasets due to extremely small object sizes. VisDrone is one of the most challenging object detection benchmarks.

---

## 💻 Installation

### Prerequisites
```bash
Python >= 3.10
CUDA >= 11.8 (for GPU)
```

### Clone Repository
```bash
git clone https://github.com/Suraj1229/uav-object-detection.git
cd uav-object-detection
```

### Create Virtual Environment
```bash
# Windows
python -m venv uav_env
uav_env\Scripts\activate

# Linux/Mac
python -m venv uav_env
source uav_env/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```txt
streamlit
ultralytics
opencv-python-headless
Pillow
plotly
pandas
numpy
torch
torchvision
streamlit-autorefresh
```

---

## 🚀 Usage

### Run Web App Locally
```bash
streamlit run app.py
```

### Run Detection on Image (Python)
```python
from detect import UAVDetector
from PIL import Image
import numpy as np

detector = UAVDetector("models/best.pt")
image    = np.array(Image.open("test.jpg").convert("RGB"))

annotated, detections = detector.detect_image(
    image, conf=0.35, iou=0.45
)

print(f"Detected {len(detections)} objects")
for d in detections:
    print(f"  {d['class']}: {d['confidence']:.2f}")
```

### Run Detection on Video
```python
from detect import UAVDetector
import cv2

detector = UAVDetector("models/best.pt")
cap      = cv2.VideoCapture("drone_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated, count = detector.detect_video_frame(frame_rgb)
    cv2.imshow("UAV Detection", annotated[:,:,::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Train Your Own Model
```python
from ultralytics import YOLO

model   = YOLO('yolov8m.pt')
results = model.train(
    data    = 'UAV_Dataset/data.yaml',
    epochs  = 100,
    imgsz   = 640,
    batch   = 16,
    device  = 0
)
```

---

## 📁 Project Structure

```
UAV_Object_Detection/
│
├── 📄 app.py                   ← Main Streamlit web app
├── 📄 detect.py                ← UAVDetector class
├── 📄 utils.py                 ← Charts, stats, tables
├── 📄 requirements.txt         ← Python dependencies
├── 📄 packages.txt             ← System dependencies
├── 📄 README.md                ← This file
│
├── 📂 models/
│   └── best.pt                 ← Trained YOLOv8m weights
│
├── 📂 UAV_Dataset/
│   ├── images/
│   │   ├── train/              ← 6,471 training images
│   │   ├── val/                ← 548 validation images
│   │   └── test/               ← 1,610 test images
│   ├── labels/
│   │   ├── train/              ← YOLO format annotations
│   │   ├── val/
│   │   └── test/
│   └── data.yaml               ← Dataset config
│
└── 📂 scripts/
    ├── organize_dataset.py     ← Extract zip files
    ├── convert_visdrone_to_yolo.py  ← Format conversion
    ├── create_yaml.py          ← Generate data.yaml
    ├── verify.py               ← Dataset verification
    ├── check_gpu.py            ← System check
    ├── train.py                ← Training script
    ├── resume_train.py         ← Resume training
    └── evaluate.py             ← Model evaluation
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **YOLOv8** (Ultralytics) | Object detection model |
| **PyTorch** | Deep learning framework |
| **OpenCV** | Image/video processing |
| **Streamlit** | Web application framework |
| **Plotly** | Interactive charts |
| **Pandas** | Data manipulation |
| **NumPy** | Array operations |
| **Pillow** | Image handling |
| **Google Colab** | Model training (GPU) |
| **Kaggle Notebooks** | Model training (GPU) |
| **Hugging Face Spaces** | App deployment |

---

## 🔮 Future Improvements

- [ ] **SAHI Integration** — Sliced inference for better small object detection
- [ ] **Object Tracking** — ByteTrack/DeepSORT for multi-object tracking
- [ ] **Higher Resolution** — Train with imgsz=1280 for more detail
- [ ] **More Epochs** — Continue training to 200+ epochs
- [ ] **YOLOv8x** — Upgrade to larger model for better accuracy
- [ ] **Heatmap** — Object density heatmap overlay
- [ ] **Alert System** — Notification when crowd/vehicle count exceeds threshold
- [ ] **GPS Overlay** — Map object detections to real GPS coordinates
- [ ] **Edge Deployment** — Export to TensorRT for Jetson Nano
- [ ] **Custom Dataset** — Train on domain-specific drone footage

---

## 🙏 Acknowledgements

- **VisDrone Team** — Tianjin University for the VisDrone 2019 dataset
- **Ultralytics** — For the excellent YOLOv8 framework
- **Streamlit** — For the easy web app deployment
- **Hugging Face** — For free GPU-powered Spaces hosting
- **Google Colab & Kaggle** — For free GPU training resources

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Suraj Patil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 👤 Author

**Suraj Patil**

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Suraj1229-yellow?style=flat&logo=huggingface)](https://huggingface.co/Suraj1229)
[![Kaggle](https://img.shields.io/badge/Kaggle-surajpatil43-blue?style=flat&logo=kaggle)](https://kaggle.com/surajpatil43)

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star!**

🚁 *Real-Time UAV Object Detection System — Powered by YOLOv8 + VisDrone 2019*

</div>
