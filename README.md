# LookOut: Multi-Phase Intelligent Person Tracking System

## Overview

**LookOut** is an intelligent, multi-phase video analytics pipeline that identifies, re-identifies, and tracks a specific person in a video based on a given query image.
The system leverages a combination of **state-of-the-art deep learning models** for face recognition, body detection, feature embedding, and tracking.

The pipeline operates in **four major phases**:

### **Phase 1: Face Recognition (ArcFace + RetinaFace)**

* Extracts facial embeddings from a query image using **ArcFace**.
* Processes each frame of the input video using **RetinaFace** to detect faces.
* Compares detected faces with the query embedding.
* Saves matched faces along with their metadata (timestamp, frame number, similarity score, etc.).

### **Phase 2: Body Detection (YOLOv8 + Pose Estimation)**

* For every matched face, retrieves the corresponding video frame.
* Uses **YOLOv8-Pose** or **YOLOv8-Person** models to detect the full body of the matched individual.
* Extracts and saves body crops with detection metadata.

### **Phase 3: Re-Identification (OSNet / ResNet50)**

* Uses **OSNet** (or **ResNet50**) to extract discriminative body features.
* Creates a **Re-ID gallery** of all detected body crops.
* Stores feature vectors and metadata for efficient identity matching.

### **Phase 4: Video Tracking (ByteTrack + Multi-Verification)**

* Runs **ByteTrack** across the full video to maintain persistent identity tracking.
* Matches new detections with the Re-ID gallery and optionally applies **face verification** for higher accuracy.
* Produces a tracked video highlighting the target person and saves tracking metadata.

---

## Features

* Modular and extensible pipeline architecture.
* Multi-model verification: Face + Body + Temporal filtering.
* Re-identification gallery for efficient identity matching.
* Detailed metadata generation for every processing phase.
* Optimized for GPU execution with configurable thresholds and frame skipping.

---

## Installation and Setup

### 1. Create a Python environment

Ensure Python version **3.13.7** (or compatible) is available.

```bash
python3 -m venv venv
source venv/bin/activate
```
---

### 2. Install required dependencies

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure CUDA and PyTorch are correctly configured for GPU acceleration.
Recommended GPU: **RTX 3050 (4GB or higher)**. Minimum: **2GB VRAM**.

---

## Running the Pipeline

Open `main.py` and modify the input parameters inside the following function call:

```python
results = pipeline.run_complete_pipeline(
    query_image="/path/to/query/image.jpg",
    video_path="/path/to/input/video.mp4",
    output_dir="./output_folder",
)
```

For example:

```python
results = pipeline.run_complete_pipeline(
    query_image="./dummy_dataset/q/yt_speed_vid1.jpg",
    video_path="./dummy_dataset/vids/yt_speed_vid1.mp4",
    output_dir="./test1",
)
```

Then, run the main script:

```bash
python main.py
```

---

## Output Structure

The pipeline generates outputs organized by phase:

```
output_dir/
│
├── phase1_faces/
│   ├── matched_face_*.jpg
│   └── matches_metadata.json
│
├── phase2_bodies/
│   ├── body_*.jpg
│   └── body_detections_metadata.json
│
├── reid_gallery.json
│
├── tracked_video.mp4
└── tracked_video_tracking.json
```
---

## Hardware and Performance Notes

* Recommended: **NVIDIA GPU (RTX 3050 or higher)**
* Minimum: **2 GB GPU memory**
* CPU fallback mode is supported but significantly slower.
* Use lower `verification_threshold` and `confidence_threshold` values for lenient face matching.
