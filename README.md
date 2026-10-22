---
title: Oil Palm Detector
emoji: 🌴
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# 🌴 Oil Palm Detector

Aerial-view oil palm tree detection powered by **RF-DETR Small** (fine-tuned).

## Usage
1. Upload or paste an aerial image
2. Adjust the confidence threshold if needed
3. Click **Detect Oil Palms**
4. View annotated results with bounding boxes and counts

## Model
- Architecture: RF-DETR Small (Roboflow)
- Task: Object Detection — Oil Palm Trees
- Input: Aerial / UAV imagery
