# Image Alignment Tool

A web app that compares two similar images, detects how far apart the camera positions are, and provides step-by-step instructions for aligning them.

## Features

- Upload two images via drag-and-drop or file picker
- Keypoint detection using ORB (with AKAZE fallback)
- Homography-based computation of translation, rotation, and scale
- Composite visualization: matched keypoints, overlay/diff heatmap, instruction panel
- Downloadable result image

## How It Works

1. **Detect** — ORB finds up to 2000 feature points in each image
2. **Match** — BFMatcher with Lowe's ratio test filters good correspondences
3. **Compute** — RANSAC homography decomposes into dx/dy shift, rotation angle, and scale factor
4. **Guide** — plain-English instructions tell you how to move the camera to align Image B with Image A

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

## Requirements

- Python 3.9+
- Flask
- OpenCV (`opencv-python-headless`)
- NumPy
- Pillow

## Usage

1. Upload **Image A** (the reference shot)
2. Upload **Image B** (the shot you want to align)
3. Click **Analyze Alignment**
4. Follow the camera movement instructions in the result panel
