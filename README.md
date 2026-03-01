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

### Standard (Linux / Mac / Windows)

```bash
pip install -r requirements.txt
pip install opencv-python-headless
python app.py
```

### Android (Termux)

1. Install [Termux from F-Droid](https://f-droid.org/packages/com.termux/) — **not** the Play Store version
2. Open Termux and run:

```bash
pkg update && pkg upgrade
pkg install python git python-opencv
pip install flask numpy Pillow
```

3. Allow Termux to access your photos (needed to upload images):

```bash
termux-setup-storage
```

4. Clone and run:

```bash
git clone <your-repo-url>
cd Images-Test-App
python app.py
```

5. Open Chrome or Firefox on your phone and go to:

```
http://localhost:5000
```

Then open `http://localhost:5000` in your browser.

## Requirements

- Python 3.9+
- Flask
- OpenCV (via `pkg install python-opencv` on Termux, or `pip install opencv-python-headless` elsewhere)
- NumPy
- Pillow

## Usage

1. Upload **Image A** (the reference shot)
2. Upload **Image B** (the shot you want to align)
3. Click **Analyze Alignment**
4. Follow the camera movement instructions in the result panel
