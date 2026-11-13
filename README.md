# IPCV - Object Detection and Icon Grid Extraction

Classical Computer Vision project implementing unsupervised 2D object detection, binary mask extraction, and UI icon grid localization using **OpenCV** and **scikit-image** (without deep learning).

---

## 🛠️ Environment Setup

A virtual environment is used to isolate dependencies.

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

*(If script execution is restricted in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.)*

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

---

## 📁 Project & Folder Structure

```text
IPCV/
├── .venv/                      # Python virtual environment (ignored by git)
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation and usage guide
│
├── icon_grid_detector.py       # Canny & contour-based grid detector
├── object_detector_simple.py   # Connected component mask detector
├── object_detector_simple_rgb.py # Otsu & morphology RGB detector
│
├── dataset/                    # All input images and masks
│   ├── synthetic/              # 40 synthetic images & matching binary masks
│   └── internet/               # Real-world icon grids & clipart images
│
└── output_images/              # All generated detection results
    ├── simple/                 # Output from object_detector_simple.py
    ├── rgb/                    # Output from object_detector_simple_rgb.py
    └── grid/                   # Output from icon_grid_detector.py
```

---

## 🚀 Usage

### 1. Detect Objects from Binary Masks (`object_detector_simple.py`)
Uses Connected Component Labeling (8-connectivity) to extract and bound objects from a ground-truth or pre-segmented binary mask.

```powershell
python object_detector_simple.py --mask dataset/synthetic/synthetic_001_mask.png --out output_images/simple/test.png
```
*Options:*
* `--mask, -m`: Path to binary mask (required).
* `--image, -i`: Optional path to original RGB image to overlay bounding boxes onto.
* `--out, -o`: Path to save annotated image (default: `output_images/simple/detected_mask.png`).
* `--min-area`: Minimum connected component area in pixels (default: `20`).

---

### 2. Detect Objects from Raw RGB Images (`object_detector_simple_rgb.py`)
End-to-end unsupervised detector using Gaussian blur, Otsu's thresholding, and morphological opening/closing.

```powershell
# On synthetic geometric dataset
python object_detector_simple_rgb.py --input dataset/synthetic/synthetic_039.png --out output_images/rgb/result_039.png

# On real-world multi-object images
python object_detector_simple_rgb.py --input dataset/internet/random2.jpeg --out output_images/rgb/random2_detected.png
```
*Options:*
* `--input, -i`: Path to input RGB image (required).
* `--out, -o`: Path to save annotated image (default: `output_images/rgb/detected_rgb.png`).
* `--min-area`: Minimum component area in pixels (default: `150`).
* `--invert` / `--no-invert`: Manually force light or dark background thresholding (auto-detected by default).

---

### 3. UI Icon / Grid Element Detector (`icon_grid_detector.py`)
Specialized detector for structured icon grids, app launchers, or icon sheets. Uses Canny edge detection, morphological dilation to bridge gaps, external contour extraction, and reading-order (row-by-row) grid sorting.

```powershell
python icon_grid_detector.py --input dataset/internet/random6.png --out output_images/grid/icons_out6.png
python icon_grid_detector.py --input dataset/internet/random.jpg --out output_images/grid/icons_random.png
```
*Options:*
* `--input, -i`: Path to input image (required).
* `--out, -o`: Path to save annotated image (default: `output_images/grid/detected_icons.png`).
* `--min-area`: Minimum contour area threshold (default: `500`).
* `--canny-low`: Canny lower threshold (default: `50`).
* `--canny-high`: Canny upper threshold (default: `150`).
