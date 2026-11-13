# Image Processing & Computer Vision Portfolio — Sourish Musib

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/sourish-24/image-processing)

A classical image processing and computer vision toolkit built with **OpenCV**, **NumPy**, **SciPy**, and **Streamlit**. Demonstrates mathematical foundations of spatial and frequency filtering, feature matching, multi-view geometry, temporal video analytics, and hardware **Image Signal Processor (ISP)** pipeline design.

---

## 🌟 Key Highlights & Engineering Depth

* **Purely Classical (No Deep Learning):** Focuses on core deterministic signal processing, mathematical algorithms, and hardware-conscious edge compute patterns.
* **Hardware Camera ISP Pipeline Simulation (Module 12):** End-to-end camera pipeline: raw Bayer CFA mosaic $\to$ Demosaicing $\to$ Black Level Calibration (BLC) $\to$ Auto White Balance (AWB) $\to$ Color Correction Matrix (CCM) $\to$ Tone Mapping Curve $\to$ HDR Exposure Fusion.
* **Temporal Video Analytics (Module 11):** Farnebäck dense optical flow vector fields, MOG2/KNN background subtraction, and cumulative activity heatmaps.
* **Frequency Domain & DSP (Module 6):** 2D Discrete Fourier Transform (DFT), Ideal/Gaussian/Butterworth low/high pass filtering, periodic noise notch rejection, and hybrid image synthesis.
* **Quantitative Image Quality Assessment (Module 7):** Direct measurement of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) across sensor noise models.
* **Interactive Web Application (`app.py`):** Multi-module Streamlit interface with preset sample images, live parameter sliders, side-by-side visualization, and concise algorithmic overviews.

---

## 📐 Progressive 4-Tier Architecture

| Tier | Module | Topic | Core Algorithms & Formulas |
| :--- | :--- | :--- | :--- |
| **Tier 1: Foundations** | **M01** | Point Operations & Histograms | $I_{\text{out}} = c \cdot I + b$, Gamma $\gamma$ correction, CLAHE adaptive equalization |
| | **M02** | Color Space Exploration | RGB $\leftrightarrow$ HSV $\leftrightarrow$ YCrCb $\leftrightarrow$ LAB, Gray-World Auto White Balance |
| | **M03** | Spatial Filtering & Convolution | Gaussian blur, Median, Edge-preserving Bilateral Filter, Unsharp masking |
| **Tier 2: Intermediate** | **M04** | Morphological Operations | Erosion $\ominus$, Dilation $\oplus$, Opening, Closing, Top-Hat, Skeletonization, Watershed |
| | **M05** | Edge Detection | First-order Sobel/Scharr, Second-order Laplacian, LoG, 4-stage Canny Edge Detector |
| | **M06** | 2D Fourier Transform (FFT) | 2D DFT $\mathcal{F}\{f\}$, Gaussian/Butterworth filters, Notch rejection, Hybrid images |
| | **M07** | Noise & Image Restoration | Gaussian, Salt & Pepper, Speckle noise; Non-Local Means (NLM), PSNR/SSIM evaluation |
| **Tier 3: Advanced** | **M08** | Segmentation & Contours | Adaptive Mean/Gaussian, Multi-Otsu, GrabCut GMM segmentation, Contour geometry |
| | **M09** | Features & Keypoint Matching | Harris corner response $R$, ORB binary descriptors, SIFT, Lowe's ratio test matching |
| | **M10** | Geometric Transforms | Affine transformation, 4-point perspective document scanner, Panorama stitching |
| **Tier 4: Production Video & ISP** | **M11** | Video Analytics & Motion | Frame differencing, MOG2 background subtraction, Farnebäck Dense Optical Flow |
| | **M12** | Camera ISP Simulation | Bayer RGGB $\to$ Debayer $\to$ BLC $\to$ AWB $\to$ $3\times3$ CCM $\to$ sRGB Tone Curve $\to$ Mertens HDR |

---

## 🛠️ Installation & Setup

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Interactive Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

* **Sample images pre-loaded:** Comes pre-loaded with sample images (`sourish.jpg`, `cameraman.png`, `coins.png`, `building.png`, `noisy_sample.png`, `raw_bayer.png`, `panorama_left.png`, `panorama_right.png`).
* **Custom testing:** Toggle the sidebar to "Upload Custom Image" to test any custom photograph.
* **Download outputs:** Every module includes a direct "Download Result" button.

---

## 🧪 Running the Automated Test Suite

All algorithms and pipelines are covered by automated unit and integration tests:

```powershell
pytest tests/ -v
```

```text
============================= 51 passed in 4.62s ==============================
```

---

## 📁 Repository Structure

```text
IPCV/
├── .venv/                      # Python virtual environment (git ignored)
├── dataset/                    # Image datasets
│   ├── samples/                # Curated benchmark images & sample generator
│   ├── synthetic/              # 40 synthetic geometric shapes & binary masks
│   └── internet/               # Real-world icon grids & clipart images
│
├── modules/                    # Core algorithm modules (12 topics)
│   ├── m01_point_ops.py        # Module 1: Point Operations & Histograms
│   ├── m02_color_spaces.py     # Module 2: Color Spaces & AWB
│   ├── m03_spatial_filters.py  # Module 3: Spatial Filtering & Convolutions
│   ├── m04_morphology.py       # Module 4: Morphology & Watershed
│   ├── m05_edge_detection.py   # Module 5: Sobel, Canny, LoG
│   ├── m06_frequency_domain.py # Module 6: 2D FFT & Frequency Filtering
│   ├── m07_noise_restoration.py# Module 7: Noise Models & PSNR/SSIM
│   ├── m08_thresholding_seg.py # Module 8: Adaptive, Multi-Otsu, GrabCut
│   ├── m09_features.py         # Module 9: Harris, ORB, SIFT Keypoints
│   ├── m10_transforms.py       # Module 10: Perspective & Panorama Stitching
│   ├── m11_video.py            # Module 11: Optical Flow & Background Subtraction
│   └── m12_isp_pipeline.py     # Module 12: Camera ISP Pipeline Simulation
│
├── tests/                      # Pytest test suite (51 test cases)
│   ├── test_samples.py
│   ├── test_tier1_foundations.py
│   ├── test_tier2_morph_edges.py
│   ├── test_tier2_fourier_noise.py
│   ├── test_tier3_advanced.py
│   ├── test_tier4_video_isp.py
│   └── test_app_integration.py
│
├── output_images/              # CLI detector output directories
│   ├── simple/
│   ├── rgb/
│   └── grid/
│
├── app.py                      # Interactive Streamlit Web Application
├── icon_grid_detector.py       # Standalone CLI icon grid detector
├── object_detector_simple.py   # Standalone CLI mask detector
├── object_detector_simple_rgb.py # Standalone CLI RGB detector
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
