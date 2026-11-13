# IPCV Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-grade classical image processing & computer vision toolkit and interactive Streamlit web application spanning 12 progressive modules from point operations to camera ISP simulation.

**Architecture:** A modular Python architecture where each domain topic lives in its own dedicated, testable module under `modules/` with clean NumPy/OpenCV/SciPy implementations. A central `dataset/samples/` directory holds curated test images. A multi-page Streamlit application (`app.py`) dynamically provides interactive sliders, side-by-side visualization, and theory explanations for all 12 modules.

**Tech Stack:** Python 3.14, OpenCV (`opencv-python`), scikit-image, NumPy, SciPy, Matplotlib, Streamlit, Pytest.

**Spec:** [implementation-plan.md](file:///c:/Users/musib/.gemini/antigravity/brain/eff62e78-8f8e-4ab6-ae33-1ba96dfc083d/implementation-plan.md)

## Global Constraints
- Purely classical computer vision algorithms only (no deep learning or PyTorch/TensorFlow models).
- Every module function must accept standard NumPy ndarrays (uint8 or float32/float64) and return valid images or structured metrics.
- Default sample images must be available offline within `dataset/samples/` so any demo visitor can test immediately without uploading.
- Non-destructive processing: original images must not be mutated in-place.
- Clean error handling: invalid paths, mismatched dimensions, and out-of-range parameters must raise descriptive exceptions or handle gracefully.

## Review Focus
1. **Color channel format consistency:** OpenCV uses BGR by default while Streamlit/Matplotlib/Pillow expect RGB. All module functions should standardise on RGB or document BGR conversions clearly.
2. **Data type and dynamic range overflow:** Point operations, convolutions, and Fourier transforms frequently overflow `uint8` or produce negative values. Clamping (`np.clip(..., 0, 255).astype(np.uint8)`) or normalization must be strictly applied.
3. **Empty or degenerate contours/boxes:** Contours or connected components that find 0 objects must return empty lists rather than throwing IndexError or ZeroDivisionError.
4. **Single-channel vs Multi-channel input handling:** Grayscale operations (Fourier, Canny, Otsu) must cleanly handle incoming 3-channel RGB images by auto-converting or validating.
5. **Video processing without video hardware:** Video processing module must support synthetic frame sequences or fallback test videos when webcam or hardware video decoders are absent.

---

### Task 1: Environment Setup, Dependencies & Sample Dataset Generation

**Files:**
- Modify: `requirements.txt`
- Create: `dataset/samples/generate_samples.py`
- Test: `tests/test_samples.py`

**Interfaces:**
- Produces: Curated benchmark images in `dataset/samples/`: `cameraman.png`, `coins.png`, `building.png`, `noisy_sample.png`, `text_doc.png`, `raw_bayer.png`, `panorama_left.png`, `panorama_right.png`.

- [ ] **Step 1: Update `requirements.txt` with UI and math dependencies**
Ensure `requirements.txt` includes:
```text
opencv-python>=4.8.0
scikit-image>=0.21.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
Pillow>=10.0.0
streamlit>=1.30.0
pytest>=7.0.0
```

- [ ] **Step 2: Install dependencies into `.venv`**
Run: `.\.venv\Scripts\pip install -r requirements.txt`
Expected: Successful package installations.

- [ ] **Step 3: Write test for sample images generation**
Create `tests/test_samples.py`:
```python
import os
import cv2
import pytest

EXPECTED_SAMPLES = [
    "cameraman.png",
    "coins.png",
    "building.png",
    "noisy_sample.png",
    "text_doc.png",
    "raw_bayer.png",
    "panorama_left.png",
    "panorama_right.png",
]

def test_samples_exist_and_readable():
    sample_dir = os.path.join("dataset", "samples")
    assert os.path.isdir(sample_dir), "Sample directory must exist"
    for name in EXPECTED_SAMPLES:
        p = os.path.join(sample_dir, name)
        assert os.path.isfile(p), f"Missing sample {name}"
        img = cv2.imread(p)
        assert img is not None, f"Failed to load image {name}"
        assert img.shape[0] > 50 and img.shape[1] > 50
```

- [ ] **Step 4: Run test to verify it fails**
Run: `.\.venv\Scripts\pytest tests/test_samples.py`
Expected: FAIL (files missing).

- [ ] **Step 5: Implement `dataset/samples/generate_samples.py`**
Generate high-quality synthetic standard test patterns programmatically using OpenCV/NumPy:
- `cameraman.png`: Geometric synthetic figure with coats and gradients.
- `coins.png`: Multiple circular overlapping & non-overlapping disks for morphology and watershed.
- `building.png`: Architectural pattern with strong corners, brick edges, and windows for Harris/SIFT.
- `noisy_sample.png`: Clean image injected with Gaussian + Salt & Pepper noise.
- `text_doc.png`: High-contrast slanted/shadowed text document for thresholding & perspective warp.
- `raw_bayer.png`: 2D single-channel RGGB pattern simulation.
- `panorama_left.png` & `panorama_right.png`: Overlapping scene with affine shift for stitching.
Run the script to populate `dataset/samples/`.

- [ ] **Step 6: Run test to verify it passes**
Run: `.\.venv\Scripts\pytest tests/test_samples.py`
Expected: PASS.

---

### Task 2: Tier 1 Foundations (Modules 1, 2, 3)

**Files:**
- Create: `modules/__init__.py`
- Create: `modules/m01_point_ops.py`
- Create: `modules/m02_color_spaces.py`
- Create: `modules/m03_spatial_filters.py`
- Test: `tests/test_tier1_foundations.py`

**Interfaces:**
- `m01_point_ops.py`:
  - `adjust_brightness_contrast(img: np.ndarray, brightness: int, contrast: float) -> np.ndarray`
  - `gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray`
  - `invert_image(img: np.ndarray) -> np.ndarray`
  - `compute_histogram(img: np.ndarray) -> dict`
  - `equalize_histogram(img: np.ndarray, clahe: bool = False, clip_limit: float = 2.0) -> np.ndarray`
- `m02_color_spaces.py`:
  - `convert_color_space(img: np.ndarray, target: str) -> np.ndarray`
  - `split_channels(img: np.ndarray) -> list[np.ndarray]`
  - `isolate_color_hsv(img_rgb: np.ndarray, lower_hsv: tuple, upper_hsv: tuple) -> tuple[np.ndarray, np.ndarray]`
  - `gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray`
- `m03_spatial_filters.py`:
  - `apply_blur(img: np.ndarray, method: str, ksize: int, sigma: float = 1.0) -> np.ndarray`
  - `apply_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray`
  - `apply_custom_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray`
  - `apply_bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray`

- [ ] **Step 1: Write tests for Tier 1 Foundations**
Create `tests/test_tier1_foundations.py` validating each function's output shape, dtype (`uint8`), range `[0, 255]`, and numerical correctness.

- [ ] **Step 2: Run test to verify failure**
Run: `.\.venv\Scripts\pytest tests/test_tier1_foundations.py`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement `modules/m01_point_ops.py`, `m02_color_spaces.py`, `m03_spatial_filters.py`**
Implement vectorized NumPy and OpenCV functions ensuring no in-place mutation, correct clamping, and color consistency.

- [ ] **Step 4: Run test to verify passing**
Run: `.\.venv\Scripts\pytest tests/test_tier1_foundations.py`
Expected: PASS.

---

### Task 3: Tier 2 Intermediate (Modules 4 & 5 - Morphology & Edge Detection)

**Files:**
- Create: `modules/m04_morphology.py`
- Create: `modules/m05_edge_detection.py`
- Test: `tests/test_tier2_morph_edges.py`

**Interfaces:**
- `m04_morphology.py`:
  - `morphology_op(img: np.ndarray, op_type: str, kernel_shape: str, ksize: int, iterations: int = 1) -> np.ndarray`
  - `extract_boundary(img: np.ndarray, ksize: int = 3) -> np.ndarray`
  - `skeletonize_binary(bin_img: np.ndarray) -> np.ndarray`
  - `separate_touching_objects(bin_img: np.ndarray, min_distance: int = 10) -> tuple[np.ndarray, int]`
- `m05_edge_detection.py`:
  - `sobel_edges(gray_img: np.ndarray, ksize: int = 3, direction: str = 'both') -> np.ndarray`
  - `laplacian_edges(gray_img: np.ndarray, ksize: int = 3) -> np.ndarray`
  - `canny_edges(gray_img: np.ndarray, low_thresh: int = 50, high_thresh: int = 150, aperture: int = 3) -> np.ndarray`
  - `laplacian_of_gaussian(gray_img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray`

- [ ] **Step 1: Write tests for Morphology and Edge Detection**
Create `tests/test_tier2_morph_edges.py` testing erosion, dilation, opening, closing, watershed separation, Sobel, Laplacian, and Canny pipelines.

- [ ] **Step 2: Run test to verify failure**
Run: `.\.venv\Scripts\pytest tests/test_tier2_morph_edges.py`
Expected: FAIL.

- [ ] **Step 3: Implement `m04_morphology.py` and `m05_edge_detection.py`**
Include custom boundary extractors, morphological watershed separation, and gradient operators.

- [ ] **Step 4: Run test to verify passing**
Run: `.\.venv\Scripts\pytest tests/test_tier2_morph_edges.py`
Expected: PASS.

---

### Task 4: Tier 2 Intermediate (Modules 6 & 7 - Fourier & Noise Restoration)

**Files:**
- Create: `modules/m06_frequency_domain.py`
- Create: `modules/m07_noise_restoration.py`
- Test: `tests/test_tier2_fourier_noise.py`

**Interfaces:**
- `m06_frequency_domain.py`:
  - `compute_fft2(gray_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]` # returns (shifted_dft, magnitude_spectrum_display)
  - `frequency_filter(gray_img: np.ndarray, filter_type: str, cutoff: float, order: int = 2) -> np.ndarray`
  - `remove_periodic_noise(gray_img: np.ndarray, notch_centers: list[tuple[int, int]], radius: int = 5) -> np.ndarray`
  - `create_hybrid_image(img_low: np.ndarray, img_high: np.ndarray, cutoff_low: float, cutoff_high: float) -> np.ndarray`
- `m07_noise_restoration.py`:
  - `add_noise(img: np.ndarray, noise_type: str, **kwargs) -> np.ndarray` # 'gaussian', 'salt_pepper', 'speckle'
  - `restore_image(img: np.ndarray, method: str, **kwargs) -> np.ndarray` # 'median', 'bilateral', 'nlm', 'wiener'
  - `compute_quality_metrics(original: np.ndarray, restored: np.ndarray) -> dict` # returns {'psnr': float, 'ssim': float, 'mse': float}

- [ ] **Step 1: Write tests for Fourier Transform and Restoration**
Create `tests/test_tier2_fourier_noise.py` covering 2D FFT, Gaussian/Butterworth low/high pass filtering, noise generation, and PSNR/SSIM calculation.

- [ ] **Step 2: Run test to verify failure**
Run: `.\.venv\Scripts\pytest tests/test_tier2_fourier_noise.py`
Expected: FAIL.

- [ ] **Step 3: Implement `m06_frequency_domain.py` and `m07_noise_restoration.py`**
Implement FFT with `np.fft`, frequency masks, SciPy Wiener restoration, OpenCV fastNlMeansDenoising, and SSIM metrics (`skimage.metrics.structural_similarity`).

- [ ] **Step 4: Run test to verify passing**
Run: `.\.venv\Scripts\pytest tests/test_tier2_fourier_noise.py`
Expected: PASS.

---

### Task 5: Tier 3 Advanced (Modules 8, 9, 10 - Segmentation, Features, Transforms)

**Files:**
- Create: `modules/m08_thresholding_seg.py`
- Create: `modules/m09_features.py`
- Create: `modules/m10_transforms.py`
- Test: `tests/test_tier3_advanced.py`

**Interfaces:**
- `m08_thresholding_seg.py`:
  - `adaptive_thresholding(gray_img: np.ndarray, method: str, block_size: int, c: int) -> np.ndarray`
  - `multi_otsu_threshold(gray_img: np.ndarray, classes: int = 3) -> np.ndarray`
  - `grabcut_segmentation(img_rgb: np.ndarray, rect: tuple[int, int, int, int], iterations: int = 5) -> tuple[np.ndarray, np.ndarray]`
  - `analyze_contours(bin_img: np.ndarray, min_area: int = 50) -> list[dict]` # returns list of {area, perimeter, circularity, bbox}
- `m09_features.py`:
  - `detect_harris_corners(gray_img: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04, threshold: float = 0.01) -> np.ndarray`
  - `detect_keypoints(gray_img: np.ndarray, detector_type: str = 'ORB', n_features: int = 500) -> tuple[list, np.ndarray]`
  - `match_features(img1: np.ndarray, img2: np.ndarray, detector_type: str = 'ORB', ratio_thresh: float = 0.75) -> tuple[np.ndarray, list]`
- `m10_transforms.py`:
  - `affine_transform(img: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> np.ndarray`
  - `four_point_perspective_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray`
  - `stitch_panorama(images: list[np.ndarray]) -> tuple[bool, np.ndarray]`

- [ ] **Step 1: Write tests for Advanced Tier**
Create `tests/test_tier3_advanced.py` asserting feature extraction keypoints, perspective homography rectification, and contour geometric metrics.

- [ ] **Step 2: Run test to verify failure**
Run: `.\.venv\Scripts\pytest tests/test_tier3_advanced.py`
Expected: FAIL.

- [ ] **Step 3: Implement `m08_thresholding_seg.py`, `m09_features.py`, `m10_transforms.py`**
Implement GrabCut, Harris corner response, ORB/SIFT matchers with Lowe's ratio test, and perspective warper.

- [ ] **Step 4: Run test to verify passing**
Run: `.\.venv\Scripts\pytest tests/test_tier3_advanced.py`
Expected: PASS.

---

### Task 6: Tier 4 Qualcomm-Level (Modules 11 & 12 - Video & Camera ISP)

**Files:**
- Create: `modules/m11_video.py`
- Create: `modules/m12_isp_pipeline.py`
- Test: `tests/test_tier4_qualcomm.py`

**Interfaces:**
- `m11_video.py`:
  - `compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 30) -> np.ndarray`
  - `apply_background_subtraction(frames: list[np.ndarray], method: str = 'MOG2') -> list[np.ndarray]`
  - `compute_dense_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]` # returns (flow_xy, flow_vis_rgb)
  - `generate_motion_heatmap(frames: list[np.ndarray], alpha: float = 0.05) -> np.ndarray`
- `m12_isp_pipeline.py`:
  - `simulate_bayer_cfa(img_rgb: np.ndarray, pattern: str = 'RGGB') -> np.ndarray`
  - `demosaic_bayer(bayer_img: np.ndarray, pattern: str = 'RGGB') -> np.ndarray`
  - `black_level_subtraction(img: np.ndarray, black_level: int = 16) -> np.ndarray`
  - `auto_white_balance(img: np.ndarray, method: str = 'gray_world') -> np.ndarray`
  - `apply_color_correction_matrix(img: np.ndarray, ccm: np.ndarray = None) -> np.ndarray`
  - `apply_tone_curve(img: np.ndarray, curve_type: str = 'sRGB') -> np.ndarray`
  - `hdr_merge_mertens(exposure_bracket: list[np.ndarray]) -> np.ndarray`
  - `run_full_isp_pipeline(bayer_raw: np.ndarray, config: dict = None) -> dict` # returns stage intermediate outputs + final RGB

- [ ] **Step 1: Write tests for Video and ISP Simulation**
Create `tests/test_tier4_qualcomm.py` testing Bayer mosaic downsampling and reconstruction, White Balance gains, HDR Mertens fusion, and optical flow vectors.

- [ ] **Step 2: Run test to verify failure**
Run: `.\.venv\Scripts\pytest tests/test_tier4_qualcomm.py`
Expected: FAIL.

- [ ] **Step 3: Implement `m11_video.py` and `m12_isp_pipeline.py`**
Implement the simulated hardware ISP pipeline step-by-step with stage dictionaries, and Farneback dense optical flow visualization in HSV space.

- [ ] **Step 4: Run test to verify passing**
Run: `.\.venv\Scripts\pytest tests/test_tier4_qualcomm.py`
Expected: PASS.

---

### Task 7: Streamlit Interactive UI Application (`app.py`)

**Files:**
- Create: `app.py`
- Test: `tests/test_app_integration.py`

**Interfaces:**
- Streamlit application hosting all 12 modules with:
  - Sidebar: Module navigation (grouped into 4 Tiers: Foundations, Intermediate, Advanced, Qualcomm ISP & Video).
  - Image Source selector: "Preset Sample Image" vs "Upload Custom Image".
  - Main panel: Side-by-side original vs processed visualization (`st.columns(2)`).
  - Parameter controls: Live sliders, toggles, dropdowns.
  - Collapsible Theory Expander: "📖 How it Works" explaining mathematical equations and algorithms.
  - Download Button: Download processed output PNG.

- [ ] **Step 1: Write integration tests for UI helpers**
Create `tests/test_app_integration.py` validating that every module can load sample images from `dataset/samples/` and execute without errors.

- [ ] **Step 2: Run test to verify initial state**
Run: `.\.venv\Scripts\pytest tests/test_app_integration.py`
Expected: FAIL if `app.py` or sample mappings are not defined.

- [ ] **Step 3: Implement `app.py`**
Write clean, modular Streamlit app importing each of the 12 modules with preset sample defaults.

- [ ] **Step 4: Verify Streamlit app runs headless**
Run: `.\.venv\Scripts\python -c "import app; print('App syntax OK')"`
Expected: Clean load without syntax errors.

---

### Task 8: End-to-End Verification & Documentation Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run complete test suite**
Run: `.\.venv\Scripts\pytest tests/ -v`
Expected: 100% tests passing across all tiers.

- [ ] **Step 2: Update `README.md`**
Document all 12 modules with mathematical foundations, Qualcomm ISP pipeline explanation, Streamlit run commands, and CLI options.
