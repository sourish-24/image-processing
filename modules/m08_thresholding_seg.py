import cv2
import numpy as np
from skimage.filters import threshold_multiotsu


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def adaptive_thresholding(gray_img: np.ndarray, method: str = "gaussian", block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Applies adaptive thresholding ('mean' or 'gaussian') to handle non-uniform lighting.
    """
    gray = ensure_grayscale(gray_img)
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(gray, 255, adaptive_method, cv2.THRESH_BINARY_INV, block_size, c)


def multi_otsu_threshold(gray_img: np.ndarray, classes: int = 3) -> np.ndarray:
    """
    Multi-level Otsu thresholding: partitions image into multiple intensity classes.
    """
    gray = ensure_grayscale(gray_img)
    classes = max(2, min(5, int(classes)))
    thresholds = threshold_multiotsu(gray, classes=classes)
    regions = np.digitize(gray, bins=thresholds).astype(np.uint8)
    # Scale labels for clear visualization
    return (regions * (255 // (classes - 1))).astype(np.uint8)


def grabcut_segmentation(img_rgb: np.ndarray, rect: tuple[int, int, int, int], iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Interactive/bounding box foreground extraction using GrabCut algorithm (Gaussian Mixture Models).
    rect: (x, y, w, h)
    Returns:
        mask (2D uint8, 0/255)
        foreground_rgb (3D uint8 with background masked out)
    """
    # GrabCut operates in BGR format
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Validate rectangle dimensions
    x, y, w, h = rect
    x = max(0, min(x, img_rgb.shape[1] - 1))
    y = max(0, min(y, img_rgb.shape[0] - 1))
    w = max(1, min(w, img_rgb.shape[1] - x))
    h = max(1, min(h, img_rgb.shape[0] - y))
    valid_rect = (x, y, w, h)

    cv2.grabCut(img_bgr, mask, valid_rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    foreground_rgb = img_rgb * final_mask[:, :, np.newaxis]
    return (final_mask * 255).astype(np.uint8), foreground_rgb


def analyze_contours(bin_img: np.ndarray, min_area: int = 50) -> list[dict]:
    r"""
    Finds and analyzes geometric properties of foreground contours:
    Area, perimeter, bounding box, centroid, circularity ($4 \pi \times \text{area} / \text{perimeter}^2$).
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        circularity = round((4 * np.pi * area) / (perimeter ** 2), 3) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(cnt)

        # Centroid
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

        results.append({
            "area": area,
            "perimeter": round(perimeter, 2),
            "circularity": circularity,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
        })

    return results
