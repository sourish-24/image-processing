import cv2
import numpy as np


def apply_blur(img: np.ndarray, method: str = "gaussian", ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies spatial smoothing filter ('gaussian', 'median', 'box').
    Ensures ksize is an odd integer >= 3.
    """
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    method = method.lower()
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    elif method == "median":
        return cv2.medianBlur(img, ksize)
    elif method in ["box", "mean", "average"]:
        return cv2.blur(img, (ksize, ksize))
    else:
        raise ValueError(f"Unsupported blur method: {method}")


def apply_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Applies unsharp masking: sharp = img + strength * (img - blurred)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3.0)
    diff = img.astype(np.float32) - blurred.astype(np.float32)
    sharp = img.astype(np.float32) + strength * diff
    return np.clip(sharp, 0, 255).astype(np.uint8)


def apply_custom_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies 2D convolution with an arbitrary user-supplied kernel matrix.
    """
    kernel = np.array(kernel, dtype=np.float32)
    filtered = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Applies bilateral filter for edge-preserving noise reduction.
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
