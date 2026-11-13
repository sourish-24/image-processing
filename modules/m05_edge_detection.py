import cv2
import numpy as np


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def sobel_edges(gray_img: np.ndarray, ksize: int = 3, direction: str = "both") -> np.ndarray:
    """
    Computes first-order spatial derivative edges using Sobel operators.
    """
    gray = ensure_grayscale(gray_img)
    direction = direction.lower()

    if direction == "x":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        return np.clip(np.abs(grad_x), 0, 255).astype(np.uint8)
    elif direction == "y":
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        return np.clip(np.abs(grad_y), 0, 255).astype(np.uint8)
    else:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = cv2.magnitude(grad_x, grad_y)
        return np.clip(mag, 0, 255).astype(np.uint8)


def laplacian_edges(gray_img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Computes second-order spatial derivative edges using Laplacian operator.
    """
    gray = ensure_grayscale(gray_img)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def canny_edges(gray_img: np.ndarray, low_thresh: int = 50, high_thresh: int = 150, aperture: int = 3) -> np.ndarray:
    """
    Applies the full Canny edge detection pipeline (Gaussian -> Sobel -> NMS -> Hysteresis).
    """
    gray = ensure_grayscale(gray_img)
    return cv2.Canny(gray, threshold1=low_thresh, threshold2=high_thresh, apertureSize=aperture)


def laplacian_of_gaussian(gray_img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies Laplacian of Gaussian (LoG): Gaussian smoothing followed by Laplacian.
    """
    gray = ensure_grayscale(gray_img)
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)
