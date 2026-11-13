import cv2
import numpy as np


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """
    Applies linear point transformation: out = contrast * img + brightness
    Clamped strictly to [0, 255].
    """
    res = img.astype(np.float32) * contrast + brightness
    return np.clip(res, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applies power-law transformation: out = 255 * (img / 255) ** (1 / gamma)
    """
    if gamma <= 0:
        raise ValueError("Gamma must be strictly positive.")
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def invert_image(img: np.ndarray) -> np.ndarray:
    """
    Photographic negative transformation: out = 255 - img
    """
    return (255 - img).astype(np.uint8)


def compute_histogram(img: np.ndarray) -> dict:
    """
    Computes 256-bin histogram for single channel or per-channel RGB.
    Returns dictionary with counts.
    """
    if img.ndim == 2:
        hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
        return {"gray": hist}
    elif img.ndim == 3 and img.shape[2] == 3:
        # Assumed RGB format
        r_hist, _ = np.histogram(img[:, :, 0].flatten(), bins=256, range=[0, 256])
        g_hist, _ = np.histogram(img[:, :, 1].flatten(), bins=256, range=[0, 256])
        b_hist, _ = np.histogram(img[:, :, 2].flatten(), bins=256, range=[0, 256])
        return {"r": r_hist, "g": g_hist, "b": b_hist}
    else:
        raise ValueError("Unsupported image shape for histogram computation.")


def equalize_histogram(img: np.ndarray, clahe: bool = False, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies global histogram equalization or Contrast Limited Adaptive Histogram Equalization (CLAHE).
    For RGB, equalization is performed in the Y (luminance) channel of YCrCb to preserve hue.
    """
    if img.ndim == 2:
        if clahe:
            cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return cl.apply(img)
        return cv2.equalizeHist(img)
    elif img.ndim == 3:
        # Convert RGB to YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if clahe:
            cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            ycrcb[:, :, 0] = cl.apply(ycrcb[:, :, 0])
        else:
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB).")
