import cv2
import numpy as np


def convert_color_space(img_rgb: np.ndarray, target: str) -> np.ndarray:
    """
    Converts RGB image to target color space ('HSV', 'YCrCb', 'LAB', 'GRAY').
    """
    target = target.upper()
    if target == "HSV":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    elif target in ["YCRCB", "YUV"]:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    elif target == "LAB":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    elif target in ["GRAY", "GRAYSCALE"]:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    elif target == "RGB":
        return img_rgb.copy()
    else:
        raise ValueError(f"Unsupported color space target: {target}")


def split_channels(img: np.ndarray) -> list[np.ndarray]:
    """
    Splits multi-channel image into individual 2D arrays.
    """
    if img.ndim == 2:
        return [img.copy()]
    return [img[:, :, i].copy() for i in range(img.shape[2])]


def isolate_color_hsv(img_rgb: np.ndarray, lower_hsv: tuple, upper_hsv: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolates a color range in HSV space.
    Returns:
        mask (2D uint8, 0/255)
        isolated_rgb (3D uint8 with non-selected pixels set to 0)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
    isolated = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    return mask, isolated


def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    """
    Implements the classical Gray-World assumption algorithm (fundamental ISP AWB).
    Assumes average scene reflectance is neutral gray, scaling R, G, B channels by their mean ratios.
    """
    img_f = img_rgb.astype(np.float32)
    mean_r = np.mean(img_f[:, :, 0])
    mean_g = np.mean(img_f[:, :, 1])
    mean_b = np.mean(img_f[:, :, 2])

    # Overall gray level
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    if mean_r > 0:
        img_f[:, :, 0] *= (mean_gray / mean_r)
    if mean_g > 0:
        img_f[:, :, 1] *= (mean_gray / mean_g)
    if mean_b > 0:
        img_f[:, :, 2] *= (mean_gray / mean_b)

    return np.clip(img_f, 0, 255).astype(np.uint8)
