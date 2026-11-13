import cv2
import numpy as np


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Computes absolute frame difference for basic motion detection.
    """
    g1 = ensure_grayscale(frame1)
    g2 = ensure_grayscale(frame2)
    diff = cv2.absdiff(g1, g2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def apply_background_subtraction(frames: list[np.ndarray], method: str = "MOG2") -> list[np.ndarray]:
    """
    Applies Gaussian Mixture-based (MOG2) or KNN background subtraction over a sequence of video frames.
    """
    if method.upper() == "KNN":
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)
    else:
        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    foreground_masks = []
    for f in frames:
        gray = ensure_grayscale(f)
        fg_mask = subtractor.apply(gray)
        foreground_masks.append(fg_mask)

    return foreground_masks


def compute_dense_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes dense optical flow vectors using the Gunner Farnebäck algorithm.
    Returns:
        flow: (H, W, 2) vector field
        flow_vis: RGB color-coded representation (Hue = angle/direction, Value = velocity magnitude)
    """
    g1 = ensure_grayscale(prev_gray)
    g2 = ensure_grayscale(curr_gray)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Convert flow vectors to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Visualize in HSV color space
    hsv = np.zeros((g1.shape[0], g1.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue corresponds to motion angle
    hsv[..., 1] = 255                    # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value is speed

    flow_vis_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow, flow_vis_rgb


def generate_motion_heatmap(frames: list[np.ndarray], alpha: float = 0.05) -> np.ndarray:
    """
    Accumulates temporal frame differences over time to generate a cumulative activity/motion heatmap.
    """
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = frames[0].shape[:2]
    accumulated = np.zeros((h, w), dtype=np.float32)

    for i in range(1, len(frames)):
        diff = compute_frame_difference(frames[i - 1], frames[i], threshold=20)
        accumulated += (diff.astype(np.float32) / 255.0)

    # Normalize accumulator
    norm_acc = cv2.normalize(accumulated, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(norm_acc, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
