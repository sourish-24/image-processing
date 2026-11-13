import cv2
import numpy as np


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def detect_harris_corners(gray_img: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04, threshold: float = 0.01) -> np.ndarray:
    r"""
    Computes Harris Corner response map: $R = \det(M) - k \cdot (\text{trace}(M))^2$
    Returns:
        corner_mask: 2D uint8 mask with 255 at detected corner locations.
    """
    gray = ensure_grayscale(gray_img).astype(np.float32)
    dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    # Dilate for visualization marking
    dst = cv2.dilate(dst, None)
    corner_mask = np.zeros(gray.shape, dtype=np.uint8)
    thresh_val = threshold * dst.max()
    corner_mask[dst > thresh_val] = 255
    return corner_mask


def detect_keypoints(gray_img: np.ndarray, detector_type: str = "ORB", n_features: int = 500) -> tuple[list, np.ndarray]:
    """
    Extracts keypoints and feature descriptors using ORB, SIFT, or FAST.
    Returns:
        keypoints: list of cv2.KeyPoint
        descriptors: ndarray of shape (N, feature_dim)
    """
    gray = ensure_grayscale(gray_img)
    det = detector_type.upper()

    if det == "ORB":
        detector = cv2.ORB_create(nfeatures=n_features)
        kps, des = detector.detectAndCompute(gray, None)
    elif det == "SIFT":
        detector = cv2.SIFT_create(nfeatures=n_features)
        kps, des = detector.detectAndCompute(gray, None)
    elif det == "FAST":
        fast = cv2.FastFeatureDetector_create(threshold=25)
        kps = fast.detect(gray, None)
        # Compute descriptors using BRIEF via ORB
        orb = cv2.ORB_create()
        kps, des = orb.compute(gray, kps)
    else:
        raise ValueError(f"Unsupported detector: {detector_type}")

    return kps, des


def match_features(img1: np.ndarray, img2: np.ndarray, detector_type: str = "ORB", ratio_thresh: float = 0.75) -> tuple[np.ndarray, list]:
    """
    Matches features between two images using Brute-Force Matcher with Lowe's ratio test.
    Returns:
        vis_img: RGB image visualizing good matches with connecting lines.
        good_matches: list of cv2.DMatch objects passing ratio test.
    """
    gray1 = ensure_grayscale(img1)
    gray2 = ensure_grayscale(img2)

    kps1, des1 = detect_keypoints(gray1, detector_type)
    kps2, des2 = detect_keypoints(gray2, detector_type)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        vis = np.hstack([gray1, gray2])
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB), []

    # Norm type: HAMMING for binary descriptors (ORB/BRIEF), L2 for floating-point (SIFT)
    norm_type = cv2.NORM_HAMMING if detector_type.upper() in ["ORB", "FAST"] else cv2.NORM_L2
    bf = cv2.BFMatcher(norm_type)

    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    vis = cv2.drawMatches(
        img1 if img1.ndim == 3 else cv2.cvtColor(gray1, cv2.COLOR_GRAY2RGB),
        kps1,
        img2 if img2.ndim == 3 else cv2.cvtColor(gray2, cv2.COLOR_GRAY2RGB),
        kps2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis, good_matches
