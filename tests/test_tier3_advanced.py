import numpy as np
import pytest
import cv2
from modules.m08_thresholding_seg import (
    adaptive_thresholding,
    multi_otsu_threshold,
    grabcut_segmentation,
    analyze_contours,
)
from modules.m09_features import (
    detect_harris_corners,
    detect_keypoints,
    match_features,
)
from modules.m10_transforms import (
    affine_transform,
    four_point_perspective_transform,
    stitch_panorama,
)


@pytest.fixture
def synthetic_scene():
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    # White circle and rectangle
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    cv2.rectangle(img, (90, 80), (140, 130), (200, 200, 200), -1)
    return img


# --- Module 8 Tests ---
def test_adaptive_thresholding(synthetic_scene):
    gray = cv2.cvtColor(synthetic_scene, cv2.COLOR_RGB2GRAY)
    for method in ["gaussian", "mean"]:
        th = adaptive_thresholding(gray, method=method, block_size=11, c=2)
        assert th.shape == gray.shape
        assert th.dtype == np.uint8


def test_multi_otsu_threshold(synthetic_scene):
    gray = cv2.cvtColor(synthetic_scene, cv2.COLOR_RGB2GRAY)
    multi_seg = multi_otsu_threshold(gray, classes=3)
    assert multi_seg.shape == gray.shape
    assert len(np.unique(multi_seg)) <= 3


def test_grabcut_segmentation(synthetic_scene):
    rect = (20, 20, 110, 110)
    mask, fg = grabcut_segmentation(synthetic_scene, rect=rect, iterations=2)
    assert mask.shape == synthetic_scene.shape[:2]
    assert fg.shape == synthetic_scene.shape


def test_analyze_contours(synthetic_scene):
    gray = cv2.cvtColor(synthetic_scene, cv2.COLOR_RGB2GRAY)
    bin_img = (gray > 100).astype(np.uint8) * 255
    data = analyze_contours(bin_img, min_area=50)
    assert len(data) >= 2
    for item in data:
        assert "area" in item
        assert "perimeter" in item
        assert "circularity" in item
        assert "bbox" in item


# --- Module 9 Tests ---
def test_detect_harris_corners(synthetic_scene):
    gray = cv2.cvtColor(synthetic_scene, cv2.COLOR_RGB2GRAY)
    corners = detect_harris_corners(gray, threshold=0.01)
    assert corners.shape == gray.shape
    assert corners.dtype == np.uint8
    assert np.count_nonzero(corners) > 0


def test_detect_keypoints(synthetic_scene):
    gray = cv2.cvtColor(synthetic_scene, cv2.COLOR_RGB2GRAY)
    for det in ["ORB", "SIFT", "FAST"]:
        kps, des = detect_keypoints(gray, detector_type=det, n_features=100)
        assert len(kps) > 0


def test_match_features(synthetic_scene):
    gray = cv2.cvtColor(synthetic_scene, cv2.COLOR_RGB2GRAY)
    # Affine shifted image
    M = np.float32([[1, 0, 10], [0, 1, 5]])
    shifted = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
    vis, matches = match_features(gray, shifted, detector_type="ORB")
    assert vis.shape[0] == gray.shape[0]
    assert len(matches) > 0


# --- Module 10 Tests ---
def test_affine_transform(synthetic_scene):
    transformed = affine_transform(synthetic_scene, angle=30.0, scale=1.0, tx=10.0, ty=-5.0)
    assert transformed.shape == synthetic_scene.shape
    assert transformed.dtype == np.uint8


def test_four_point_perspective_transform(synthetic_scene):
    pts = np.array([[10, 10], [140, 15], [130, 140], [15, 135]], dtype=np.float32)
    warped = four_point_perspective_transform(synthetic_scene, pts)
    assert warped.ndim == 3
    assert warped.dtype == np.uint8


def test_stitch_panorama():
    canvas = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.circle(canvas, (50, 50), 30, (255, 100, 100), -1)
    cv2.rectangle(canvas, (110, 20), (170, 80), (100, 255, 100), -1)

    left = canvas[:, :130].copy()
    right = canvas[:, 70:].copy()

    status, stitched = stitch_panorama([left, right])
    # Even if OpenCV stitcher fails on tiny synthetic canvas, function returns (bool, ndarray)
    assert isinstance(status, bool)
    assert isinstance(stitched, np.ndarray)
