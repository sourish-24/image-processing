import numpy as np
import pytest
import cv2
from modules.m04_morphology import (
    morphology_op,
    extract_boundary,
    skeletonize_binary,
    separate_touching_objects,
)
from modules.m05_edge_detection import (
    sobel_edges,
    laplacian_edges,
    canny_edges,
    laplacian_of_gaussian,
)


@pytest.fixture
def binary_square():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 255
    return img


@pytest.fixture
def touching_circles():
    # Two overlapping circles
    img = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(img, (50, 60), 30, 255, -1)
    cv2.circle(img, (90, 60), 30, 255, -1)
    return img


# --- Module 4 Tests ---
def test_morphology_operations(binary_square):
    for op in ["erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat"]:
        out = morphology_op(binary_square, op_type=op, kernel_shape="rect", ksize=5, iterations=1)
        assert out.shape == binary_square.shape
        assert out.dtype == np.uint8

    # Erosion should shrink the white area
    eroded = morphology_op(binary_square, "erosion", "rect", 5)
    assert np.sum(eroded > 0) < np.sum(binary_square > 0)

    # Dilation should expand the white area
    dilated = morphology_op(binary_square, "dilation", "rect", 5)
    assert np.sum(dilated > 0) > np.sum(binary_square > 0)


def test_extract_boundary(binary_square):
    boundary = extract_boundary(binary_square, ksize=3)
    assert boundary.shape == binary_square.shape
    assert boundary.dtype == np.uint8
    # Center of square should now be 0 in boundary image
    assert boundary[50, 50] == 0


def test_skeletonize_binary(binary_square):
    skel = skeletonize_binary(binary_square)
    assert skel.shape == binary_square.shape
    assert skel.dtype == np.uint8
    assert np.sum(skel > 0) < np.sum(binary_square > 0)


def test_separate_touching_objects(touching_circles):
    labels, count = separate_touching_objects(touching_circles, min_distance=15)
    assert count >= 2
    assert labels.shape == touching_circles.shape


# --- Module 5 Tests ---
def test_sobel_edges(binary_square):
    for direction in ["both", "x", "y"]:
        edges = sobel_edges(binary_square, ksize=3, direction=direction)
        assert edges.shape == binary_square.shape
        assert edges.dtype == np.uint8
        assert np.max(edges) > 0


def test_laplacian_edges(binary_square):
    edges = laplacian_edges(binary_square, ksize=3)
    assert edges.shape == binary_square.shape
    assert edges.dtype == np.uint8
    assert np.max(edges) > 0


def test_canny_edges(binary_square):
    edges = canny_edges(binary_square, low_thresh=50, high_thresh=150)
    assert edges.shape == binary_square.shape
    assert edges.dtype == np.uint8
    assert np.max(edges) == 255


def test_laplacian_of_gaussian(binary_square):
    edges = laplacian_of_gaussian(binary_square, ksize=5, sigma=1.0)
    assert edges.shape == binary_square.shape
    assert edges.dtype == np.uint8
    assert np.max(edges) > 0
