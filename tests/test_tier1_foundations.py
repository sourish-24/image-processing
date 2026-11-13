import numpy as np
import pytest
from modules.m01_point_ops import (
    adjust_brightness_contrast,
    gamma_correction,
    invert_image,
    compute_histogram,
    equalize_histogram,
)
from modules.m02_color_spaces import (
    convert_color_space,
    split_channels,
    isolate_color_hsv,
    gray_world_white_balance,
)
from modules.m03_spatial_filters import (
    apply_blur,
    apply_sharpen,
    apply_custom_kernel,
    apply_bilateral,
)


@pytest.fixture
def sample_rgb():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = (255, 0, 0)
    img[50:, :50] = (0, 255, 0)
    img[:50, 50:] = (0, 0, 255)
    img[50:, 50:] = (128, 128, 128)
    return img


@pytest.fixture
def sample_gray():
    return np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))


# --- Module 1 Tests ---
def test_adjust_brightness_contrast(sample_rgb):
    out = adjust_brightness_contrast(sample_rgb, brightness=30, contrast=1.2)
    assert out.shape == sample_rgb.shape
    assert out.dtype == np.uint8
    assert np.all(out >= 0) and np.all(out <= 255)


def test_gamma_correction(sample_rgb):
    out = gamma_correction(sample_rgb, gamma=2.2)
    assert out.shape == sample_rgb.shape
    assert out.dtype == np.uint8


def test_invert_image(sample_rgb):
    out = invert_image(sample_rgb)
    assert np.all(out + sample_rgb == 255)


def test_histogram_and_equalize(sample_gray, sample_rgb):
    hist = compute_histogram(sample_gray)
    assert "gray" in hist
    assert len(hist["gray"]) == 256

    hist_rgb = compute_histogram(sample_rgb)
    assert "r" in hist_rgb and "g" in hist_rgb and "b" in hist_rgb

    eq = equalize_histogram(sample_gray, clahe=False)
    assert eq.shape == sample_gray.shape
    assert eq.dtype == np.uint8

    eq_clahe = equalize_histogram(sample_rgb, clahe=True, clip_limit=3.0)
    assert eq_clahe.shape == sample_rgb.shape
    assert eq_clahe.dtype == np.uint8


# --- Module 2 Tests ---
def test_color_space_conversion(sample_rgb):
    hsv = convert_color_space(sample_rgb, "HSV")
    assert hsv.shape == sample_rgb.shape
    ycrcb = convert_color_space(sample_rgb, "YCrCb")
    assert ycrcb.shape == sample_rgb.shape
    lab = convert_color_space(sample_rgb, "LAB")
    assert lab.shape == sample_rgb.shape


def test_split_channels(sample_rgb):
    channels = split_channels(sample_rgb)
    assert len(channels) == 3
    for c in channels:
        assert c.shape == (100, 100)


def test_isolate_color_hsv(sample_rgb):
    # Blue is at (0, 0, 255) in RGB
    # In OpenCV HSV, Blue Hue is around 110-130
    mask, isolated = isolate_color_hsv(sample_rgb, (100, 150, 150), (140, 255, 255))
    assert mask.shape == (100, 100)
    assert isolated.shape == sample_rgb.shape
    assert isolated.dtype == np.uint8


def test_gray_world_white_balance(sample_rgb):
    # Make heavily tinted image
    tinted = sample_rgb.astype(np.float32)
    tinted[:, :, 0] *= 2.0  # boost red
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    wb = gray_world_white_balance(tinted)
    assert wb.shape == sample_rgb.shape
    assert wb.dtype == np.uint8


# --- Module 3 Tests ---
def test_spatial_blur(sample_rgb):
    for method in ["gaussian", "median", "box"]:
        blurred = apply_blur(sample_rgb, method=method, ksize=5, sigma=1.2)
        assert blurred.shape == sample_rgb.shape
        assert blurred.dtype == np.uint8


def test_spatial_sharpen(sample_rgb):
    sharp = apply_sharpen(sample_rgb, strength=1.5)
    assert sharp.shape == sample_rgb.shape
    assert sharp.dtype == np.uint8


def test_custom_kernel(sample_rgb):
    # 3x3 identity kernel
    k = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    out = apply_custom_kernel(sample_rgb, k)
    assert np.allclose(out, sample_rgb)


def test_bilateral_filter(sample_rgb):
    out = apply_bilateral(sample_rgb, d=5, sigma_color=50, sigma_space=50)
    assert out.shape == sample_rgb.shape
    assert out.dtype == np.uint8
