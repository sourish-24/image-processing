import numpy as np
import pytest
import cv2
from modules.m06_frequency_domain import (
    compute_fft2,
    frequency_filter,
    remove_periodic_noise,
    create_hybrid_image,
)
from modules.m07_noise_restoration import (
    add_noise,
    restore_image,
    compute_quality_metrics,
)


@pytest.fixture
def synthetic_pattern():
    img = np.zeros((128, 128), dtype=np.uint8)
    img[32:96, 32:96] = 200
    cv2.circle(img, (64, 64), 20, 50, -1)
    return img


# --- Module 6 Tests ---
def test_fft2_computation(synthetic_pattern):
    dft_shift, mag_spec = compute_fft2(synthetic_pattern)
    assert dft_shift.shape == synthetic_pattern.shape
    assert mag_spec.shape == synthetic_pattern.shape
    assert mag_spec.dtype == np.uint8


def test_frequency_filtering(synthetic_pattern):
    for ftype in ["ideal_lowpass", "gaussian_lowpass", "butterworth_lowpass", "gaussian_highpass"]:
        filtered = frequency_filter(synthetic_pattern, filter_type=ftype, cutoff=30.0, order=2)
        assert filtered.shape == synthetic_pattern.shape
        assert filtered.dtype == np.uint8


def test_remove_periodic_noise(synthetic_pattern):
    # Add sinusoidal periodic noise
    y, x = np.indices(synthetic_pattern.shape)
    noise = (30 * np.sin(2 * np.pi * x / 16.0)).astype(np.int16)
    noisy = np.clip(synthetic_pattern.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    h, w = synthetic_pattern.shape
    cy, cx = h // 2, w // 2
    # Notch filters symmetrically placed
    notches = [(cy, cx + 8), (cy, cx - 8)]
    cleaned = remove_periodic_noise(noisy, notch_centers=notches, radius=6)
    assert cleaned.shape == synthetic_pattern.shape
    assert cleaned.dtype == np.uint8


def test_create_hybrid_image(synthetic_pattern):
    img2 = 255 - synthetic_pattern
    hybrid = create_hybrid_image(synthetic_pattern, img2, cutoff_low=20.0, cutoff_high=40.0)
    assert hybrid.shape == synthetic_pattern.shape
    assert hybrid.dtype == np.uint8


# --- Module 7 Tests ---
def test_add_noise(synthetic_pattern):
    for ntype in ["gaussian", "salt_pepper", "speckle"]:
        noisy = add_noise(synthetic_pattern, noise_type=ntype)
        assert noisy.shape == synthetic_pattern.shape
        assert noisy.dtype == np.uint8
        assert not np.array_equal(noisy, synthetic_pattern)


def test_restore_image(synthetic_pattern):
    noisy = add_noise(synthetic_pattern, "salt_pepper", amount=0.04)
    for method in ["median", "bilateral", "nlm"]:
        restored = restore_image(noisy, method=method)
        assert restored.shape == synthetic_pattern.shape
        assert restored.dtype == np.uint8


def test_quality_metrics(synthetic_pattern):
    noisy = add_noise(synthetic_pattern, "gaussian", std=15)
    metrics = compute_quality_metrics(synthetic_pattern, noisy)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "mse" in metrics
    assert metrics["psnr"] > 0
    assert 0.0 <= metrics["ssim"] <= 1.0
