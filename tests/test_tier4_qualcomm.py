import numpy as np
import pytest
import cv2
from modules.m11_video import (
    compute_frame_difference,
    apply_background_subtraction,
    compute_dense_optical_flow,
    generate_motion_heatmap,
)
from modules.m12_isp_pipeline import (
    simulate_bayer_cfa,
    demosaic_bayer,
    black_level_subtraction,
    auto_white_balance,
    apply_color_correction_matrix,
    apply_tone_curve,
    hdr_merge_mertens,
    run_full_isp_pipeline,
)


@pytest.fixture
def frame_pair():
    f1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(f1, (30, 50), 15, 200, -1)
    f2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(f2, (45, 50), 15, 200, -1)  # moved right by 15px
    return f1, f2


@pytest.fixture
def sample_rgb_scene():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:, :25] = (200, 30, 30)
    img[:, 25:55] = (30, 200, 30)
    img[:, 55:] = (30, 30, 200)
    return img


# --- Module 11 Video Tests ---
def test_compute_frame_difference(frame_pair):
    f1, f2 = frame_pair
    diff = compute_frame_difference(f1, f2, threshold=20)
    assert diff.shape == f1.shape
    assert diff.dtype == np.uint8
    assert np.count_nonzero(diff) > 0


def test_background_subtraction(frame_pair):
    f1, f2 = frame_pair
    frames = [f1, f1, f2]
    fg_masks = apply_background_subtraction(frames, method="MOG2")
    assert len(fg_masks) == 3
    for m in fg_masks:
        assert m.shape == f1.shape
        assert m.dtype == np.uint8


def test_dense_optical_flow(frame_pair):
    f1, f2 = frame_pair
    flow_xy, flow_vis = compute_dense_optical_flow(f1, f2)
    assert flow_xy.shape == (100, 100, 2)
    assert flow_vis.shape == (100, 100, 3)
    assert flow_vis.dtype == np.uint8
    # Motion was along x-axis
    assert np.max(np.abs(flow_xy[:, :, 0])) > 0


def test_motion_heatmap(frame_pair):
    f1, f2 = frame_pair
    heatmap = generate_motion_heatmap([f1, f2, f1])
    assert heatmap.shape == (100, 100, 3)
    assert heatmap.dtype == np.uint8


# --- Module 12 Camera ISP Tests ---
def test_bayer_simulate_and_demosaic(sample_rgb_scene):
    bayer = simulate_bayer_cfa(sample_rgb_scene, pattern="RGGB")
    assert bayer.shape == sample_rgb_scene.shape[:2]
    assert bayer.dtype == np.uint8

    demosaiced = demosaic_bayer(bayer, pattern="RGGB")
    assert demosaiced.shape == sample_rgb_scene.shape
    assert demosaiced.dtype == np.uint8


def test_black_level_subtraction(sample_rgb_scene):
    bls = black_level_subtraction(sample_rgb_scene, black_level=16)
    assert bls.shape == sample_rgb_scene.shape
    assert bls.dtype == np.uint8
    assert np.all(bls <= sample_rgb_scene)


def test_auto_white_balance(sample_rgb_scene):
    awb = auto_white_balance(sample_rgb_scene, method="gray_world")
    assert awb.shape == sample_rgb_scene.shape
    assert awb.dtype == np.uint8


def test_color_correction_matrix(sample_rgb_scene):
    ccm_img = apply_color_correction_matrix(sample_rgb_scene)
    assert ccm_img.shape == sample_rgb_scene.shape
    assert ccm_img.dtype == np.uint8


def test_apply_tone_curve(sample_rgb_scene):
    toned = apply_tone_curve(sample_rgb_scene, curve_type="sRGB")
    assert toned.shape == sample_rgb_scene.shape
    assert toned.dtype == np.uint8


def test_hdr_merge_mertens(sample_rgb_scene):
    dark = (sample_rgb_scene * 0.5).astype(np.uint8)
    bright = np.clip(sample_rgb_scene * 1.5, 0, 255).astype(np.uint8)
    hdr = hdr_merge_mertens([dark, sample_rgb_scene, bright])
    assert hdr.shape == sample_rgb_scene.shape
    assert hdr.dtype == np.uint8


def test_run_full_isp_pipeline(sample_rgb_scene):
    bayer = simulate_bayer_cfa(sample_rgb_scene, "RGGB")
    stages = run_full_isp_pipeline(bayer)
    assert "bayer" in stages
    assert "demosaiced" in stages
    assert "bls" in stages
    assert "awb" in stages
    assert "ccm" in stages
    assert "tone_mapped" in stages
    assert "final" in stages
    assert stages["final"].shape == sample_rgb_scene.shape
