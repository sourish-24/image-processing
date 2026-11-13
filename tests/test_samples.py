import os
import cv2
import pytest

EXPECTED_SAMPLES = [
    "sourish.jpg",
    "cameraman.png",
    "coins.png",
    "building.png",
    "noisy_sample.png",
    "text_doc.png",
    "raw_bayer.png",
    "panorama_left.png",
    "panorama_right.png",
]

def test_samples_exist_and_readable():
    sample_dir = os.path.join("dataset", "samples")
    assert os.path.isdir(sample_dir), "Sample directory must exist"
    for name in EXPECTED_SAMPLES:
        p = os.path.join(sample_dir, name)
        assert os.path.isfile(p), f"Missing sample {name}"
        img = cv2.imread(p)
        assert img is not None, f"Failed to load image {name}"
        assert img.shape[0] >= 50 and img.shape[1] >= 50
