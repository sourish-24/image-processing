import os
import cv2
import pytest
import numpy as np


def test_app_sample_mapping():
    # Verify that all default sample images referenced by the app exist and are valid
    sample_dir = os.path.join("dataset", "samples")
    assert os.path.isdir(sample_dir)
    samples = [
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
    for s in samples:
        path = os.path.join(sample_dir, s)
        assert os.path.isfile(path), f"Sample {s} not found"
        img = cv2.imread(path)
        assert img is not None, f"Failed reading sample {s}"


def test_app_imports_cleanly():
    # Verify app syntax and imports
    import app
    assert hasattr(app, "main")
