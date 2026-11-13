import cv2
import numpy as np


def simulate_bayer_cfa(img_rgb: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Simulates raw sensor readout through a Bayer Color Filter Array (CFA).
    Default RGGB layout:
      Row 0: R  Gr
      Row 1: Gb B
    """
    h, w = img_rgb.shape[:2]
    # Ensure even dimensions for 2x2 Bayer grid
    if h % 2 != 0:
        img_rgb = img_rgb[: h - 1, :]
    if w % 2 != 0:
        img_rgb = img_rgb[:, : w - 1]
    h, w = img_rgb.shape[:2]

    bayer = np.zeros((h, w), dtype=np.uint8)
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    pattern = pattern.upper()
    if pattern == "RGGB":
        bayer[0::2, 0::2] = r[0::2, 0::2]
        bayer[0::2, 1::2] = g[0::2, 1::2]
        bayer[1::2, 0::2] = g[1::2, 0::2]
        bayer[1::2, 1::2] = b[1::2, 1::2]
    elif pattern == "BGGR":
        bayer[0::2, 0::2] = b[0::2, 0::2]
        bayer[0::2, 1::2] = g[0::2, 1::2]
        bayer[1::2, 0::2] = g[1::2, 0::2]
        bayer[1::2, 1::2] = r[1::2, 1::2]
    else:
        # Default fallback to RGGB
        bayer[0::2, 0::2] = r[0::2, 0::2]
        bayer[0::2, 1::2] = g[0::2, 1::2]
        bayer[1::2, 0::2] = g[1::2, 0::2]
        bayer[1::2, 1::2] = b[1::2, 1::2]

    return bayer


def demosaic_bayer(bayer_img: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Demosaicing (Debayering) block: reconstructs full 3-channel RGB image from raw 2D Bayer sensor mosaic.
    """
    pattern = pattern.upper()
    if pattern == "RGGB":
        code = cv2.COLOR_BayerRG2RGB
    elif pattern == "BGGR":
        code = cv2.COLOR_BayerBG2RGB
    elif pattern == "GRBG":
        code = cv2.COLOR_BayerGR2RGB
    elif pattern == "GBRG":
        code = cv2.COLOR_BayerGB2RGB
    else:
        code = cv2.COLOR_BayerRG2RGB

    return cv2.cvtColor(bayer_img, code)


def black_level_subtraction(img: np.ndarray, black_level: int = 16) -> np.ndarray:
    """
    Hardware ISP Black Level Correction (BLC):
    Subtracts dark current sensor offset and rescales dynamic range to [0, 255].
    """
    img_f = img.astype(np.float32)
    sub = np.maximum(0.0, img_f - black_level)
    # Rescale back to 255
    rescaled = sub * (255.0 / max(1.0, 255.0 - black_level))
    return np.clip(rescaled, 0, 255).astype(np.uint8)


def auto_white_balance(img: np.ndarray, method: str = "gray_world") -> np.ndarray:
    """
    ISP Auto White Balance (AWB) block:
    Estimates illuminant and applies per-channel gains to eliminate color casts.
    """
    img_f = img.astype(np.float32)
    method = method.lower()

    if method == "perfect_reflector":
        # White patch assumption
        max_r = np.percentile(img_f[:, :, 0], 99)
        max_g = np.percentile(img_f[:, :, 1], 99)
        max_b = np.percentile(img_f[:, :, 2], 99)
        ref = max(max_r, max_g, max_b)
        if max_r > 0:
            img_f[:, :, 0] *= (ref / max_r)
        if max_g > 0:
            img_f[:, :, 1] *= (ref / max_g)
        if max_b > 0:
            img_f[:, :, 2] *= (ref / max_b)
    else:
        # Gray world assumption
        mean_r = np.mean(img_f[:, :, 0])
        mean_g = np.mean(img_f[:, :, 1])
        mean_b = np.mean(img_f[:, :, 2])
        mean_gray = (mean_r + mean_g + mean_b) / 3.0
        if mean_r > 0:
            img_f[:, :, 0] *= (mean_gray / mean_r)
        if mean_g > 0:
            img_f[:, :, 1] *= (mean_gray / mean_g)
        if mean_b > 0:
            img_f[:, :, 2] *= (mean_gray / mean_b)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_color_correction_matrix(img: np.ndarray, ccm: np.ndarray = None) -> np.ndarray:
    """
    ISP Color Correction Matrix (CCM):
    3x3 matrix multiplication converting sensor spectral response to standard sRGB color space.
    """
    if ccm is None:
        # Standard balanced daylight CCM
        ccm = np.array([
            [1.25, -0.15, -0.10],
            [-0.10, 1.20, -0.10],
            [-0.05, -0.15, 1.20]
        ], dtype=np.float32)

    img_f = img.astype(np.float32)
    # Reshape for matrix multiplication
    orig_shape = img.shape
    pixels = img_f.reshape(-1, 3)
    corrected = np.dot(pixels, ccm.T)
    return np.clip(corrected.reshape(orig_shape), 0, 255).astype(np.uint8)


def apply_tone_curve(img: np.ndarray, curve_type: str = "sRGB", gamma: float = 2.2) -> np.ndarray:
    """
    ISP Gamma & Tone Mapping Curve:
    Maps linear sensor intensities to display non-linear transfer function.
    """
    norm = img.astype(np.float32) / 255.0
    if curve_type.lower() == "srgb":
        # IEC 61966-2-1 sRGB tone transfer function
        linear_mask = norm <= 0.0031308
        srgb = np.where(linear_mask, 12.92 * norm, 1.055 * (norm ** (1.0 / 2.4)) - 0.055)
        return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
    else:
        # Standard power-law gamma
        toned = (norm ** (1.0 / gamma)) * 255.0
        return np.clip(toned, 0, 255).astype(np.uint8)


def hdr_merge_mertens(exposure_bracket: list[np.ndarray]) -> np.ndarray:
    """
    Computational Photography / HDR:
    Exposure fusion (Mertens algorithm) merging under/over-exposed brackets without ghosting.
    """
    if not exposure_bracket:
        raise ValueError("Exposure bracket cannot be empty.")
    if len(exposure_bracket) == 1:
        return exposure_bracket[0]

    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(exposure_bracket)
    # Mertens outputs float32 in [0, 1]
    return np.clip(fusion * 255.0, 0, 255).astype(np.uint8)


def run_full_isp_pipeline(bayer_raw: np.ndarray, config: dict = None) -> dict:
    """
    Simulates a full end-to-end hardware camera Image Signal Processor (ISP):
    Raw Bayer -> Demosaicing -> Black Level Subtraction -> Auto White Balance -> CCM -> Tone Mapping -> Denoise
    """
    if config is None:
        config = {
            "pattern": "RGGB",
            "black_level": 16,
            "awb_method": "gray_world",
            "tone_curve": "sRGB",
            "denoise": True,
        }

    stages = {}
    stages["bayer"] = bayer_raw

    # 1. Demosaic
    demosaiced = demosaic_bayer(bayer_raw, pattern=config.get("pattern", "RGGB"))
    stages["demosaiced"] = demosaiced

    # 2. Black Level Subtraction
    bls = black_level_subtraction(demosaiced, black_level=config.get("black_level", 16))
    stages["bls"] = bls

    # 3. Auto White Balance
    awb = auto_white_balance(bls, method=config.get("awb_method", "gray_world"))
    stages["awb"] = awb

    # 4. Color Correction Matrix
    ccm = apply_color_correction_matrix(awb)
    stages["ccm"] = ccm

    # 5. Tone Mapping / Gamma
    toned = apply_tone_curve(ccm, curve_type=config.get("tone_curve", "sRGB"))
    stages["tone_mapped"] = toned

    # 6. Final ISP output (with optional edge-preserving bilateral filter)
    if config.get("denoise", True):
        final = cv2.bilateralFilter(toned, d=5, sigmaColor=30, sigmaSpace=30)
    else:
        final = toned
    stages["final"] = final

    return stages
