import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.m01_point_ops import adjust_brightness_contrast, gamma_correction, equalize_histogram
from modules.m02_color_spaces import convert_color_space, gray_world_white_balance
from modules.m03_spatial_filters import apply_bilateral, apply_sharpen
from modules.m04_morphology import separate_touching_objects
from modules.m05_edge_detection import sobel_edges, canny_edges, laplacian_of_gaussian
from modules.m06_frequency_domain import compute_fft2, frequency_filter
from modules.m07_noise_restoration import add_noise, restore_image, compute_quality_metrics
from modules.m09_features import detect_keypoints, match_features
from modules.m11_video import compute_dense_optical_flow
from modules.m12_isp_pipeline import simulate_bayer_cfa, run_full_isp_pipeline

os.makedirs("assets", exist_ok=True)

# Helper function to add a title banner to an image
def add_banner(img, text, font_scale=0.6, bg_color=(20, 20, 20), text_color=(255, 255, 255)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    banner_h = int(max(32, h * 0.12))
    canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = bg_color
    canvas[banner_h:, :] = img
    
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (banner_h + th) // 2 - 2
    cv2.putText(canvas, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return canvas

# -------------------------------------------------------------------------
# 1. Camera ISP Pipeline (4 Stages)
# -------------------------------------------------------------------------
print("Generating demo_isp_pipeline.png...")
raw_bayer = cv2.imread("dataset/samples/raw_bayer.png", cv2.IMREAD_GRAYSCALE)
if raw_bayer is None:
    # fallback to simulated
    ref = cv2.imread("dataset/samples/sourish.jpg")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    raw_bayer = simulate_bayer_cfa(ref)

isp_stages = run_full_isp_pipeline(raw_bayer)
p1 = add_banner(cv2.resize(isp_stages["bayer"], (260, 260)), "1. Raw Bayer CFA Pattern")
p2 = add_banner(cv2.resize(isp_stages["demosaiced"], (260, 260)), "2. Bilinear Demosaicing")
p3 = add_banner(cv2.resize(isp_stages["awb"], (260, 260)), "3. Auto White Balance (AWB)")
p4 = add_banner(cv2.resize(isp_stages["final"], (260, 260)), "4. Tone Curve + Denoised")
demo_isp = np.hstack([p1, p2, p3, p4])
cv2.imwrite("assets/demo_isp_pipeline.png", cv2.cvtColor(demo_isp, cv2.COLOR_RGB2BGR))

# -------------------------------------------------------------------------
# 2. 2D Fourier Transform & Frequency Filtering
# -------------------------------------------------------------------------
print("Generating demo_frequency_fft.png...")
cameraman = cv2.imread("dataset/samples/cameraman.png", cv2.IMREAD_GRAYSCALE)
if cameraman is not None:
    cameraman_res = cv2.resize(cameraman, (320, 320))
    _, mag_spec = compute_fft2(cameraman_res)
    hp_filtered = frequency_filter(cameraman_res, filter_type="butterworth_highpass", cutoff=40.0, order=2)
    
    p1 = add_banner(cameraman_res, "Spatial Domain (Cameraman)")
    p2 = add_banner(mag_spec, "2D FFT Log Magnitude Spectrum")
    p3 = add_banner(hp_filtered, "Butterworth High-Pass Filtered")
    demo_fft = np.hstack([p1, p2, p3])
    cv2.imwrite("assets/demo_frequency_fft.png", cv2.cvtColor(demo_fft, cv2.COLOR_RGB2BGR))

# -------------------------------------------------------------------------
# 3. Feature Detection & Keypoint Matching (ORB / SIFT)
# -------------------------------------------------------------------------
print("Generating demo_feature_matching.png...")
building = cv2.imread("dataset/samples/building.png")
if building is not None:
    building = cv2.cvtColor(building, cv2.COLOR_BGR2RGB)
    building = cv2.resize(building, (400, 300))
    h, w = building.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 18, 0.95)
    rotated = cv2.warpAffine(building, M, (w, h))
    match_vis, matches = match_features(building, rotated, detector_type="ORB")
    demo_match = add_banner(match_vis, f"ORB Feature Matching under 18 deg Rotation & Scale ({len(matches)} Inliers via Lowe's Ratio)")
    cv2.imwrite("assets/demo_feature_matching.png", cv2.cvtColor(demo_match, cv2.COLOR_RGB2BGR))

# -------------------------------------------------------------------------
# 4. Multi-Stage Edge Detection (Sobel -> Canny -> LoG)
# -------------------------------------------------------------------------
print("Generating demo_edge_detection.png...")
if building is not None:
    gray_b = cv2.cvtColor(building, cv2.COLOR_RGB2GRAY)
    sobel = sobel_edges(gray_b, ksize=3, direction="both")
    canny = canny_edges(gray_b, low_thresh=60, high_thresh=160)
    log = laplacian_of_gaussian(gray_b, ksize=7, sigma=1.2)
    
    sz = (260, 260)
    p1 = add_banner(cv2.resize(gray_b, sz), "Input Scene")
    p2 = add_banner(cv2.resize(sobel, sz), "Sobel Gradient Magnitude")
    p3 = add_banner(cv2.resize(canny, sz), "4-Stage Canny Edge Map")
    p4 = add_banner(cv2.resize(log, sz), "Laplacian of Gaussian (LoG)")
    demo_edges = np.hstack([p1, p2, p3, p4])
    cv2.imwrite("assets/demo_edge_detection.png", cv2.cvtColor(demo_edges, cv2.COLOR_RGB2BGR))

# -------------------------------------------------------------------------
# 5. Morphological Watershed & Touching Object Separation
# -------------------------------------------------------------------------
print("Generating demo_morphology_watershed.png...")
coins = cv2.imread("dataset/samples/coins.png", cv2.IMREAD_GRAYSCALE)
if coins is not None:
    coins_res = cv2.resize(coins, (320, 320))
    _, binary_coins = cv2.threshold(coins_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_coins, cv2.DIST_L2, 5)
    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Watershed
    labels, count = separate_touching_objects(binary_coins, min_distance=15)
    norm_labels = (labels * (255 // max(1, count))).astype(np.uint8)
    ws_color = cv2.applyColorMap(norm_labels, cv2.COLORMAP_JET)
    ws_color = cv2.cvtColor(ws_color, cv2.COLOR_BGR2RGB)
    
    p1 = add_banner(binary_coins, "Binarized Input (Coins)")
    p2 = add_banner(dist_vis, "Euclidean Distance Transform")
    p3 = add_banner(ws_color, f"Watershed Clustered ({count} Coins)")
    demo_morph = np.hstack([p1, p2, p3])
    cv2.imwrite("assets/demo_morphology_watershed.png", cv2.cvtColor(demo_morph, cv2.COLOR_RGB2BGR))

# -------------------------------------------------------------------------
# 6. Point Operations & CLAHE Enhancement
# -------------------------------------------------------------------------
print("Generating demo_enhancement.png...")
sourish = cv2.imread("dataset/samples/sourish.jpg")
if sourish is not None:
    sourish = cv2.cvtColor(sourish, cv2.COLOR_BGR2RGB)
    sourish_res = cv2.resize(sourish, (260, 260))
    gamma_img = gamma_correction(sourish_res, gamma=1.6)
    clahe_img = equalize_histogram(sourish_res, clahe=True, clip_limit=3.0)
    bilateral_img = apply_bilateral(sourish_res, d=9, sigma_color=75, sigma_space=75)
    
    p1 = add_banner(sourish_res, "1. Original RGB")
    p2 = add_banner(gamma_img, "2. Gamma Tuned (gamma=1.6)")
    p3 = add_banner(clahe_img, "3. CLAHE Equalized")
    p4 = add_banner(bilateral_img, "4. Bilateral Denoised")
    demo_enhance = np.hstack([p1, p2, p3, p4])
    cv2.imwrite("assets/demo_enhancement.jpg", cv2.cvtColor(demo_enhance, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if os.path.exists("assets/demo_enhancement.png"):
        os.remove("assets/demo_enhancement.png")

# -------------------------------------------------------------------------
# 7. Noise Models & Restoration with PSNR/SSIM
# -------------------------------------------------------------------------
print("Generating demo_denoising_restoration.jpg...")
if sourish is not None:
    sourish_res = cv2.resize(sourish, (320, 320))
    noisy_img = add_noise(sourish_res, "gaussian", std=35.0)
    restored_img = restore_image(noisy_img, method="nlm")
    
    m_noisy = compute_quality_metrics(sourish_res, noisy_img)
    m_rest = compute_quality_metrics(sourish_res, restored_img)
    
    p1 = add_banner(sourish_res, "Ground Truth")
    p2 = add_banner(noisy_img, f"Gaussian Noise ({m_noisy['psnr']} dB)")
    p3 = add_banner(restored_img, f"NLM Restored ({m_rest['psnr']} dB / +{round(m_rest['psnr']-m_noisy['psnr'], 1)}dB)")
    demo_denoise = np.hstack([p1, p2, p3])
    cv2.imwrite("assets/demo_denoising_restoration.jpg", cv2.cvtColor(demo_denoise, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if os.path.exists("assets/demo_denoising_restoration.png"):
        os.remove("assets/demo_denoising_restoration.png")

# -------------------------------------------------------------------------
# 8. Dense Optical Flow (Farnebäck Motion Field)
# -------------------------------------------------------------------------
print("Generating demo_optical_flow.png...")
if cameraman is not None:
    frame0 = cv2.resize(cameraman, (320, 320))
    h, w = frame0.shape[:2]
    M = np.float32([[1, 0, 16], [0, 1, 6]])
    frame1 = cv2.warpAffine(frame0, M, (w, h))
    _, flow_vis = compute_dense_optical_flow(frame0, frame1)
    
    p1 = add_banner(frame0, "Video Frame t")
    p2 = add_banner(frame1, "Video Frame t+1 (Panning Motion)")
    p3 = add_banner(flow_vis, "Dense Optical Flow (Farneback)")
    demo_flow = np.hstack([p1, p2, p3])
    cv2.imwrite("assets/demo_optical_flow.png", cv2.cvtColor(demo_flow, cv2.COLOR_RGB2BGR))

print("All README assets successfully generated in assets/!")
