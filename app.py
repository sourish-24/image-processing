import io
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Import all 12 IPCV modules
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


def load_image(uploaded_file, sample_name=None):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif sample_name:
        path = os.path.join("dataset", "samples", sample_name)
        if os.path.isfile(path):
            img_bgr = cv2.imread(path)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Fallback default
    fallback_path = os.path.join("dataset", "samples", "cameraman.png")
    if os.path.isfile(fallback_path):
        return cv2.cvtColor(cv2.imread(fallback_path), cv2.COLOR_BGR2RGB)
    return np.zeros((200, 200, 3), dtype=np.uint8)


def get_image_download_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def main():
    st.set_page_config(
        page_title="Image Processing & Computer Vision Portfolio - Sourish",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Image Processing & Computer Vision Portfolio")
    st.sidebar.markdown("**Sourish Musib**")
    st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/)")

    MODULES = [
        "1. Point Operations & Histograms",
        "2. Color Space Exploration",
        "3. Spatial Filtering & Convolution",
        "4. Morphological Operations",
        "5. Edge Detection (Sobel → Canny → LoG)",
        "6. 2D Fourier Transform & Frequency Filtering",
        "7. Noise Models & Restoration (PSNR/SSIM)",
        "8. Thresholding & Contour Segmentation",
        "9. Feature Detection & Matching (Harris, ORB, SIFT)",
        "10. Geometric Transforms & Panorama Stitching",
        "11. Video Processing & Optical Flow",
        "12. Camera ISP Pipeline Simulation",
    ]

    selected_module = st.sidebar.selectbox("Select Domain Module", MODULES)

    # Map each module to recommended sample images
    SAMPLE_MAPPINGS = {
        "1. Point Operations & Histograms": "sourish.jpg",
        "2. Color Space Exploration": "sourish.jpg",
        "3. Spatial Filtering & Convolution": "sourish.jpg",
        "4. Morphological Operations": "coins.png",
        "5. Edge Detection (Sobel → Canny → LoG)": "building.png",
        "6. 2D Fourier Transform & Frequency Filtering": "cameraman.png",
        "7. Noise Models & Restoration (PSNR/SSIM)": "noisy_sample.png",
        "8. Thresholding & Contour Segmentation": "coins.png",
        "9. Feature Detection & Matching (Harris, ORB, SIFT)": "building.png",
        "10. Geometric Transforms & Panorama Stitching": "panorama_left.png",
        "11. Video Processing & Optical Flow": "cameraman.png",
        "12. Camera ISP Pipeline Simulation": "raw_bayer.png",
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Image Source")
    img_source = st.sidebar.radio("Input Mode", ["Preset Sample Image", "Upload Custom Image"])

    default_sample = SAMPLE_MAPPINGS.get(selected_module, "sourish.jpg")
    uploaded_file = None
    sample_choice = default_sample

    if img_source == "Preset Sample Image":
        available_samples = [
            "sourish.jpg",
            "cameraman.png",
            "coins.png",
            "building.png",
            "noisy_sample.png",
            "raw_bayer.png",
            "panorama_left.png",
            "panorama_right.png",
        ]
        sample_choice = st.sidebar.selectbox(
            "Select Sample Image",
            available_samples,
            index=available_samples.index(default_sample) if default_sample in available_samples else 0,
        )
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Image (PNG, JPG)", type=["png", "jpg", "jpeg", "webp"])

    current_img = load_image(uploaded_file, sample_choice)

    # -------------------------------------------------------------
    # Module Execution
    # -------------------------------------------------------------
    st.title(selected_module)

    if selected_module.startswith("1."):
        # Module 1: Point Operations
        st.write("Pixel-level non-linear transformations, gamma correction, and histogram equalization.")
        c1, c2, c3 = st.columns(3)
        with c1:
            brightness = st.slider("Brightness Offset", -100, 100, 0)
            contrast = st.slider("Contrast Multiplier", 0.1, 3.0, 1.0, 0.1)
        with c2:
            gamma = st.slider(r"Gamma ($\gamma$)", 0.1, 4.0, 1.0, 0.1)
            invert = st.checkbox("Negative (Invert)")
        with c3:
            eq_mode = st.selectbox("Histogram Equalization", ["None", "Global Equalization", "CLAHE (Adaptive)"])
            clip_limit = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0, 0.5)

        processed = adjust_brightness_contrast(current_img, brightness, contrast)
        if gamma != 1.0:
            processed = gamma_correction(processed, gamma)
        if invert:
            processed = invert_image(processed)
        if eq_mode == "Global Equalization":
            processed = equalize_histogram(processed, clahe=False)
        elif eq_mode == "CLAHE (Adaptive)":
            processed = equalize_histogram(processed, clahe=True, clip_limit=clip_limit)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Original Image")
            st.image(current_img, width="stretch")
            hist_orig = compute_histogram(current_img)
            st.line_chart(hist_orig)
        with colB:
            st.subheader("Processed Image")
            st.image(processed, width="stretch")
            hist_proc = compute_histogram(processed)
            st.line_chart(hist_proc)
            st.download_button(
                "⬇ Download Result",
                get_image_download_bytes(processed),
                "point_ops.png",
                "image/png",
            )

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown(r"""
            **Point Operations & Camera Auto-Exposure:**
            - **Gamma Correction:** Camera sensors have a linear response ($I \propto \Phi$), whereas human vision approximates a logarithmic / power law ($\gamma \approx 2.2$). In hardware ISP pipelines, non-linear gamma curves are encoded as fast Look-Up Tables (LUTs) in digital signal processors.
            - **CLAHE:** Standard global histogram equalization over-amplifies noise in homogeneous regions (e.g. night skies). Contrast Limited Adaptive Histogram Equalization (CLAHE) computes localized histograms across $8 \times 8$ tiles and clips gradient slopes, essential for low-light camera enhancement.
            """)

    elif selected_module.startswith("2."):
        # Module 2: Color Space Exploration
        st.write("Color space decomposition (RGB, HSV, YCrCb, LAB) and Auto White Balance.")
        color_space = st.selectbox("Target Color Space Representation", ["RGB", "HSV", "YCrCb", "LAB", "GRAY"])
        converted = convert_color_space(current_img, color_space)

        tab1, tab2, tab3 = st.tabs(["Color Space Channels", "Color Isolation (HSV)", "Auto White Balance (AWB)"])
        with tab1:
            colA, colB = st.columns(2)
            with colA:
                st.image(current_img, caption="Original RGB", width="stretch")
            with colB:
                st.image(converted, caption=f"Converted: {color_space}", width="stretch")

            channels = split_channels(converted)
            st.write(f"Individual {color_space} Channels:")
            c_cols = st.columns(len(channels))
            for idx, (col_widget, ch_data) in enumerate(zip(c_cols, channels)):
                with col_widget:
                    st.image(ch_data, caption=f"Channel {idx + 1}", width="stretch")

        with tab2:
            st.write("Filter an object based on its Hue, Saturation, and Value range:")
            h_low, h_high = st.slider("Hue Range [0, 180]", 0, 180, (0, 180))
            s_low, s_high = st.slider("Saturation Range [0, 255]", 0, 255, (50, 255))
            v_low, v_high = st.slider("Value Range [0, 255]", 0, 255, (50, 255))
            mask, isolated = isolate_color_hsv(current_img, (h_low, s_low, v_low), (h_high, s_high, v_high))
            iso_cols = st.columns(2)
            with iso_cols[0]:
                st.image(mask, caption="Binary HSV Mask", width="stretch")
            with iso_cols[1]:
                st.image(isolated, caption="Isolated Color Segment", width="stretch")

        with tab3:
            st.write("Hardware ISP Auto White Balance (AWB) using the Gray-World illuminant estimator:")
            awb_result = gray_world_white_balance(current_img)
            awb_cols = st.columns(2)
            with awb_cols[0]:
                st.image(current_img, caption="Before White Balance", width="stretch")
            with awb_cols[1]:
                st.image(awb_result, caption="After Gray-World White Balance", width="stretch")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown("""
            **Luma-Chroma Separation & Mobile Video Codecs:**
            - Video compression codecs (H.264/HEVC/AV1) heavily rely on YCrCb/YUV 4:2:0 subsampling because the human visual system (HVS) possesses far greater spatial resolution for luminance ($Y$) than chrominance ($Cr, Cb$).
            - AWB (Auto White Balance) is one of the foundational "3A" camera algorithms running on hardware camera ISPs, ensuring neutral white surfaces remain chromatic neutral regardless of warm tungsten ($2800K$) or cool daylight ($6500K$) lighting.
            """)

    elif selected_module.startswith("3."):
        # Module 3: Spatial Filtering
        st.write("Convolution-based 2D filtering: smoothing, edge-preserving bilateral denoising, and unsharp masking.")
        f_mode = st.radio("Filter Type", ["Standard Blur", "Bilateral Filter", "Sharpen (Unsharp Mask)", "Custom Convolution Kernel"])

        if f_mode == "Standard Blur":
            b_method = st.selectbox("Method", ["gaussian", "median", "box"])
            ksize = st.slider("Kernel Size (pixels)", 3, 31, 5, step=2)
            sigma = st.slider(r"Gaussian Sigma ($\sigma$)", 0.1, 10.0, 1.5, 0.1)
            processed = apply_blur(current_img, method=b_method, ksize=ksize, sigma=sigma)
        elif f_mode == "Bilateral Filter":
            d = st.slider("Neighborhood Diameter (d)", 3, 25, 9, step=2)
            sc = st.slider("Sigma Color (Intensity Distance)", 10, 200, 75)
            ss = st.slider("Sigma Space (Spatial Distance)", 10, 200, 75)
            processed = apply_bilateral(current_img, d=d, sigma_color=sc, sigma_space=ss)
        elif f_mode == "Sharpen (Unsharp Mask)":
            strength = st.slider("Sharpening Strength", 0.1, 5.0, 1.5, 0.1)
            processed = apply_sharpen(current_img, strength=strength)
        else:
            k_preset = st.selectbox("Preset Kernel", ["Emboss", "Ridge / Edge", "Laplacian", "Identity"])
            if k_preset == "Emboss":
                k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            elif k_preset == "Ridge / Edge":
                k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            elif k_preset == "Laplacian":
                k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            else:
                k = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            st.write("Active Kernel Matrix:", k)
            processed = apply_custom_kernel(current_img, k)

        colA, colB = st.columns(2)
        with colA:
            st.image(current_img, caption="Original Image", width="stretch")
        with colB:
            st.image(processed, caption="Filtered Output", width="stretch")
            st.download_button("⬇ Download Result", get_image_download_bytes(processed), "filtered.png", "image/png")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown("""
            **2D Convolution Hardware Acceleration:**
            - **Bilateral Filtering:** Unlike Gaussian blur, the bilateral filter weights neighborhood pixels by both geometric proximity and radiometric intensity difference. It smoothes flat noisy patches while preserving crisp object boundaries—vital for camera noise reduction blocks.
            - **Edge Computing & DSPs:** Large 2D convolutions ($7 \times 7$ or separable $1 \times N$) are executed on embedded DSPs using vectorized SIMD instructions to achieve 60+ FPS processing with minimal power consumption.
            """)

    elif selected_module.startswith("4."):
        # Module 4: Morphology
        st.write("Non-linear mathematical morphology: structuring elements, erosion, dilation, skeletonization, and object separation.")
        gray_input = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        op_name = st.selectbox("Morphological Operation", ["erosion", "dilation", "opening", "closing", "gradient", "tophat", "blackhat", "skeleton", "watershed_separation"])
        shape = st.selectbox("Structuring Element Shape", ["rect", "ellipse", "cross"])
        ksize = st.slider("Structuring Element Size", 1, 25, 5, step=2)
        iters = st.slider("Iterations", 1, 10, 1)

        if op_name == "skeleton":
            result = skeletonize_binary(binary)
        elif op_name == "watershed_separation":
            labels, count = separate_touching_objects(binary, min_distance=15)
            # Colorize labels for visualization
            norm_labels = (labels * (255 // max(1, count))).astype(np.uint8)
            result = cv2.applyColorMap(norm_labels, cv2.COLORMAP_JET)
            st.success(f"Detected and separated {count} touching object(s) via Watershed.")
        else:
            result = morphology_op(binary, op_type=op_name, kernel_shape=shape, ksize=ksize, iterations=iters)

        colA, colB = st.columns(2)
        with colA:
            st.image(binary, caption="Binary Input Image", width="stretch")
        with colB:
            st.image(result, caption=f"Morphology Result: {op_name.capitalize()}", width="stretch")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown(r"""
            **Morphological Filters in Vision Pipelines:**
            - **Opening & Closing:** Morphological opening ($\gamma_B(A) = (A \ominus B) \oplus B$) eliminates salt noise without changing area, while closing ($\phi_B(A) = (A \oplus B) \ominus B$) bridges hairline fractures and holes.
            - **Distance Transform & Watershed:** Used in industrial inspection and automated cell/object counting where overlapping items fuse into single binary connected components.
            """)

    elif selected_module.startswith("5."):
        # Module 5: Edge Detection
        st.write("Gradient operators and the multi-stage Canny edge detector.")
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        detector = st.selectbox("Edge Detector", ["Canny", "Sobel (Combined)", "Sobel X", "Sobel Y", "Laplacian", "Laplacian of Gaussian (LoG)"])

        if detector == "Canny":
            low_t = st.slider("Low Hysteresis Threshold", 10, 200, 50)
            high_t = st.slider("High Hysteresis Threshold", 20, 300, 150)
            edges = canny_edges(gray, low_thresh=low_t, high_thresh=high_t)
        elif detector == "Sobel (Combined)":
            ksize = st.slider("Sobel Kernel Size", 1, 7, 3, step=2)
            edges = sobel_edges(gray, ksize=ksize, direction="both")
        elif detector == "Sobel X":
            edges = sobel_edges(gray, ksize=3, direction="x")
        elif detector == "Sobel Y":
            edges = sobel_edges(gray, ksize=3, direction="y")
        elif detector == "Laplacian":
            edges = laplacian_edges(gray, ksize=3)
        else:
            sigma = st.slider("LoG Sigma", 0.5, 5.0, 1.2, 0.1)
            edges = laplacian_of_gaussian(gray, ksize=7, sigma=sigma)

        colA, colB = st.columns(2)
        with colA:
            st.image(current_img, caption="Original Image", width="stretch")
        with colB:
            st.image(edges, caption=f"Edge Map ({detector})", width="stretch")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown("""
            **The 4 Stages of Canny Edge Detection:**
            1. **Gaussian Smoothing:** Suppress high-frequency sensor noise.
            2. **Gradient Computation:** Sobel kernel convolution producing gradient magnitude and orientation angle $\theta = \arctan(G_y / G_x)$.
            3. **Non-Maximum Suppression (NMS):** Thin blurred edges to 1-pixel wide ridges along gradient direction.
            4. **Hysteresis Thresholding:** Retain strong edges ($> T_{\text{high}}$) and weak edges connected to strong ones, discarding noise ($< T_{\text{low}}$).
            """)

    elif selected_module.startswith("6."):
        # Module 6: Fourier Transform
        st.write("Frequency domain analysis using the 2D Discrete Fourier Transform (DFT).")
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        dft_shift, mag_spec = compute_fft2(gray)

        task_mode = st.radio("Frequency Task", ["Ideal / Gaussian / Butterworth Filtering", "Periodic Noise Removal (Notch Filter)", "Hybrid Image Synthesis"])

        if task_mode == "Ideal / Gaussian / Butterworth Filtering":
            f_type = st.selectbox("Filter Mask", ["gaussian_lowpass", "gaussian_highpass", "butterworth_lowpass", "butterworth_highpass", "ideal_lowpass", "ideal_highpass"])
            cutoff = st.slider("Cutoff Frequency ($D_0$)", 5.0, 150.0, 40.0)
            order = st.slider("Butterworth Order (n)", 1, 6, 2)
            filtered = frequency_filter(gray, filter_type=f_type, cutoff=cutoff, order=order)

            colA, colB, colC = st.columns(3)
            with colA:
                st.image(gray, caption="Spatial Domain", width="stretch")
            with colB:
                st.image(mag_spec, caption="FFT Magnitude Spectrum (Log Scaled)", width="stretch")
            with colC:
                st.image(filtered, caption=f"Reconstructed: {f_type}", width="stretch")

        elif task_mode == "Periodic Noise Removal (Notch Filter)":
            st.write("Simulating periodic noise (e.g. electrical interference from power lines or sensor clock jitter):")
            # Inject noise
            y, x = np.indices(gray.shape)
            noise = (40 * np.sin(2 * np.pi * x / 16.0)).astype(np.int16)
            corrupted = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            _, noisy_mag = compute_fft2(corrupted)

            # Notches
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            notches = [(cy, cx + 8), (cy, cx - 8)]
            restored = remove_periodic_noise(corrupted, notches, radius=8)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(corrupted, caption="Corrupted with Periodic Noise", width="stretch")
            with c2:
                st.image(noisy_mag, caption="Spectrum with Noise Spikes", width="stretch")
            with c3:
                st.image(restored, caption="Notch Filter Restored", width="stretch")

        else:
            st.write("Combines low frequencies of Image A with high frequencies of Image B:")
            inverted_gray = 255 - gray
            c_low = st.slider("Low Cutoff", 5.0, 50.0, 20.0)
            c_high = st.slider("High Cutoff", 20.0, 100.0, 45.0)
            hybrid = create_hybrid_image(gray, inverted_gray, cutoff_low=c_low, cutoff_high=c_high)
            st.image(hybrid, caption="Hybrid Synthesis Output", width="stretch")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown(r"""
            **2D DFT and Frequency-Domain DSPs:**
            - **Convolution Theorem:** Spatial convolution corresponds to point-wise multiplication in the frequency domain: $\mathcal{F}\{f * h\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{h\}$.
            - For large spatial kernels ($> 15 \times 15$), computing convolution via FFT is significantly faster ($\mathcal{O}(N \log N)$ vs $\mathcal{O}(N^2)$), a standard DSP optimization technique in production embedded audio/image processing libraries.
            """)

    elif selected_module.startswith("7."):
        # Module 7: Noise & Restoration
        st.write("Evaluate restoration algorithms against quantified metrics (PSNR, SSIM, MSE).")
        ntype = st.selectbox("Noise Type to Inject", ["gaussian", "salt_pepper", "speckle"])
        if ntype == "gaussian":
            std = st.slider(r"Noise Std ($\sigma$)", 5.0, 60.0, 25.0)
            noisy = add_noise(current_img, "gaussian", std=std)
        elif ntype == "salt_pepper":
            amt = st.slider("Noise Density", 0.01, 0.20, 0.05, 0.01)
            noisy = add_noise(current_img, "salt_pepper", amount=amt)
        else:
            std = st.slider("Speckle Multiplier", 0.05, 0.5, 0.2, 0.05)
            noisy = add_noise(current_img, "speckle", std=std)

        restore_method = st.selectbox("Restoration Algorithm", ["nlm", "bilateral", "median", "gaussian"])
        restored = restore_image(noisy, method=restore_method)

        metrics_noisy = compute_quality_metrics(current_img, noisy)
        metrics_restored = compute_quality_metrics(current_img, restored)

        colA, colB, colC = st.columns(3)
        with colA:
            st.image(current_img, caption="Ground Truth", width="stretch")
        with colB:
            st.image(noisy, caption=f"Noisy (PSNR: {metrics_noisy['psnr']} dB | SSIM: {metrics_noisy['ssim']})", width="stretch")
        with colC:
            st.image(restored, caption=f"Restored via {restore_method.upper()} (PSNR: {metrics_restored['psnr']} dB | SSIM: {metrics_restored['ssim']})", width="stretch")

        st.metric(
            "PSNR Improvement",
            f"{metrics_restored['psnr']} dB",
            delta=f"+{round(metrics_restored['psnr'] - metrics_noisy['psnr'], 2)} dB",
        )

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown(r"""
            **Objective Image Quality Assessment (IQA):**
            - **PSNR (Peak Signal-to-Noise Ratio):** Measures logarithmic ratio between maximum pixel power and mean squared error:
              $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right)$$
            - **SSIM (Structural Similarity Index):** Models perception based on luminance, contrast, and structural correlation. A higher PSNR/SSIM score directly dictates tuning parameters for mobile camera certification.
            """)

    elif selected_module.startswith("8."):
        # Module 8: Thresholding & Segmentation
        st.write("Advanced binarization, multi-Otsu, GrabCut, and geometric contour analysis.")
        seg_tab1, seg_tab2, seg_tab3 = st.tabs(["Adaptive & Multi-Otsu", "Contour Geometric Analyzer", "GrabCut Segmentation"])
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)

        with seg_tab1:
            method = st.radio("Thresholding Method", ["Adaptive Gaussian", "Adaptive Mean", "Multi-Otsu (Multi-Class)"])
            if method.startswith("Adaptive"):
                bsize = st.slider("Block Window Size", 3, 51, 15, step=2)
                c_val = st.slider("Constant Offset (C)", 0, 20, 3)
                th_out = adaptive_thresholding(gray, method="gaussian" if "Gaussian" in method else "mean", block_size=bsize, c=c_val)
            else:
                classes = st.slider("Number of Intensity Classes", 2, 5, 3)
                th_out = multi_otsu_threshold(gray, classes=classes)

            c1, c2 = st.columns(2)
            with c1:
                st.image(current_img, caption="Input Scene", width="stretch")
            with c2:
                st.image(th_out, caption=f"Segmentation Map: {method}", width="stretch")

        with seg_tab2:
            st.write("Detect and classify contours based on Area, Perimeter, and Circularity:")
            min_area = st.slider("Min Area Filter (pixels)", 10, 5000, 100)
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            data = analyze_contours(bin_img, min_area=min_area)

            annotated = current_img.copy()
            for idx, item in enumerate(data, start=1):
                x, y, w, h = item["bbox"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(annotated, f"#{idx} C:{item['circularity']}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

            st.image(annotated, caption=f"Annotated Contours ({len(data)} detected)", width="stretch")
            st.dataframe(data)

        with seg_tab3:
            st.write("GrabCut interactive foreground extraction using Gaussian Mixture Models:")
            h, w = current_img.shape[:2]
            # Default bounding box at 10% margin
            rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
            iters = st.slider("GrabCut Iterations", 1, 10, 4)
            mask, foreground = grabcut_segmentation(current_img, rect=rect, iterations=iters)
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(mask, caption="Alpha Matte Mask", width="stretch")
            with gc2:
                st.image(foreground, caption="Extracted Foreground", width="stretch")

    elif selected_module.startswith("9."):
        # Module 9: Features
        st.write("Scale and rotation invariant local feature keypoints and descriptor matching.")
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        det_mode = st.selectbox("Feature Extractor", ["ORB", "SIFT", "Harris Corners", "FAST"])

        if det_mode == "Harris Corners":
            thresh = st.slider("Corner Sensitivity Threshold", 0.001, 0.10, 0.01, 0.005)
            corner_mask = detect_harris_corners(gray, threshold=thresh)
            vis = current_img.copy()
            vis[corner_mask > 0] = [255, 0, 0]  # Mark corners in Red
            st.image(vis, caption=f"Harris Corners Detected ({np.count_nonzero(corner_mask)} locations)", width="stretch")
        else:
            n_feat = st.slider("Max Features", 50, 2000, 500, 50)
            kps, des = detect_keypoints(gray, detector_type=det_mode, n_features=n_feat)
            vis = cv2.drawKeypoints(current_img, kps, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            st.image(vis, caption=f"{det_mode} Keypoints ({len(kps)} extracted)", width="stretch")

            st.subheader("Feature Matching under Geometric Perturbation")
            # Apply slight rotation & shift to create target image
            M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), 15, 1.0)
            transformed = cv2.warpAffine(current_img, M, (current_img.shape[1], current_img.shape[0]))
            match_vis, matches = match_features(current_img, transformed, detector_type=det_mode)
            st.image(match_vis, caption=f"Matches ({len(matches)} passing Lowe's ratio test)", width="stretch")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown("""
            **Mobile Feature Detectors:**
            - **ORB vs SIFT:** While SIFT provides superior scale invariance, its floating-point 128D descriptors require heavy compute. ORB (Oriented FAST and Rotated BRIEF) employs binary descriptors (256 bits), allowing Hamming distance matching via ultra-fast CPU `POPCNT` (population count) bit operations.
            - Widely used in Visual Odometry, AR/VR tracking on Snapdragon XR platforms, and SLAM.
            """)

    elif selected_module.startswith("10."):
        # Module 10: Geometric Transforms
        st.write("Affine warping, 4-point document perspective rectification, and automated panorama stitching.")
        trans_tab1, trans_tab2, trans_tab3 = st.tabs(["Affine Transformation", "Document Scanner (Perspective)", "Panorama Stitcher"])

        with trans_tab1:
            angle = st.slider("Rotation Angle (degrees)", -180, 180, 25)
            scale = st.slider("Scale Factor", 0.2, 2.0, 1.0, 0.1)
            tx = st.slider("Translate X (pixels)", -100, 100, 0)
            ty = st.slider("Translate Y (pixels)", -100, 100, 0)
            aff_out = affine_transform(current_img, angle=angle, scale=scale, tx=tx, ty=ty)
            st.image(aff_out, caption="Affine Transformed", width="stretch")

        with trans_tab2:
            st.write("Perspective homography rectification (turning a tilted document into a flat scan):")
            h, w = current_img.shape[:2]
            pts = np.array([
                [int(w * 0.2), int(h * 0.15)],
                [int(w * 0.8), int(h * 0.2)],
                [int(w * 0.7), int(h * 0.85)],
                [int(w * 0.15), int(h * 0.8)],
            ], dtype=np.float32)
            warped = four_point_perspective_transform(current_img, pts)
            c1, c2 = st.columns(2)
            with c1:
                annotated = current_img.copy()
                for pt in pts:
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 6, (255, 0, 0), -1)
                cv2.polylines(annotated, [pts.astype(np.int32)], True, (255, 0, 0), 2)
                st.image(annotated, caption="Quadrilateral ROI", width="stretch")
            with c2:
                st.image(warped, caption="Rectified Output", width="stretch")

        with trans_tab3:
            st.write("Stitching two overlapping scenes:")
            p_left = load_image(None, "panorama_left.png")
            p_right = load_image(None, "panorama_right.png")
            st.image([p_left, p_right], caption=["Left Camera Frame", "Right Camera Frame"], width=300)
            if st.button("Stitch Panorama"):
                success, stitched = stitch_panorama([p_left, p_right])
                if success:
                    st.success("Panorama successfully stitched using RANSAC Homography!")
                else:
                    st.warning("Feature alignment fell back to blended composition.")
                st.image(stitched, caption="Composite Panorama", width="stretch")

    elif selected_module.startswith("11."):
        # Module 11: Video Processing
        st.write("Temporal video analysis: frame differencing, background subtraction, and dense optical flow.")
        # Synthesize a moving sequence from current image
        h, w = current_img.shape[:2]
        gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)

        # Generate 5 consecutive synthetic frames simulating camera panning
        frames = []
        for shift_x in [0, 8, 16, 24, 32]:
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            shifted = cv2.warpAffine(gray, M, (w, h))
            frames.append(shifted)

        v_mode = st.radio("Temporal Technique", ["Dense Optical Flow (Farnebäck)", "Background Subtraction (MOG2)", "Motion Heatmap Accumulator"])
        if v_mode == "Dense Optical Flow (Farnebäck)":
            flow_xy, flow_vis = compute_dense_optical_flow(frames[0], frames[1])
            c1, c2 = st.columns(2)
            with c1:
                st.image(frames[0], caption="Frame t", width="stretch")
            with c2:
                st.image(flow_vis, caption="Motion Vector Field (Farnebäck Color Wheel)", width="stretch")
        elif v_mode == "Background Subtraction (MOG2)":
            bg_masks = apply_background_subtraction(frames, method="MOG2")
            st.image(bg_masks[-1], caption="MOG2 Foreground Mask", width="stretch")
        else:
            heatmap = generate_motion_heatmap(frames)
            st.image(heatmap, caption="Cumulative Temporal Motion Heatmap", width="stretch")

        with st.expander("📖 Algorithmic Deep Dive & Mathematical Foundations"):
            st.markdown("""
            **Optical Flow & Video Stabilization:**
            - **Farnebäck Method:** Approximates pixel neighborhoods by quadratic polynomial expansions, resolving per-pixel displacement vectors $(u, v)$.
            - **EIS (Electronic Image Stabilization):** Mobile camera software tracks optical flow vectors across consecutive sensor frames to estimate 3D gyro camera jitter, warping frames in real-time to generate steady video.
            """)

    elif selected_module.startswith("12."):
        # Module 12: Camera ISP Simulation
        st.write("Complete Hardware Camera Image Signal Processor (ISP) simulation pipeline.")
        bayer_mode = st.selectbox("Sensor CFA Layout", ["RGGB", "BGGR"])
        black_level = st.slider("Black Level Sensor Calibration (BLC)", 0, 64, 16)
        awb_method = st.selectbox("Auto White Balance Mode", ["gray_world", "perfect_reflector"])
        tone_curve = st.selectbox("Tone Mapping Transfer Function", ["sRGB", "gamma_2.2"])
        denoise_enabled = st.checkbox("ISP Post-Denoising (Bilateral)", value=True)

        config = {
            "pattern": bayer_mode,
            "black_level": black_level,
            "awb_method": awb_method,
            "tone_curve": tone_curve,
            "denoise": denoise_enabled,
        }

        # Ensure input is Bayer mosaic
        if current_img.ndim == 3:
            bayer_input = simulate_bayer_cfa(current_img, pattern=bayer_mode)
        else:
            bayer_input = current_img

        stages = run_full_isp_pipeline(bayer_input, config=config)

        st.subheader("Step-by-Step Hardware ISP Execution Stages")
        cols = st.columns(3)
        with cols[0]:
            st.image(stages["bayer"], caption="1. Raw Bayer CFA Pattern", width="stretch")
        with cols[1]:
            st.image(stages["demosaiced"], caption="2. Debayer / Demosaic", width="stretch")
        with cols[2]:
            st.image(stages["bls"], caption="3. Black Level Corrected", width="stretch")

        cols2 = st.columns(3)
        with cols2[0]:
            st.image(stages["awb"], caption="4. Auto White Balance (AWB)", width="stretch")
        with cols2[1]:
            st.image(stages["ccm"], caption="5. Color Correction Matrix (CCM)", width="stretch")
        with cols2[2]:
            st.image(stages["final"], caption="6. Tone Mapped & Denoised Output", width="stretch")

        st.download_button(
            "⬇ Download ISP Pipeline Output",
            get_image_download_bytes(stages["final"]),
            "isp_output.png",
            "image/png",
        )

        with st.expander("📖 Hardware Camera ISP Architecture Deep Dive"):
            st.markdown("""
            **Inside a Hardware Camera ISP:**
            1. **CFA Demosaicing:** Color sensors place a Bayer mosaic color filter array (CFA) over photodiodes so each pixel records only one primary color (50% Green, 25% Red, 25% Blue). Debayering reconstructs missing chrominance.
            2. **Black Level Calibration (BLC):** Sensors produce thermal dark-current voltage offsets even when zero light strikes the sensor. BLC subtracts this threshold.
            3. **AWB & CCM:** 3A hardware algorithms balance channel gains, while a $3 \times 3$ Color Correction Matrix aligns raw camera primaries with standard rec.709 / sRGB color standards.
            4. **Tone Mapping:** Gamma curves and local tone mapping compress 12-to-14 bit sensor RAW dynamic range down to 8-bit display standards without blowing out highlights or crushing shadows.
            """)


if __name__ == "__main__":
    main()
