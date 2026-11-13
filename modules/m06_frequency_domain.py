import cv2
import numpy as np


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def compute_fft2(gray_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D Discrete Fourier Transform shifted so DC is centered.
    Returns:
        dft_shift: Complex 2D array
        magnitude_spectrum: 2D uint8 log-scaled magnitude for visualization
    """
    gray = ensure_grayscale(gray_img).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    # Log scale: 20 * log(1 + |F|)
    log_mag = 20 * np.log(1 + magnitude)
    norm_mag = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX)
    return dft_shift, norm_mag.astype(np.uint8)


def create_filter_mask(shape: tuple, filter_type: str, cutoff: float, order: int = 2) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    cutoff = max(0.1, float(cutoff))

    filter_type = filter_type.lower()
    if filter_type == "ideal_lowpass":
        mask = (d <= cutoff).astype(np.float32)
    elif filter_type == "ideal_highpass":
        mask = (d > cutoff).astype(np.float32)
    elif filter_type == "gaussian_lowpass":
        mask = np.exp(-(d ** 2) / (2 * (cutoff ** 2))).astype(np.float32)
    elif filter_type == "gaussian_highpass":
        mask = (1.0 - np.exp(-(d ** 2) / (2 * (cutoff ** 2)))).astype(np.float32)
    elif filter_type == "butterworth_lowpass":
        mask = (1.0 / (1.0 + (d / cutoff) ** (2 * order))).astype(np.float32)
    elif filter_type == "butterworth_highpass":
        # Avoid division by zero at DC
        d_safe = np.where(d == 0, 1e-6, d)
        mask = (1.0 / (1.0 + (cutoff / d_safe) ** (2 * order))).astype(np.float32)
        mask[crow, ccol] = 0.0
    else:
        mask = np.ones(shape, dtype=np.float32)
    return mask


def frequency_filter(gray_img: np.ndarray, filter_type: str = "gaussian_lowpass", cutoff: float = 30.0, order: int = 2) -> np.ndarray:
    """
    Applies frequency domain filtering using 2D FFT, mask multiplication, and inverse FFT.
    """
    gray = ensure_grayscale(gray_img).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    mask = create_filter_mask(gray.shape, filter_type, cutoff, order)
    filtered_shift = dft_shift * mask

    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_real = np.real(img_back)
    return np.clip(img_real, 0, 255).astype(np.uint8)


def remove_periodic_noise(gray_img: np.ndarray, notch_centers: list[tuple[int, int]], radius: int = 5) -> np.ndarray:
    """
    Notch rejection filter: zeroes out specific frequency peaks in FFT domain.
    """
    gray = ensure_grayscale(gray_img).astype(np.float32)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    y, x = np.ogrid[:rows, :cols]
    mask = np.ones((rows, cols), dtype=np.float32)

    for (ny, nx) in notch_centers:
        dist = np.sqrt((x - nx) ** 2 + (y - ny) ** 2)
        mask[dist <= radius] = 0.0

    filtered_shift = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.clip(np.real(img_back), 0, 255).astype(np.uint8)


def create_hybrid_image(img_low: np.ndarray, img_high: np.ndarray, cutoff_low: float = 20.0, cutoff_high: float = 40.0) -> np.ndarray:
    """
    Creates a hybrid image by combining low frequencies of img_low with high frequencies of img_high.
    """
    gray1 = ensure_grayscale(img_low)
    gray2 = ensure_grayscale(img_high)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    low_pass = frequency_filter(gray1, "gaussian_lowpass", cutoff=cutoff_low)
    high_pass = frequency_filter(gray2, "gaussian_highpass", cutoff=cutoff_high)

    hybrid = low_pass.astype(np.float32) * 0.5 + high_pass.astype(np.float32) * 0.5
    return np.clip(hybrid, 0, 255).astype(np.uint8)
