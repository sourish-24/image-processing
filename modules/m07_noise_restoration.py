import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import mean_squared_error as mse_fn


def add_noise(img: np.ndarray, noise_type: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Corrupts image with realistic camera and sensor noise models:
    - 'gaussian': Additive Gaussian noise (std, mean)
    - 'salt_pepper': Impulse noise (amount, salt_vs_pepper)
    - 'speckle': Multiplicative noise (std)
    - 'poisson': Shot noise
    """
    out = img.copy().astype(np.float32)
    noise_type = noise_type.lower()

    if noise_type == "gaussian":
        std = kwargs.get("std", 20.0)
        mean = kwargs.get("mean", 0.0)
        noise = np.random.normal(mean, std, img.shape)
        out += noise
    elif noise_type in ["salt_pepper", "s&p"]:
        amount = kwargs.get("amount", 0.04)
        s_vs_p = kwargs.get("s_vs_p", 0.5)
        # Salt
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
        out[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
        out[tuple(coords)] = 0
    elif noise_type == "speckle":
        std = kwargs.get("std", 0.2)
        noise = np.random.normal(0, std, img.shape)
        out = out + out * noise
    elif noise_type == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(out * vals) / float(vals)
        out = noisy
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return np.clip(out, 0, 255).astype(np.uint8)


def restore_image(img: np.ndarray, method: str = "median", **kwargs) -> np.ndarray:
    """
    Restores noisy image using specified spatial or edge-preserving filtering:
    - 'median': Median filter (ksize=3 or 5)
    - 'bilateral': Bilateral edge-preserving filter
    - 'nlm': Non-Local Means denoising
    - 'gaussian': Gaussian smoothing
    """
    method = method.lower()
    ksize = kwargs.get("ksize", 3)
    if ksize % 2 == 0:
        ksize += 1

    if method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "bilateral":
        d = kwargs.get("d", 9)
        sigma_color = kwargs.get("sigma_color", 75)
        sigma_space = kwargs.get("sigma_space", 75)
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    elif method in ["nlm", "non_local_means"]:
        h = kwargs.get("h", 10.0)
        template_window = kwargs.get("template_window", 7)
        search_window = kwargs.get("search_window", 21)
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(img, None, h, template_window, search_window)
        else:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 1.2)
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
    else:
        raise ValueError(f"Unsupported restoration method: {method}")


def compute_quality_metrics(original: np.ndarray, restored: np.ndarray) -> dict:
    """
    Computes objective image quality assessment (IQA) metrics:
    - Peak Signal-to-Noise Ratio (PSNR in dB)
    - Structural Similarity Index Measure (SSIM in [-1, 1])
    - Mean Squared Error (MSE)
    """
    if original.shape != restored.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs restored {restored.shape}")

    mse_val = float(mse_fn(original, restored))
    psnr_val = float(psnr_fn(original, restored, data_range=255))

    channel_axis = 2 if original.ndim == 3 else None
    ssim_val = float(ssim_fn(original, restored, data_range=255, channel_axis=channel_axis))

    return {
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "mse": round(mse_val, 2),
    }
