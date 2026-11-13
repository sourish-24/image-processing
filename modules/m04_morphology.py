import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, feature, segmentation


def get_structuring_element(shape_name: str, ksize: int) -> np.ndarray:
    shape_name = shape_name.lower()
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(1, ksize)

    if shape_name in ["rect", "rectangle"]:
        shape = cv2.MORPH_RECT
    elif shape_name in ["ellipse", "disk", "circle"]:
        shape = cv2.MORPH_ELLIPSE
    elif shape_name in ["cross", "plus"]:
        shape = cv2.MORPH_CROSS
    else:
        shape = cv2.MORPH_RECT
    return cv2.getStructuringElement(shape, (ksize, ksize))


def morphology_op(img: np.ndarray, op_type: str = "erosion", kernel_shape: str = "rect", ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Performs fundamental morphological operation:
    'erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat'.
    """
    kernel = get_structuring_element(kernel_shape, ksize)
    op_type = op_type.lower()

    if op_type == "erosion":
        return cv2.erode(img, kernel, iterations=iterations)
    elif op_type == "dilation":
        return cv2.dilate(img, kernel, iterations=iterations)
    elif op_type == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_type == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op_type == "gradient":
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif op_type == "tophat":
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif op_type == "blackhat":
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    else:
        raise ValueError(f"Unknown morphological operation: {op_type}")


def extract_boundary(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Extracts internal boundary: Boundary(A) = A - Erosion(A)
    """
    kernel = get_structuring_element("rect", ksize)
    eroded = cv2.erode(img, kernel, iterations=1)
    diff = img.astype(np.int16) - eroded.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)


def skeletonize_binary(bin_img: np.ndarray) -> np.ndarray:
    """
    Computes morphological skeleton (medial axis) of a binary image.
    """
    bool_img = bin_img > 0
    skel = morphology.skeletonize(bool_img)
    return (skel.astype(np.uint8) * 255)


def separate_touching_objects(bin_img: np.ndarray, min_distance: int = 10) -> tuple[np.ndarray, int]:
    """
    Separates touching objects using Distance Transform and Watershed segmentation.
    Returns:
        labels: 2D integer array with distinct object IDs
        count: Number of unique objects detected
    """
    binary = (bin_img > 0).astype(np.uint8)
    distance = ndimage.distance_transform_edt(binary)

    coords = feature.peak_local_max(distance, min_distance=min_distance, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    unique_objects = np.unique(labels)
    count = int(np.count_nonzero(unique_objects > 0))
    return labels, count
