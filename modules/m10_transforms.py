import cv2
import numpy as np


def affine_transform(img: np.ndarray, angle: float = 0.0, scale: float = 1.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """
    Applies rigid/similarity affine transformation (rotation around center, scaling, translation).
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in consistent top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest x + y
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest x + y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest y - x
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest y - x
    return rect


def four_point_perspective_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Warps a 4-point quadrilateral region into a rectified top-down rectangular view (document scanner).
    """
    rect = order_quad_points(pts.astype(np.float32))
    tl, tr, br, bl = rect

    # Calculate width of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Calculate height of new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def stitch_panorama(images: list[np.ndarray]) -> tuple[bool, np.ndarray]:
    """
    Stitches multiple overlapping images into a seamless panorama using OpenCV Stitcher.
    Returns:
        (success_bool, stitched_image)
    """
    if len(images) < 2:
        return False, images[0] if images else np.zeros((10, 10, 3), dtype=np.uint8)

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return True, stitched
    else:
        # Fallback: simple horizontal concatenation
        h = min(im.shape[0] for im in images)
        resized = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images]
        fallback = np.hstack(resized)
        return False, fallback
