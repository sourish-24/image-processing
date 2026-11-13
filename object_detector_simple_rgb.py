import os
import argparse
import cv2
import numpy as np
from skimage import measure


def detect_objects(image_path, out_path="output_images/rgb/detected_rgb.png", min_area=150, invert=None):
    """
    Detects and localizes objects directly from an RGB image using
    Gaussian blur, Otsu thresholding, morphological filtering, and connected components.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from '{image_path}'.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Determine threshold polarity
    if invert is None:
        # Auto-detect background brightness from perimeter pixels
        border_pixels = np.concatenate([blur[0, :], blur[-1, :], blur[:, 0], blur[:, -1]])
        thresh_type = cv2.THRESH_BINARY_INV if np.median(border_pixels) > 127 else cv2.THRESH_BINARY
    elif invert:
        thresh_type = cv2.THRESH_BINARY_INV
    else:
        thresh_type = cv2.THRESH_BINARY

    _, th = cv2.threshold(blur, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # Morphological cleaning (remove noise specks, then fill small holes)
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Label connected components (8-connectivity)
    labels = measure.label(th, connectivity=2)
    props = measure.regionprops(labels)

    boxes = []
    for p in props:
        if p.area < min_area:
            continue
        minr, minc, maxr, maxc = p.bbox
        # Filter out accidental full-screen bounding box
        if (maxc - minc) > img.shape[1] * 0.98 and (maxr - minr) > img.shape[0] * 0.98:
            continue
        boxes.append((minc, minr, maxc - minc, maxr - minr))
        cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 0, 255), 2)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(out_path, img)
    if not success:
        raise IOError(f"Failed to write output image to '{out_path}'.")

    print(f"Detected {len(boxes)} object(s).")
    print(f"Saved output to: {out_path}")
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Detect objects from RGB image using Otsu thresholding & morphology.")
    parser.add_argument("--input", "-i", required=True, help="Path to input RGB image.")
    parser.add_argument(
        "--out",
        "-o",
        default="output_images/rgb/detected_rgb.png",
        help="Path to save annotated output image (default: output_images/rgb/detected_rgb.png).",
    )
    parser.add_argument("--min-area", type=int, default=150, help="Minimum object area in pixels (default: 150).")
    parser.add_argument("--invert", dest="invert", action="store_true", help="Force invert threshold (light background).")
    parser.add_argument("--no-invert", dest="invert", action="store_false", help="Force direct threshold (dark background).")
    parser.set_defaults(invert=None)
    args = parser.parse_args()

    detect_objects(args.input, args.out, min_area=args.min_area, invert=args.invert)


if __name__ == "__main__":
    main()
