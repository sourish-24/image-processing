import os
import argparse
import cv2
import numpy as np
from skimage import measure


def detect_from_mask(mask_path, image_path=None, out_path="output_images/simple/detected_mask.png", min_area=20):
    """
    Extracts bounding boxes from a binary segmentation mask using Connected Component Labeling.
    """
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Read mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask image from '{mask_path}'.")

    # Ensure binary format (0 or 1)
    mask_bin = (mask > 0).astype(np.uint8)

    # Connected component labeling (8-connectivity)
    labels = measure.label(mask_bin, connectivity=2)
    props = measure.regionprops(labels)

    # Base image for drawing bounding boxes
    if image_path is None:
        img = cv2.cvtColor(mask_bin * 255, cv2.COLOR_GRAY2BGR)
    else:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from '{image_path}'.")

    boxes = []
    for p in props:
        if p.area < min_area:
            continue
        minr, minc, maxr, maxc = p.bbox
        w = maxc - minc
        h = maxr - minr
        boxes.append((minc, minr, w, h))

        # Draw red bounding box (BGR: (0, 0, 255))
        cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 0, 255), 2)

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(out_path, img)
    if not success:
        raise IOError(f"Failed to write output image to '{out_path}'.")

    print(f"Detected {len(boxes)} object(s).")
    print(f"Saved result to: {out_path}")
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Detect objects from binary mask using Connected Component Labeling.")
    parser.add_argument("--mask", "-m", required=True, help="Path to binary mask image.")
    parser.add_argument("--image", "-i", required=False, default=None, help="Optional path to original RGB image.")
    parser.add_argument(
        "--out",
        "-o",
        default="output_images/simple/detected_mask.png",
        help="Path to save annotated output image (default: output_images/simple/detected_mask.png).",
    )
    parser.add_argument("--min-area", type=int, default=20, help="Minimum connected component area (default: 20).")
    args = parser.parse_args()

    detect_from_mask(args.mask, args.image, args.out, min_area=args.min_area)


if __name__ == "__main__":
    main()
