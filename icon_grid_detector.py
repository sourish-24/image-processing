import os
import argparse
import cv2
import numpy as np


def sort_grid_boxes(boxes):
    """
    Sorts bounding boxes in reading order: top-to-bottom by row, then left-to-right within each row.
    """
    if not boxes:
        return []

    # Sort primarily by y
    boxes_by_y = sorted(boxes, key=lambda b: b[1])
    avg_h = np.mean([b[3] for b in boxes_by_y])
    row_tolerance = avg_h * 0.5

    rows = []
    current_row = [boxes_by_y[0]]

    for box in boxes_by_y[1:]:
        # If the y coordinate is within tolerance of current row, append to current row
        if abs(box[1] - current_row[0][1]) < row_tolerance:
            current_row.append(box)
        else:
            # Sort previous row from left to right
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    # Flatten rows
    return [box for row in rows for box in row]


def detect_icons(
    input_path,
    output_path="output_images/grid/detected_icons.png",
    min_area=500,
    canny_low=50,
    canny_high=150,
):
    """
    Detects individual icons or UI elements using Canny edge detection, dilation,
    contour analysis, and grid sorting.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot load input image from '{input_path}'.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Dilate so contours close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Avoid capturing outer frame or full-image borders
        if w > img.shape[1] * 0.95 or h > img.shape[0] * 0.95:
            continue

        boxes.append((x, y, w, h))

    # Sort in clean row-major order (top-to-bottom, left-to-right)
    boxes = sort_grid_boxes(boxes)

    # Draw bounding boxes and index numbers
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            img,
            str(idx),
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    success = cv2.imwrite(output_path, img)
    if not success:
        raise IOError(f"Failed to write output image to '{output_path}'.")

    print(f"Detected {len(boxes)} icon(s).")
    print(f"Saved result to: {output_path}")
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Detect icons/elements in grid layouts using Canny & contours.")
    parser.add_argument("--input", "-i", required=True, help="Path to input image.")
    parser.add_argument(
        "--out",
        "-o",
        default="output_images/grid/detected_icons.png",
        help="Path to save annotated output image (default: output_images/grid/detected_icons.png).",
    )
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area (default: 500).")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny lower threshold (default: 50).")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny higher threshold (default: 150).")
    args = parser.parse_args()

    detect_icons(args.input, args.out, min_area=args.min_area, canny_low=args.canny_low, canny_high=args.canny_high)


if __name__ == "__main__":
    main()
