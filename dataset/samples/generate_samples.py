import os
import cv2
import numpy as np


def generate_all_samples(out_dir="dataset/samples"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. cameraman.png (512x512 grayscale/RGB scene)
    cameraman = np.zeros((512, 512, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(256):
        cameraman[y, :] = int(180 + (255 - 180) * (y / 256.0))
    # Grass / ground
    for y in range(256, 512):
        cameraman[y, :] = int(60 + (90 - 60) * ((y - 256) / 256.0))
    # Background building
    cv2.rectangle(cameraman, (50, 160), (180, 320), (140, 140, 140), -1)
    for wy in range(180, 300, 30):
        for wx in range(70, 160, 25):
            cv2.rectangle(cameraman, (wx, wy), (wx + 15, wy + 20), (220, 220, 220), -1)
    # Figure coat and head
    cv2.circle(cameraman, (320, 180), 35, (30, 30, 30), -1)  # Head
    cv2.ellipse(cameraman, (320, 300), (60, 100), 0, 0, 360, (20, 20, 20), -1)  # Body/Coat
    # Tripod and camera
    cv2.rectangle(cameraman, (240, 190), (280, 230), (10, 10, 10), -1)  # Camera
    cv2.line(cameraman, (260, 230), (200, 420), (15, 15, 15), 5)  # Left leg
    cv2.line(cameraman, (260, 230), (260, 430), (15, 15, 15), 5)  # Center leg
    cv2.line(cameraman, (260, 230), (310, 420), (15, 15, 15), 5)  # Right leg
    cv2.imwrite(os.path.join(out_dir, "cameraman.png"), cameraman)

    # 2. coins.png (400x400 with isolated and touching coins)
    coins = np.full((400, 400, 3), 40, dtype=np.uint8)
    coin_specs = [
        ((90, 90), 45, (190, 190, 190)),
        ((165, 90), 40, (210, 210, 210)),  # Touching coin 1
        ((280, 100), 50, (180, 180, 180)),
        ((90, 240), 42, (200, 200, 200)),
        ((180, 250), 38, (195, 195, 195)),
        ((250, 250), 38, (205, 205, 205)),  # Touching coin 2
        ((320, 250), 38, (200, 200, 200)),  # Touching coin 3
        ((120, 340), 35, (185, 185, 185)),
        ((280, 340), 45, (190, 190, 190)),
    ]
    for center, radius, color in coin_specs:
        cv2.circle(coins, center, radius, color, -1)
        cv2.circle(coins, center, radius - 6, (color[0] - 30, color[1] - 30, color[2] - 30), 2)
        cv2.circle(coins, center, radius, (80, 80, 80), 2)
    cv2.imwrite(os.path.join(out_dir, "coins.png"), coins)

    # 3. building.png (480x480 architectural scene rich in corners & keypoints)
    bldg = np.full((480, 480, 3), 220, dtype=np.uint8)
    # Sky
    bldg[:150, :] = (240, 200, 160)
    # Main facade
    cv2.rectangle(bldg, (60, 140), (420, 460), (120, 140, 160), -1)
    # Roof pediment (triangle)
    pts = np.array([[60, 140], [240, 40], [420, 140]], np.int32)
    cv2.fillPoly(bldg, [pts], (90, 100, 120))
    # Columns
    for cx in range(90, 400, 50):
        cv2.rectangle(bldg, (cx, 160), (cx + 25, 450), (200, 215, 230), -1)
        cv2.rectangle(bldg, (cx - 4, 150), (cx + 29, 165), (170, 185, 200), -1)
        cv2.rectangle(bldg, (cx - 4, 445), (cx + 29, 460), (170, 185, 200), -1)
    # Door and steps
    cv2.rectangle(bldg, (210, 340), (270, 450), (60, 70, 90), -1)
    cv2.rectangle(bldg, (40, 460), (440, 480), (150, 150, 150), -1)
    cv2.imwrite(os.path.join(out_dir, "building.png"), bldg)

    # 4. noisy_sample.png (corrupted with Gaussian & Salt-Pepper noise)
    base = cameraman.copy()
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 25, base.shape).astype(np.float32)
    noisy = np.clip(base.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)
    # Add Salt & Pepper noise to 3% of pixels
    num_sp = int(0.03 * base.shape[0] * base.shape[1])
    for _ in range(num_sp // 2):
        ry, rx = np.random.randint(0, base.shape[0]), np.random.randint(0, base.shape[1])
        noisy[ry, rx] = 255
    for _ in range(num_sp // 2):
        ry, rx = np.random.randint(0, base.shape[0]), np.random.randint(0, base.shape[1])
        noisy[ry, rx] = 0
    cv2.imwrite(os.path.join(out_dir, "noisy_sample.png"), noisy)

    # 5. text_doc.png (slanted text document with lighting gradient)
    doc = np.full((500, 500, 3), 40, dtype=np.uint8)  # dark desk
    paper = np.full((360, 280), 245, dtype=np.uint8)
    # Add lighting gradient (darker at bottom-right)
    for y in range(360):
        for x in range(280):
            paper[y, x] = max(0, min(255, int(paper[y, x] - (x * 0.2 + y * 0.15))))
    # Draw simulated text lines
    for ty in range(40, 330, 18):
        cv2.line(paper, (30, ty), (250, ty), (30, 30, 30), 2)
    # Draw title and logo block
    cv2.rectangle(paper, (30, 15), (100, 30), (10, 10, 10), -1)
    cv2.rectangle(paper, (180, 15), (250, 32), (100, 100, 100), -1)
    # Warp paper into tilted perspective on desk
    src_pts = np.float32([[0, 0], [280, 0], [280, 360], [0, 360]])
    dst_pts = np.float32([[110, 70], [380, 95], [340, 440], [80, 390]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_paper = cv2.warpPerspective(paper, M, (500, 500))
    mask = cv2.warpPerspective(np.ones_like(paper) * 255, M, (500, 500))
    for c in range(3):
        doc[:, :, c] = np.where(mask > 0, warped_paper, doc[:, :, c])
    cv2.imwrite(os.path.join(out_dir, "text_doc.png"), doc)

    # 6. raw_bayer.png (Simulated Bayer CFA pattern)
    # Generate clean color scene (300x300x3) then sample RGGB
    scene = np.zeros((300, 300, 3), dtype=np.uint8)
    scene[:, :100] = (220, 50, 50)    # Red block
    scene[:, 100:200] = (50, 220, 50)  # Green block
    scene[:, 200:] = (50, 50, 220)    # Blue block
    # Circle in center with white
    cv2.circle(scene, (150, 150), 60, (240, 240, 240), -1)

    # Convert to Bayer CFA (RGGB)
    # R  G
    # G  B
    bayer = np.zeros((300, 300), dtype=np.uint8)
    bayer[0::2, 0::2] = scene[0::2, 0::2, 2]  # R (in BGR, R is idx 2)
    bayer[0::2, 1::2] = scene[0::2, 1::2, 1]  # Gr
    bayer[1::2, 0::2] = scene[1::2, 0::2, 1]  # Gb
    bayer[1::2, 1::2] = scene[1::2, 1::2, 0]  # B (in BGR, B is idx 0)
    cv2.imwrite(os.path.join(out_dir, "raw_bayer.png"), bayer)

    # 7. panorama_left.png & panorama_right.png (Overlapping views of city)
    pano_canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    # Sky
    pano_canvas[:180, :] = (230, 190, 140)
    # Mountains / hills
    cv2.ellipse(pano_canvas, (180, 220), (220, 80), 0, 0, 360, (80, 120, 90), -1)
    cv2.ellipse(pano_canvas, (460, 220), (250, 90), 0, 0, 360, (70, 110, 80), -1)
    # City buildings with distinct windows
    for bx in range(40, 600, 45):
        h = 100 + int(60 * np.sin(bx * 0.05) ** 2)
        cv2.rectangle(pano_canvas, (bx, 360 - h), (bx + 35, 360), (120, 130, 140), -1)
        for wy in range(360 - h + 15, 350, 20):
            for wx in range(bx + 6, bx + 28, 12):
                cv2.rectangle(pano_canvas, (wx, wy), (wx + 7, wy + 10), (230, 240, 200), -1)

    # Crop left view (0 to 420) and right view (220 to 640) -> 200px overlap
    left_view = pano_canvas[:, :420].copy()
    right_view = pano_canvas[:, 220:].copy()
    cv2.imwrite(os.path.join(out_dir, "panorama_left.png"), left_view)
    cv2.imwrite(os.path.join(out_dir, "panorama_right.png"), right_view)

    print(f"Successfully generated all sample images in '{out_dir}'.")


if __name__ == "__main__":
    generate_all_samples()
