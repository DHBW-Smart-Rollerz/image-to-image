import cv2
import random
import numpy as np
from pathlib import Path

INPUT_DIR = Path("scripts/data/1_dashcam")
TARGET_W, TARGET_H = 512, 512
# Wahrscheinlichkeit, ein Bild zu bearbeiten (60-70% => 0.65)
PROCESS_PROB = 1.0

# Masken-Farbe (dunkelgrau) im OpenCV-BGR-Format
MASK_COLOR_BGR = (64, 64, 64)
# Weiche Kante: 2–3 px Gaussian Blur am Rand (Sigma in Pixeln)
EDGE_BLUR_SIGMA = 2.5

# Beispiel-Autobox (x1, y1, x2, y2) — Koordinaten im skalierten 512x512 Bild
x1, y1, x2, y2 = 110, 215, 380, 512

def process_image(path: Path, do_blur: bool = False) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Skipping (can't read): {path}")
        return False

    img_resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    if do_blur:
        # Statt Blur: dunkelgrau maskieren, aber mit leicht weichgezeichnetem Rand
        # (damit keine harten Kanten/Pixelbrüche als Features gelernt werden).
        if not (0 <= x1 < x2 <= TARGET_W and 0 <= y1 < y2 <= TARGET_H):
            print(f"Skipping mask (ROI out of bounds): {path}")
        else:
            mask = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            # Weiche Kante: GaussianBlur auf der Maske (kleines Sigma ~2–3 px)
            # Kernel (0,0) => OpenCV wählt passend zu Sigma.
            if EDGE_BLUR_SIGMA and EDGE_BLUR_SIGMA > 0:
                mask = cv2.GaussianBlur(mask, (0, 0), EDGE_BLUR_SIGMA)

            mask = np.clip(mask, 0.0, 1.0)
            mask3 = np.repeat(mask[:, :, None], 3, axis=2)

            base = img_resized.astype(np.float32)
            color = np.full((TARGET_H, TARGET_W, 3), MASK_COLOR_BGR, dtype=np.float32)

            out = base * (1.0 - mask3) + color * mask3
            img_resized = out.clip(0, 255).astype("uint8")

    success = cv2.imwrite(str(path), img_resized)
    if not success:
        print(f"Failed to write: {path}")
    return success


def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input directory not found: {INPUT_DIR}")

    patterns = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for p in patterns:
        files.extend(INPUT_DIR.glob(p))

    files = sorted(files)
    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    random.seed()
    for f in files:
        do_blur = random.random() < PROCESS_PROB
        if do_blur:
            print(f"Processing (resize+mask): {f}")
        else:
            print(f"Processing (resize only): {f}")
        process_image(f, do_blur=do_blur)


if __name__ == "__main__":
    main()
