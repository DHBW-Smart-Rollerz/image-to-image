import cv2
import random
from pathlib import Path

INPUT_DIR = Path("scripts/data/1_dashcam")
TARGET_W, TARGET_H = 512, 512
# Wahrscheinlichkeit, ein Bild zu bearbeiten (60-70% => 0.65)
PROCESS_PROB = 0.65

# Beispiel-Autobox (x1, y1, x2, y2) — Koordinaten im skalierten 512x512 Bild
x1, y1, x2, y2 = 120, 280, 370, 512

def process_image(path: Path, do_blur: bool = False) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Skipping (can't read): {path}")
        return False

    img_resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    if do_blur:
        roi = img_resized[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"Skipping blur (empty ROI): {path}")
            # still write resized image
            success = cv2.imwrite(str(path), img_resized)
            if not success:
                print(f"Failed to write: {path}")
            return success

        blurred = cv2.GaussianBlur(roi, (51, 51), 0)
        img_resized[y1:y2, x1:x2] = blurred

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
            print(f"Processing (resize+blur): {f}")
        else:
            print(f"Processing (resize only): {f}")
        process_image(f, do_blur=do_blur)


if __name__ == "__main__":
    main()
