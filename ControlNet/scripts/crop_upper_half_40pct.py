#!/usr/bin/env python3
"""Crop 40% of images to keep only the upper half.

Default target: scripts/data/1_dashcam
- Selects a random subset (reproducible via --seed)
- Crops selected images in-place: keeps the upper half (top 50% of pixels)
- Leaves non-image files (e.g. matching .txt captions) untouched

Example:
  python3 scripts/crop_upper_half_40pct.py --dir scripts/data/1_dashcam --fraction 0.4 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageFile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _crop_upper_half_inplace(path: Path) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    with Image.open(path) as img:
        width, height = img.size
        new_height = max(1, height // 2)

        exif_bytes = img.info.get("exif")
        icc_profile = img.info.get("icc_profile")

        cropped = img.crop((0, 0, width, new_height))

        tmp_path = path.with_name(path.name + ".tmp")
        fmt = (img.format or path.suffix.lstrip(".")).upper()

        save_kwargs: dict = {}
        if exif_bytes is not None:
            save_kwargs["exif"] = exif_bytes
        if icc_profile is not None:
            save_kwargs["icc_profile"] = icc_profile

        if fmt in {"JPG", "JPEG"}:
            save_kwargs.update({"quality": 95, "optimize": True})
        elif fmt == "PNG":
            save_kwargs.update({"optimize": True})

        cropped.save(tmp_path, format=fmt, **save_kwargs)
        tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop a fraction of images to their upper half (in-place).")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("scripts/data/1_dashcam"),
        help="Directory containing images (default: scripts/data/1_dashcam)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.4,
        help="Fraction of images to crop (0..1, default: 0.4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for selecting images (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be cropped; do not modify files.",
    )
    args = parser.parse_args()

    if not args.dir.exists() or not args.dir.is_dir():
        raise SystemExit(f"Directory not found: {args.dir}")

    if not (0.0 <= args.fraction <= 1.0):
        raise SystemExit("--fraction must be between 0 and 1")

    images = sorted([p for p in args.dir.iterdir() if _is_image(p)])
    if not images:
        print(f"No images found in: {args.dir}")
        return 0

    rng = random.Random(args.seed)
    rng.shuffle(images)

    count = int(math.floor(len(images) * args.fraction + 1e-9))
    if args.fraction > 0 and count == 0:
        count = 1

    selected = images[:count]

    print(f"Found {len(images)} images in {args.dir}")
    print(f"Cropping {len(selected)} images ({args.fraction * 100:.1f}%) to upper half")

    failures = 0
    for p in selected:
        if args.dry_run:
            print(f"DRY-RUN: would crop {p.name}")
            continue

        try:
            _crop_upper_half_inplace(p)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAILED: {p.name}: {e}")

    if not args.dry_run:
        print(f"Done. Failures: {failures}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
