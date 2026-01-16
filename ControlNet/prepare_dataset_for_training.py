
#!/usr/bin/env python3
"""
Minimal tool to generate Canny control images for sim images,
pair each sim-control with exactly one real image from the same
category and write a single `metadata.jsonl` with repo-relative paths.

Output JSONL lines: {"image": "path/to/data/real/..png", "control_image": "path/to/data/sim/..._canny.png", "prompt": "... category: <cat>"}

Categories (fixed): curve, intersection, lane_change, left_lane, straight
"""

import argparse
import json
import os
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def is_image(path: Path):
    return path.suffix.lower() in IMG_EXTS


def generate_canny(in_path: Path, out_path: Path, thr1: int = 100, thr2: int = 200):
    if cv2 is None:
        raise RuntimeError('OpenCV (cv2) and numpy are required for Canny generation. Install with `pip install opencv-python numpy`')
    # read with imdecode to support unicode paths on Windows
    arr = np.fromfile(str(in_path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    edges = cv2.Canny(img, thr1, thr2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success, buf = cv2.imencode('.png', edges)
    if not success:
        return False
    buf.tofile(str(out_path))
    return True


def resize_and_pad_image(in_path: Path, out_path: Path, size: int):
    try:
        with Image.open(in_path) as im:
            # convert to RGB for real images, keep L for canny
            mode = 'RGB'
            if im.mode == 'L' or im.mode == '1':
                mode = 'L'
            if mode == 'RGB':
                im = im.convert('RGB')
            else:
                im = im.convert('L')

            w, h = im.size
            scale = size / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

            if mode == 'RGB':
                new_im = Image.new('RGB', (size, size), (0, 0, 0))
            else:
                new_im = Image.new('L', (size, size), 0)

            paste_x = (size - new_w) // 2
            paste_y = (size - new_h) // 2
            new_im.paste(im, (paste_x, paste_y))
            # ensure parent exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # save as PNG to preserve exact content
            new_im.save(out_path, format='PNG')
            return True
    except Exception:
        return False


def build_prompt(category: str) -> str:
    lines = [
        'realistic indoor autonomous driving test track,',
        'front-facing low-mounted camera view from a model car,',
        'black asphalt road with white lane markings,',
        'miniature traffic signs, laboratory environment,',
        'technical research setup,',
        'wide-angle lens, slight fisheye distortion,',
        'monochrome image, high contrast,',
        'raw sensor-like appearance,',
        'daytime, high dynamic range lighting'
    ]
    return ' '.join(lines) + f' category: {category}'


def main(args):
    data_root = Path(args.src)
    sim_root = data_root / 'sim'
    real_root = data_root / 'real'

    if not sim_root.exists() or not real_root.exists():
        raise SystemExit(f'Expected folders not found: {sim_root} or {real_root}')

    categories = ['curve', 'intersection', 'lane_change', 'left_lane', 'straight']
    random.seed(args.seed)
    entries = []
    repo_root = Path.cwd()

    for cat in categories:
        sim_dir = sim_root / cat
        real_dir = real_root / cat
        if not sim_dir.exists() or not real_dir.exists():
            # skip missing categories silently
            continue

        sim_files = sorted([p for p in sim_dir.iterdir() if p.is_file() and is_image(p)])
        real_files = sorted([p for p in real_dir.iterdir() if p.is_file() and is_image(p)])
        if not sim_files or not real_files:
            continue

        for sim_f in tqdm(sim_files, desc=f'cat:{cat}'):
            canny_path = sim_f.with_name(sim_f.stem + '_canny.png')
            if not canny_path.exists() or args.overwrite:
                ok = generate_canny(sim_f, canny_path, args.canny_thr1, args.canny_thr2)
                if not ok:
                    print(f'Warning: cannot generate Canny for {sim_f}; skipping')
                    continue

            # resize/ pad canny to target size (overwrite canny file)
            ok2 = resize_and_pad_image(canny_path, canny_path, args.size)
            if not ok2:
                print(f'Warning: cannot resize canny for {canny_path}; skipping')
                continue

            # resize real image to <original>_resized.png next to original
            real_choice = random.choice(real_files)
            real_resized = real_choice.with_name(real_choice.stem + '_resized.png')
            if not real_resized.exists() or args.overwrite:
                ok3 = resize_and_pad_image(real_choice, real_resized, args.size)
                if not ok3:
                    print(f'Warning: cannot resize real image {real_choice}; skipping')
                    continue

            rel_real = Path(os.path.relpath(real_resized, repo_root)).as_posix()
            rel_canny = Path(os.path.relpath(canny_path, repo_root)).as_posix()
            prompt = build_prompt(cat)
            entries.append({'image': rel_real, 'control_image': rel_canny, 'prompt': prompt})

    out_dir = Path(args.dst)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'metadata.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for e in entries:
            json.dump(e, f, ensure_ascii=False)
            f.write('\n')

    print(f'Wrote {len(entries)} entries to {jsonl_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Path to data root (contains sim/ and real/)')
    p.add_argument('--dst', default='.', help='Output folder for metadata.jsonl')
    p.add_argument('--size', type=int, default=512, help='Target square size for resize/pad')
    p.add_argument('--canny-thr1', type=int, default=100, help='Canny threshold1')
    p.add_argument('--canny-thr2', type=int, default=200, help='Canny threshold2')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing _canny.png files')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
