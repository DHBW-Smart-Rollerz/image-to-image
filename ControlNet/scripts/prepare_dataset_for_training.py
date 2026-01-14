#!/usr/bin/env python3
"""Kurzskript zur Vorbereitung von Bilddatensätzen für Training.

Funktionen:
- Inventar (Anzahl, Größe)
- Prüfen und Entfernen korrupten Bilder
- Konvertieren nach RGB
- Resize + Pad auf Quadrat-Target
- Split in train/val/test
- Manifest (CSV) + Checksummen

Usage:
  python prepare_dataset_for_training.py \
    --src /path/to/raw_images --dst /path/to/prepared --size 512 \
    --split 0.8 0.1 0.1
"""

import argparse
import csv
import hashlib
import os
import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def is_image(path: Path):
    return path.suffix.lower() in IMG_EXTS


def sha256_of_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    return h.hexdigest()


def resize_and_pad(im: Image.Image, target: int):
    w, h = im.size
    scale = target / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    im = im.resize((new_w, new_h), Image.LANCZOS)
    new_im = Image.new('RGB', (target, target), (0, 0, 0))
    paste_x = (target - new_w) // 2
    paste_y = (target - new_h) // 2
    new_im.paste(im, (paste_x, paste_y))
    return new_im


def prepare(args):
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob('*') if p.is_file() and is_image(p)]
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"Found {len(files)} images, total {total_bytes / (1024**3):.2f} GB")

    processed = []
    for p in tqdm(files, desc='Processing'):
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                ow, oh = im.size
                if ow < args.min_size or oh < args.min_size:
                    continue
                out_im = resize_and_pad(im, args.size)
                # save to temp path under dst/tmp to compute checksum
                rel = p.relative_to(src)
                out_name = rel.with_suffix('.jpg').name
                out_sub = dst / 'all'
                out_sub.mkdir(parents=True, exist_ok=True)
                out_path = out_sub / out_name
                out_im.save(out_path, format='JPEG', quality=args.quality)
                h = sha256_of_file(out_path)
                processed.append({
                    'src': str(p),
                    'dst': str(out_path),
                    'sha256': h,
                    'orig_w': ow,
                    'orig_h': oh,
                    'w': args.size,
                    'h': args.size,
                })
        except UnidentifiedImageError:
            continue
        except Exception:
            continue

    if not processed:
        print('No images processed. Exiting.')
        return

    # split
    random.seed(args.seed)
    random.shuffle(processed)
    n = len(processed)
    t, v, te = args.split
    i1 = int(n * t)
    i2 = i1 + int(n * v)
    splits = [('train', processed[:i1]), ('val', processed[i1:i2]), ('test', processed[i2:])]

    manifest_path = dst / args.manifest
    with manifest_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['split', 'src', 'dst', 'sha256', 'orig_w', 'orig_h', 'w', 'h'])
        writer.writeheader()
        for split_name, items in splits:
            split_dir = dst / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                src_p = Path(it['dst'])
                dst_p = split_dir / src_p.name
                if not dst_p.exists():
                    src_p.replace(dst_p)
                it_row = it.copy()
                it_row['split'] = split_name
                it_row['dst'] = str(dst_p)
                writer.writerow(it_row)

    # write checksums
    checksums = dst / args.checksums
    with checksums.open('w', encoding='utf-8') as f:
        for it in processed:
            f.write(f"{it['sha256']}  {it['dst']}\n")

    print(f"Processed {n} images -> manifest at {manifest_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--size', type=int, default=512, help='Target size (square)')
    p.add_argument('--quality', type=int, default=95)
    p.add_argument('--min-size', type=int, default=32, help='Ignore very small images')
    p.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='train val test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--manifest', default='manifest.csv')
    p.add_argument('--checksums', default='checksums.sha256')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    prepare(args)
