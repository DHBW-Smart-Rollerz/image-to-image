#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys


def safe_rename_sequential(folder, files, start, digits, keep_ext, dry_run):
    targets = []
    for i, p in enumerate(files, start):
        ext = p.suffix.lower() if keep_ext else '.jpg'
        targets.append(folder / (f"{i:0{digits}d}" + ext))

    # Check for existing conflicting targets not in sources
    src_names = {p.name for p in files}
    conflicts = [t for t in targets if t.exists() and t.name not in src_names]
    if conflicts:
        print("Aborting: target filenames already exist:")
        for c in conflicts:
            print(" -", c)
        return False

    if dry_run:
        for src, tgt in zip(files, targets):
            print(f"Would rename: {src.name} -> {tgt.name}")
        print(f"Would write captions like: {targets[0].with_suffix('.txt').name} ...")
        return True

    # Use temporary names to avoid collisions during rename
    temp_map = []  # list of (original, temp_path)
    try:
        for i, p in enumerate(files):
            temp = folder / (f".rename_tmp_{i}_{p.name}")
            p.rename(temp)
            temp_map.append((p, temp))

        # rename temps to final targets
        for (_, temp), tgt in zip(temp_map, targets):
            temp.rename(tgt)

    except Exception as ex:
        print("Error during renaming:", ex)
        # rollback: try to restore originals
        for orig, temp in temp_map:
            try:
                if temp.exists():
                    temp.rename(orig)
                else:
                    # maybe already moved to tgt; try to find by index
                    possible = folder / orig.name
                    if possible.exists():
                        possible.rename(orig)
            except Exception:
                pass
        print("Attempted rollback. Some files may need manual fixing.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Create a caption file for every image in a folder")
    parser.add_argument("--dir", "-d", default="scripts/data/1_dashcam", help="Path to images folder")
    parser.add_argument("--ext", "-e", nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"],
                        help="Image file extensions to consider")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without writing files")
    parser.add_argument("--sequential", "-s", action="store_true",
                        help="Rename images sequentially to 0001.jpg, 0002.jpg etc and create matching .txt captions")
    parser.add_argument("--start", type=int, default=1, help="Starting index for sequential names")
    parser.add_argument("--digits", type=int, default=4, help="Zero-padding digits for sequential names")
    parser.add_argument("--keep-ext", action="store_true",
                        help="Keep original file extensions when renaming (otherwise .jpg is used)")
    parser.add_argument("--caption", "-c", required=True,
                        help="Caption text written into each generated .txt file")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return

    exts = {e.lower() for e in args.ext}
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        print(f"No images found in {folder} for extensions: {', '.join(sorted(exts))}")
        return

    if args.sequential:
        ok = safe_rename_sequential(folder, files, args.start, args.digits, args.keep_ext, args.dry_run)
        if not ok:
            return

        # recompute file list as the sequentially named files
        files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts or p.suffix.lower()=='.jpg'])
        # build ordered list by numeric name
        def key_num(p):
            name = p.stem
            try:
                return int(name)
            except Exception:
                return 10**9

        files = sorted(files, key=key_num)

        # write captions matching sequential numbers
        created = 0
        for i, p in enumerate(files, args.start):
            caption_path = folder / (f"{i:0{args.digits}d}.txt")
            if args.dry_run:
                print(f"Would write caption: {caption_path}")
            else:
                caption_path.write_text(CAPTION + "\n", encoding='utf-8')
                created += 1

        if args.dry_run:
            print(f"Dry run: {len(files)} caption files would be created.")
        else:
            print(f"Wrote {created} caption files in {folder}")

    else:
        created = 0
        for p in files:
            caption_path = p.with_suffix('.txt')
            if args.dry_run:
                print(f"Would write caption for: {p} -> {caption_path}")
            else:
                caption_path.write_text(CAPTION + "\n", encoding='utf-8')
                created += 1

        if args.dry_run:
            print(f"Dry run: {len(files)} caption files would be created.")
        else:
            print(f"Wrote {created} caption files in {folder}")


if __name__ == '__main__':
    main()
