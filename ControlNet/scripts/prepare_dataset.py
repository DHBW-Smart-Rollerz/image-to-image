import argparse
from pathlib import Path
from PIL import Image
import random
import shutil

from generate_control_maps import make_canny_control


def prepare(sim_dir: Path, out_dir: Path, val_ratio: float = 0.1, size=(512, 512)):
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_control = out_dir / "sim_control"
    sim_images = out_dir / "sim_images"
    sim_control.mkdir(exist_ok=True)
    sim_images.mkdir(exist_ok=True)

    imgs = sorted([p for p in sim_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not imgs:
        print("Keine Sim-Bilder gefunden in:", sim_dir)
        return

    # shuffle for split reproducibility
    random.seed(42)
    random.shuffle(imgs)
    val_count = max(1, int(len(imgs) * val_ratio))
    val_set = set(imgs[:val_count])

    train_txt = out_dir / "train_pairs.txt"
    val_txt = out_dir / "val_pairs.txt"

    with train_txt.open("w") as ftr, val_txt.open("w") as fval:
        for p in imgs:
            im = Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
            control = make_canny_control(im)

            # save standardized image + control map
            dest_img = sim_images / p.name
            dest_control = sim_control / p.name
            im.save(dest_img)
            control.save(dest_control)

            line = f"{dest_img.relative_to(out_dir.parent)},{dest_control.relative_to(out_dir.parent)}\n"
            if p in val_set:
                fval.write(line)
            else:
                ftr.write(line)

    print("Prepared dataset:")
    print(" images:", sim_images)
    print(" controls:", sim_control)
    print(" train pairs:", train_txt)
    print(" val pairs:", val_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=Path, default=Path("ControlNet/data/sim"))
    parser.add_argument("--out_dir", type=Path, default=Path("ControlNet/data/prepared"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    prepare(args.sim_dir, args.out_dir, args.val_ratio)
