import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import torch

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

# Ensure local script directory is on sys.path so imports work when run as a script
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from generate_control_maps import make_canny_control


def get_image_paths(input_dir: Path):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    return [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]


def run_inference(input_dir: Path, output_dir: Path, prompt: str, device: str):
    device_torch = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device_torch == "cuda" else torch.float32,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device_torch == "cuda" else torch.float32,
    ).to(device_torch)

    if device_torch == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    paths = get_image_paths(input_dir)
    if not paths:
        print(f"Keine Bilder in {input_dir} gefunden. Bitte Dateien (.png/.jpg) hinzufügen.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        print("Processing:", p.name)
        img = Image.open(p).convert("RGB")
        img = img.resize((512, 512), Image.BILINEAR)
        control = make_canny_control(img)

        result = pipe(
            prompt=prompt,
            image=img,
            control_image=control,
            negative_prompt="",
            num_inference_steps=20,
            strength=0.6,
            guidance_scale=7.0,
            controlnet_conditioning_scale=0.8,
        )

        out = result.images[0]
        out.save(output_dir / p.name)
        print("Saved:", output_dir / p.name)
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, default=Path("ControlNet/inputs"))
    parser.add_argument("--output", "-o", type=Path, default=Path("ControlNet/outputs"))
    parser.add_argument("--prompt", "-p", type=str, default="photorealistic scene, natural lighting, preserve geometry and object shapes")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    run_inference(args.input, args.output, args.prompt, args.device)


if __name__ == "__main__":
    main()
