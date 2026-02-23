import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
LORA_PATH = "output_lora/sim2real_dashcam.safetensors"

# Auto Bounding Box (geschützt)
x1, y1, x2, y2 = 0,0,0,0 # 110, 200, 380, 512 #50, 230, 460, 512

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def load_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    return img.resize(size, Image.BILINEAR)

def compute_canny(image: Image.Image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)

def create_mask(size=(512, 512)):
    """
    Weiß = wird generiert (Straße)
    Schwarz = bleibt unverändert (Auto)
    """
    mask = np.ones((size[1], size[0]), dtype=np.uint8) * 255  # alles weiß

    # Auto schwarz maskieren
    mask[y1:y2, x1:x2] = 0

    # weiche Kanten (wichtig!)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    return Image.fromarray(mask)

# ------------------------------------------------------------
# Pipeline laden
# ------------------------------------------------------------

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.load_lora_weights(LORA_PATH, weight=1.6)
pipe.fuse_lora()

pipe = pipe.to(DEVICE)

# ------------------------------------------------------------
# Inferenz
# ------------------------------------------------------------

def sim2real(
    sim_image_path: str,
    out_path: str,
    seed: int = 42,
):
    init_image = load_image(sim_image_path)
    control_image = compute_canny(init_image)
    mask_image = create_mask(init_image.size)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=(
            "realistic dashcam photo, natural lighting, realistic asphalt texture, "
            "road surface details, soft shadows, realistic reflections, high dynamic range"
        ),
        negative_prompt="car details, vehicle focus, flat lighting, overexposed, underexposed",
        image=init_image,
        mask_image=mask_image, 
        control_image=control_image,
        strength=0.85,
        guidance_scale=6.5,
        controlnet_conditioning_scale=0.9,
        num_inference_steps=40,
        generator=generator,
    )

    out_img = result.images[0]

    out_img.save(out_path)
    print(f"✓ Gespeichert: {out_path}")


if __name__ == "__main__":
    input_dir = "inputs"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files.sort()

    if not files:
        print(f"Keine Eingabebilder in '{input_dir}' gefunden.")
    else:
        for f in files:
            out_path = os.path.join(output_dir, os.path.basename(f))
            try:
                print(f"→ Verarbeite: {f}  ->  {out_path}")
                sim2real(sim_image_path=f, out_path=out_path)
            except Exception as e:
                print(f"✗ Fehler beim Verarbeiten von {f}: {e}")
