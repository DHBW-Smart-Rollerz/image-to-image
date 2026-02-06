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
x1, y1, x2, y2 = 50, 230, 460, 512

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

def compute_spec_mask(image: Image.Image, percentile=98, min_thresh=180):
    """
    Erzeuge eine Specular-Maske basierend auf Helligkeit (V-Kanal).
    Rückgabe: RGB-PIL-Image (weiß = stark specular, weichgefiltert)
    """
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]

    p = np.percentile(v, percentile)
    thresh = int(max(min_thresh, p))

    mask = (v >= thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    mask3 = np.stack([mask] * 3, axis=-1)
    return Image.fromarray(mask3)


def apply_bloom(image: Image.Image, spec_mask: Image.Image, intensity=1.0, sigma=15):
    """Ein einfacher Bloom-Postprocess, nur als schneller Test."""
    img = np.array(image).astype(np.float32) / 255.0
    mask = np.array(spec_mask.convert("L")).astype(np.float32) / 255.0

    bright = img * mask[..., None]
    # sigma als Gauss-Sigma (cv2 erlaubt 0,0 kernel mit sigma)
    blurred = cv2.GaussianBlur((bright * 255).astype(np.uint8), (0, 0), sigma)
    blurred = blurred.astype(np.float32) / 255.0

    out = np.clip(img + blurred * intensity, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))

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

pipe.load_lora_weights(LORA_PATH, weight=1.2)
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
    spec_mask = compute_spec_mask(init_image)

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
        strength=0.80,
        guidance_scale=4.0,
        controlnet_conditioning_scale=0.55,
        num_inference_steps=30,
        generator=generator,
    )

    out_img = result.images[0]

    # speichere SpecMask für Analyse
    spec_out_path = os.path.splitext(out_path)[0] + "_specmask.png"
    spec_mask.save(spec_out_path)

    # optionaler schneller Bloom-Test (verbessert Wahrnehmung von Reflexionen)
    try:
        bloom_img = apply_bloom(out_img, spec_mask, intensity=1.0, sigma=15)
        bloom_out_path = os.path.splitext(out_path)[0] + "_bloom.png"
        bloom_img.save(bloom_out_path)
    except Exception:
        pass

    out_img.save(out_path)
    print(f"✓ Gespeichert: {out_path} (SpecMask: {spec_out_path})")


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
