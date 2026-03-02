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
print(f"[debug] torch.__version__={torch.__version__}", flush=True)
print(f"[debug] torch.version.cuda={torch.version.cuda}", flush=True)
print(f"[debug] cuda.is_available={torch.cuda.is_available()}", flush=True)
print(f"[debug] cuda.device_count={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    try:
        print(f"[debug] cuda.current_device={torch.cuda.current_device()}", flush=True)
        print(f"[debug] cuda.device_name={torch.cuda.get_device_name(0)}", flush=True)
    except Exception as e:
        print(f"[debug] cuda.device_query_failed={e!r}", flush=True)
print(f"[debug] using DEVICE={DEVICE}", flush=True)

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
LORA_ROAD_PATH = "output_lora/sim2real_dashcam.safetensors"
LORA_CAR_PATH = "output_lora/sim2real_car.safetensors"  # LoRA für Auto-Generierung

# Auto Bounding Box
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

def create_road_mask(size=(512, 512)):
    """
    Weiß = wird generiert (Straße)
    Schwarz = bleibt unverändert (Auto)
    """
    mask = np.ones((size[1], size[0]), dtype=np.uint8) * 255  # alles weiß

    # Auto schwarz maskieren (bleibt unverändert)
    mask[y1:y2, x1:x2] = 0

    # weiche Kanten (wichtig!)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    return Image.fromarray(mask)

def create_car_mask(size=(512, 512)):
    """
    Weiß = wird generiert (Auto)
    Schwarz = bleibt unverändert (Rest)
    """
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)  # alles schwarz

    # Auto-Bereich weiß markieren (wird generiert)
    mask[y1:y2, x1:x2] = 255

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
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Schritt 1: Straße generieren (Auto-Bereich bleibt unverändert)
    print("  → Schritt 1: Generiere Straße...")
    pipe.load_lora_weights(LORA_ROAD_PATH)
    pipe.fuse_lora()
    
    road_mask = create_road_mask(init_image.size)
    
    road_result = pipe(
        prompt=(
            "realistic dashcam photo, natural lighting, realistic asphalt texture, "
            "road surface details, soft shadows, realistic reflections, high dynamic range"
        ),
        negative_prompt="car details, vehicle focus, flat lighting, overexposed, underexposed",
        image=init_image,
        mask_image=road_mask, 
        control_image=control_image,
        strength=0.85,
        guidance_scale=6.5,
        controlnet_conditioning_scale=0.9,
        num_inference_steps=40,
        generator=generator,
    ).images[0]

    # LoRA entfernen für nächste Generierung
    pipe.unfuse_lora()
    pipe.unload_lora_weights()

    # Schritt 2: Auto generieren (auf dem Ergebnis von Schritt 1)
    print("  → Schritt 2: Generiere Auto...")
    pipe.load_lora_weights(LORA_CAR_PATH)
    pipe.fuse_lora()
    
    car_mask = create_car_mask(init_image.size)
    control_image_car = compute_canny(road_result)  # Canny vom Zwischenergebnis
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 1)  # Anderer Seed für Auto
    
    final_result = pipe(
        prompt=(
            "realistic car from behind, detailed vehicle, natural lighting, "
            "car on road, dashcam perspective, photorealistic automobile"
        ),
        negative_prompt="distorted vehicle, unrealistic car, flat lighting, low quality",
        image=road_result,  # Verwende das Straßen-Ergebnis als Basis
        mask_image=car_mask, 
        control_image=control_image_car,
        strength=0.85,
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.85,
        num_inference_steps=40,
        generator=generator,
    ).images[0]

    # LoRA entfernen
    pipe.unfuse_lora()
    pipe.unload_lora_weights()

    final_result.save(out_path)
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
