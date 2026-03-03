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

# Auto-Trapez (Basis: 512x512). Reihenfolge: oben-links, oben-rechts, unten-rechts, unten-links
CAR_TRAPEZOID_512 = np.array([
    [70, 260],
    [442, 260],
    [510, 512],
    [2, 512],
], dtype=np.float32)

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

def get_car_trapezoid(size=(512, 512)):
    """
    Gibt das Auto-Trapez auf die Zielgröße skaliert zurück.
    """
    w, h = size
    scale_x = w / 512.0
    scale_y = h / 512.0
    points = CAR_TRAPEZOID_512.copy()
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y
    return points.astype(np.int32)

def create_road_mask(size=(512, 512)):
    """
    Weiß = wird generiert (Straße)
    Schwarz = bleibt unverändert (Auto)
    """
    mask = np.ones((size[1], size[0]), dtype=np.uint8) * 255  # alles weiß

    # Auto als Trapez schwarz maskieren (bleibt unverändert)
    trapezoid = get_car_trapezoid(size)
    cv2.fillPoly(mask, [trapezoid], 0)

    # weiche Kanten (wichtig!)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    return Image.fromarray(mask)

def create_car_mask(size=(512, 512)):
    """
    Weiß = wird generiert (Auto)
    Schwarz = bleibt unverändert (Rest)
    """
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)  # alles schwarz

    # Auto-Bereich als Trapez weiß markieren (wird generiert)
    trapezoid = get_car_trapezoid(size)
    cv2.fillPoly(mask, [trapezoid], 255)

    # weiche Kanten (wichtig!)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    return Image.fromarray(mask)

def mask_model_car(image: Image.Image):
    """
    Maskiert das Modellauto mit einem schwarzen Trapez,
    um Verwirrung bei der Generierung zu vermeiden.
    """
    img_array = np.array(image)
    # Überschreibe den Auto-Bereich als Trapez mit schwarz
    trapezoid = get_car_trapezoid((image.width, image.height))
    cv2.fillPoly(img_array, [trapezoid], (0, 0, 0))
    return Image.fromarray(img_array)

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
    # Bereite Ausgabeverzeichnisse vor
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    out_dir = os.path.dirname(out_path)
    
    masked_dir = os.path.join(out_dir, "masked")
    canny_dir = os.path.join(out_dir, "canny")
    step1_dir = os.path.join(out_dir, "step1_road")
    
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(canny_dir, exist_ok=True)
    os.makedirs(step1_dir, exist_ok=True)
    
    init_image = load_image(sim_image_path)
    # Maskiere das Modellauto vor der Inferenz
    init_image = mask_model_car(init_image)
    
    # Speichere maskiertes Bild
    masked_path = os.path.join(masked_dir, f"{base_name}_masked.png")
    init_image.save(masked_path)
    print(f"  ✓ Maskiertes Bild: {masked_path}")
    
    control_image = compute_canny(init_image)
    
    # Speichere Canny Edges
    canny_path = os.path.join(canny_dir, f"{base_name}_canny.png")
    control_image.save(canny_path)
    print(f"  ✓ Canny Edges: {canny_path}")
    
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
        num_inference_steps=35,
        generator=generator,
    ).images[0]

    # LoRA entfernen für nächste Generierung
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    
    # Speichere Straßen-Ergebnis
    step1_path = os.path.join(step1_dir, f"{base_name}_step1.png")
    road_result.save(step1_path)
    print(f"  ✓ Schritt 1 (Straße): {step1_path}")

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
        num_inference_steps=35,
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