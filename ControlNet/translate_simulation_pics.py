import os
from pathlib import Path
from PIL import Image

import cv2
import numpy as np

import torch

from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test: Verwende erst das vortrainierte Modell um zu testen ob Pipeline funktioniert
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_TRAINED = Path(__file__).parent / "model"  # Dein trainiertes Modell
CONTROLNET_MODEL_PRETRAINED = "lllyasviel/sd-controlnet-canny"  # Vortrainiert

# TODO: Wechsle zu CONTROLNET_MODEL_TRAINED wenn Training erfolgreich war
USE_TRAINED_MODEL = False

if USE_TRAINED_MODEL:
    controlnet_path = str(CONTROLNET_MODEL_TRAINED)
    print(f"⚙ Verwende TRAINIERTES Modell: {controlnet_path}")
else:
    controlnet_path = CONTROLNET_MODEL_PRETRAINED
    print(f"⚙ Verwende VORTRAINIERTES Modell: {controlnet_path}")

controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
print(f"✓ ControlNet geladen")

from diffusers import StableDiffusionControlNetPipeline

print(f"Loading Stable Diffusion Pipeline...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
pipe = pipe.to(DEVICE)
pipe.safety_checker = None
print(f"✓ Pipeline geladen auf {DEVICE}\n")

def compute_canny_edges(sim_rgb_path: Path, target_size=(512, 512)) -> Image.Image:
    img = load_image(sim_rgb_path, target_size)
    img_np = np.array(img)
    
    # Konvertiere zu Grayscale für bessere Kantendetektkion
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Erhöhe Kontrast mit CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Gaussian Blur für weniger Rauschen
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Canny mit besseren Schwellwerten für schwache Kanten
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilatation um Kanten stärker zu machen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 3 Kanäle für ControlNet
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def load_image(path: Path, target_size=(512, 512)) -> Image.Image:
  """Lädt ein RGB-Bild und skaliert es auf die Zielauflösung."""
  img = Image.open(path).convert("RGB")
  if target_size is not None:
      img = img.resize(target_size, Image.BILINEAR)
  return img

def build_prompt(base_prompt: str = None) -> str:
    """
    Erzeugt einen Prompt für realistische Fahrzeug-Aufnahmen.
    Optimiert für CLIP Token-Limit (max 77 Tokens).
    """
    if base_prompt:
        return base_prompt
    # Einfacher, effektiver Prompt
    return "realistic detailed photo, vehicle on road, daytime"

@torch.no_grad()
def sim_to_real_single(
    sim_rgb_path: Path,
    out_path: Path,
    base_prompt: str = None,
    num_inference_steps: int = 20,
    strength: float = 0.6,
    guidance_scale: float = 5.0,
    control_scale: float = 1.0,
    use_depth: bool = False,
    seed: int = None,
):
    """
    Wandelt ein Simulationsbild in ein stilisiertes 'realistisches' Bild um und speichert es.
    strength: wie stark das Bild verändert wird (0=kaum, 1=stark).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        init_image = load_image(sim_rgb_path)
        print(f"  ✓ Input-Bild geladen: {init_image.size}")
        
        if use_depth:
            control_image = load_or_compute_depth_map(sim_rgb_path)
        else:
            control_image = compute_canny_edges(sim_rgb_path)
        print(f"  ✓ Control-Bild erzeugt: {control_image.size}")

        prompt = build_prompt(base_prompt)
        print(f"  ✓ Prompt: '{prompt}'")
        print(f"  ℹ Parameter: steps={num_inference_steps}, strength={strength}, guidance={guidance_scale}, control={control_scale}")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Standard ControlNet Pipeline (kein img2img)
        result = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=control_scale,
            generator=generator,
            negative_prompt="black, dark, blurry, low quality, distorted",
        )

        gen_image = result.images[0]
        
        # Check auf NaN-Werte
        img_array = np.array(gen_image)
        if np.isnan(img_array).any():
            print(f"  ✗ FEHLER: Output enthält NaN-Werte - Modell-Fehler!")
            return
        
        # Check ob Bild komplett schwarz ist
        if np.mean(img_array) < 10:
            print(f"  ⚠ WARNUNG: Ausgabebild ist sehr dunkel (Durchschnittswert: {np.mean(img_array):.2f})")
        else:
            print(f"  ✓ Ausgabebild erzeugt (Helligkeit: {np.mean(img_array):.2f})")
        
        gen_image.save(out_path)
        print(f"  ✓ Gespeichert: {out_path}\n")
        
    except Exception as e:
        print(f"  ✗ FEHLER: {e}\n")
        raise

def load_or_compute_depth_map(sim_rgb_path: Path, target_size=(512, 512)) -> Image.Image:
    """
    Lädt oder berechnet eine Tiefenkarte als Control-Bild.
    Variante A: du hast aus der Sim bereits eine Depth-Map auf Platte.
    Variante B: du approximierst die Tiefe mit einem Monodepth-Modell (nicht im Code enthalten).
    """
    # TODO: Pfadlogik anpassen, falls Depth-Bilder separat gespeichert werden
    depth_candidate = sim_rgb_path.with_name(sim_rgb_path.stem + "_depth.png")

    if depth_candidate.exists():
        depth = Image.open(depth_candidate).convert("L")
        depth = depth.resize(target_size, Image.BILINEAR)
        depth = np.array(depth)
    else:
        # Platzhalter: einfache Fake-Depth aus dem Grauwert (nur als Beispiel)
        rgb = load_image(sim_rgb_path, target_size)
        gray = np.array(rgb.convert("L"), dtype=np.float32)
        depth = cv2.GaussianBlur(gray, (15, 15), 0)

        # TODO: Stattdessen ein echtes Depth-Netz aufrufen
        # depth = run_monodepth_model(np.array(rgb))

    # drei Kanäle für ControlNet
    depth_3c = np.stack([depth] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(depth_3c)


def process_split(
    split_name: str,
    sim_root: Path,
    out_root: Path,
    base_prompt: str = None,
):
    """
    Durchläuft z.B. data/sim/train und schreibt Ergebnisse nach data/sim2real/train.
    Bild- und Label-Namen bleiben idealerweise erhalten.
    """
    sim_split_dir = sim_root / split_name
    out_split_dir = out_root / split_name

    image_exts = {".png", ".jpg", ".jpeg"}
    sim_paths = [
        p for p in sim_split_dir.rglob("*")
        if p.suffix.lower() in image_exts
    ]

    for i, sim_path in enumerate(sim_paths):
        rel = sim_path.relative_to(sim_split_dir)
        out_path = out_split_dir / rel

        print(f"[{split_name}] ({i+1}/{len(sim_paths)}) {sim_path} -> {out_path}")
        sim_to_real_single(
            sim_rgb_path=sim_path,
            out_path=out_path,
            base_prompt=base_prompt,
        )

def main():
    sim_root = Path("inputs")
    out_root = Path("outputs")

    # Sehr kurzer und einfacher Prompt - trainiertes Modell sollte Features selbst lernen
    base_prompt = "realistic photo, detailed"

    out_root.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg"}
    sim_paths = [
        p for p in sim_root.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    if not sim_paths:
        print(f"⚠ Keine Bilder in '{sim_root}' gefunden!")
        return

    print(f"=== Starte Sim2Real Translation für {len(sim_paths)} Bilder ===\n")

    for i, sim_path in enumerate(sim_paths):
        out_path = out_root / sim_path.name

        print(f"[{i+1}/{len(sim_paths)}] {sim_path.name}")
        try:
            sim_to_real_single(
                sim_rgb_path=sim_path,
                out_path=out_path,
                base_prompt=base_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                control_scale=1.0,
                seed=42,
            )
        except Exception as e:
            print(f"  ✗ Fehler bei {sim_path.name}: {e}\n")
            continue

    print("=== Fertig: Sim2Real-Bilder erzeugt ===")

if __name__ == "__main__":
    main()
