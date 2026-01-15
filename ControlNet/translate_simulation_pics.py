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

# TODO: Wähle ein geeignetes SD- und ControlNet-Modell
BASE_MODEL = "runwayml/stable-diffusion-v1-5"  # oder eigenes fein­getuntes Modell
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"  # oder z.B. canny/segmentation

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
pipe = pipe.to(DEVICE)
# pipe.enable_xformers_memory_efficient_attention()  # optional, falls xformers installiert
# pipe.safety_checker = None  # optional deaktivieren, falls Bilder blockiert werden

def compute_canny_edges(sim_rgb_path: Path, target_size=(512, 512)) -> Image.Image:
    img = load_image(sim_rgb_path, target_size)
    img_np = np.array(img)
    edges = cv2.Canny(img_np, 100, 200)
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
    Erzeugt einen Prompt, der gezielt Schatten, Reflexionen und Sensorrauschen beschreibt.
    base_prompt kann genutzt werden, um Strecken- oder Umgebungsinfos einzubauen.
    """
    style_parts = [
        "photo from a front-facing camera of a small autonomous model car on the road",
        "realistic lighting with strong shadows on the asphalt",
        "reflections on wet road surface and car body",
        "subtle sensor noise, motion blur and slight lens dirt",
    ]
    if base_prompt:
        style_parts.insert(0, base_prompt)
    return ", ".join(style_parts)

@torch.no_grad()
def sim_to_real_single(
    sim_rgb_path: Path,
    out_path: Path,
    base_prompt: str = None,
    num_inference_steps: int = 30,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    control_scale: float = 1.0,
    use_depth: bool = False,
):
    """
    Wandelt ein Simulationsbild in ein stilisiertes 'realistisches' Bild um und speichert es.
    strength: wie stark das Bild verändert wird (0=kaum, 1=stark).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    init_image = load_image(sim_rgb_path)
    if use_depth:
        control_image = load_or_compute_depth_map(sim_rgb_path)
    else:
        control_image = compute_canny_edges(sim_rgb_path)

    prompt = build_prompt(base_prompt)

    result = pipe(
        prompt=prompt,
        image=init_image,
        control_image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=control_scale,
        strength=strength,
    )

    gen_image = result.images[0]
    gen_image.save(out_path)

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
    # TODO: an dein Projekt anpassen
    sim_root = Path("inputs")        # enthält nur Bilder (keine Unterordner)
    out_root = Path("outputs")   # neuer Datensatz

    base_prompt = (
    "realistic indoor autonomous driving test track, "
    "front-facing low-mounted camera view from a model car, "
    "black asphalt road with white lane markings, "
    "miniature traffic signs, laboratory environment, "
    "technical research setup, "
    "wide-angle lens, slight fisheye distortion, "
    "monochrome image, high contrast, "
    "raw sensor-like appearance, "
    "daytime, high dynamic range lighting"
    )


    out_root.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg"}
    sim_paths = [
        p for p in sim_root.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    for i, sim_path in enumerate(sim_paths):
        out_path = out_root / sim_path.name

        print(f"[sim] ({i+1}/{len(sim_paths)}) {sim_path} -> {out_path}")
        sim_to_real_single(
            sim_rgb_path=sim_path,
            out_path=out_path,
            base_prompt=base_prompt,
        )

    print("Fertig: Sim2Real-Bilder erzeugt.")

if __name__ == "__main__":
    main()
