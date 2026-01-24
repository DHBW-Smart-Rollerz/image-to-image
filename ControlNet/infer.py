import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
LORA_PATH = "lora/sim2real_dashcam.safetensors"

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

# ------------------------------------------------------------
# Pipeline laden
# ------------------------------------------------------------

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.load_lora_weights(LORA_PATH)
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

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt="a realistic dashcam photo of a road, daytime",
        negative_prompt="cartoon, illustration, painting, face, people, surreal",
        image=init_image,
        control_image=control_image,
        strength=0.3,                       # <<< entscheidend
        guidance_scale=5.5,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=30,
        generator=generator,
    )

    result.images[0].save(out_path)
    print(f"✓ Gespeichert: {out_path}")

# ------------------------------------------------------------
# Beispiel
# ------------------------------------------------------------

if __name__ == "__main__":
    sim2real(
        sim_image_path="inputs/rosbag2_2026_01_15-13_39_25_frame000003.jpg",
        out_path="outputs/rosbag2_2026_01_15-13_39_25_frame000003.jpg"
    )
