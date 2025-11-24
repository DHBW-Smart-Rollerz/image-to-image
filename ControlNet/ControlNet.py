import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# ControlNet: z.B. Canny
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

pipe.enable_xformers_memory_efficient_attention()

def load_sim_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img

def make_canny_control(img_pil):
    img = np.array(img_pil)
    low, high = 100, 200
    edges = cv2.Canny(img, low, high)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


sim_dir = "sim_images"          # Ordner mit Simulationsbildern
out_dir = "translated_images"   # Zielordner
os.makedirs(out_dir, exist_ok=True)

prompt = "realistic industrial camera image of the same scene"
negative_prompt = "blurry, low quality, distorted"

for fname in os.listdir(sim_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    sim_path = os.path.join(sim_dir, fname)
    sim_img = load_sim_image(sim_path)
    control_img = make_canny_control(sim_img)

    result = pipe(
        prompt=prompt,
        image=sim_img,
        control_image=control_img,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        strength=0.6,                    # wie stark vom Simulationsbild abweichen
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.8
    )

    out_img = result.images[0]
    out_img.save(os.path.join(out_dir, fname))
