import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from datasets import load_dataset

# 1) Öffentliches Beispielbild laden (Hugging Face "beans"-Datensatz)
dataset = load_dataset("beans", split="train")  # lädt automatisch ein paar Beispielbilder
sample = dataset[0]                              # erstes Bild nehmen
sim_img = sample["image"].convert("RGB")        # PIL.Image
sim_img = sim_img.resize((512, 512), Image.BILINEAR)

def make_canny_control(img_pil):
    img = np.array(img_pil)
    low, high = 100, 200
    edges = cv2.Canny(img, low, high)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)

control_img = make_canny_control(sim_img)

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()


prompt = "high quality realistic photo of a single plant in a field"
negative_prompt = "blurry, low quality, distorted"

result = pipe(
    prompt=prompt,
    image=sim_img,
    control_image=control_img,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    strength=0.6,
    guidance_scale=7.0,
    controlnet_conditioning_scale=0.8,
)

out_img = result.images[0]
out_img.save("test_controlnet_beans.png")
print("Fertig: test_controlnet_beans.png")

