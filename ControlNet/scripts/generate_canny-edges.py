from PIL import Image
from pathlib import Path
import cv2, os
import numpy as np

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
