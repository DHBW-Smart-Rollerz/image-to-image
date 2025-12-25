import numpy as np
import cv2
from PIL import Image


def make_canny_control(img_pil, low=100, high=200):
    """Erzeuge ein 3-Kanal Canny-Edge-Bild aus einer PIL.Image."""
    img = np.array(img_pil.convert("RGB"))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    edges = cv2.Canny(img, low, high)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


if __name__ == "__main__":
    # kleines Smoke-Test-Beispiel
    from PIL import Image
    im = Image.new("RGB", (512, 512), color=(128, 128, 128))
    e = make_canny_control(im)
    e.save("./control_test.png")
    print("Saved control_test.png")
