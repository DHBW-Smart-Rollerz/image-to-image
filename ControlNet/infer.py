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
CONTROLNET_MODEL_CANNY = "lllyasviel/sd-controlnet-canny"
CONTROLNET_MODEL_DEPTH = "lllyasviel/sd-controlnet-depth"
LORA_PATH = "output_lora/sim2real_dashcam.safetensors"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


USE_CANNY = _env_bool("USE_CANNY", False)
CANNY_SCALE = float(os.getenv("CANNY_SCALE", "0.40"))
DEPTH_SCALE = float(os.getenv("DEPTH_SCALE", "0.70"))

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


def _ensure_3c_uint8_pil(x, size):
    if isinstance(x, Image.Image):
        return x.convert("RGB").resize(size, Image.BILINEAR)

    arr = np.array(x)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected depth map shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Image.fromarray(arr).resize(size, Image.BILINEAR)

_MIDAS = None
_MIDAS_TRANSFORM = None


def _get_midas(device: str):
    global _MIDAS, _MIDAS_TRANSFORM
    if _MIDAS is None or _MIDAS_TRANSFORM is None:
        model_type = "DPT_Hybrid"
        _MIDAS = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _MIDAS_TRANSFORM = midas_transforms.dpt_transform
    _MIDAS.to(device).eval()
    return _MIDAS, _MIDAS_TRANSFORM


def compute_depth_midas(image: Image.Image, device: str = DEVICE):
    midas, transform = _get_midas(device)

    # MiDaS expects float RGB in [0, 1]
    img = np.array(image).astype(np.float32) / 255.0
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

    depth = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().float().cpu().numpy()

    # Robust normalization for visualization / ControlNet input
    lo, hi = np.percentile(depth, (2, 98))
    depth = np.clip(depth, lo, hi)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth_3c = np.stack([depth] * 3, axis=-1)
    return Image.fromarray(depth_3c)


_AUX_DEPTH_DETECTOR = None
_AUX_DEPTH_KIND = None


def _get_aux_depth_detector(device: str, kind: str):
    global _AUX_DEPTH_DETECTOR, _AUX_DEPTH_KIND
    if _AUX_DEPTH_DETECTOR is not None and _AUX_DEPTH_KIND == kind:
        return _AUX_DEPTH_DETECTOR

    try:
        import importlib

        controlnet_aux = importlib.import_module("controlnet_aux")
    except Exception as e:
        raise RuntimeError(
            "controlnet-aux is not available (install it via requirements.txt)."
        ) from e

    detector = None
    kind = kind.lower().strip()

    if kind in {"aux-zoe", "zoe"} and hasattr(controlnet_aux, "ZoeDetector"):
        detector_cls = getattr(controlnet_aux, "ZoeDetector")
        detector = (
            detector_cls.from_pretrained("lllyasviel/Annotators")
            if hasattr(detector_cls, "from_pretrained")
            else detector_cls()
        )
        _AUX_DEPTH_KIND = "aux-zoe"
    elif kind in {"aux-midas", "aux", "midas-aux"} and hasattr(controlnet_aux, "MidasDetector"):
        detector_cls = getattr(controlnet_aux, "MidasDetector")
        detector = (
            detector_cls.from_pretrained("lllyasviel/Annotators")
            if hasattr(detector_cls, "from_pretrained")
            else detector_cls()
        )
        _AUX_DEPTH_KIND = "aux-midas"
    elif kind in {"aux-depth-anything", "depth-anything", "aux-da"} and hasattr(controlnet_aux, "DepthAnythingDetector"):
        detector_cls = getattr(controlnet_aux, "DepthAnythingDetector")
        # Different releases use different weight sources; try a couple common ones.
        if hasattr(detector_cls, "from_pretrained"):
            try:
                detector = detector_cls.from_pretrained("lllyasviel/Annotators")
            except Exception:
                detector = detector_cls.from_pretrained("LiheYoung/depth-anything-small-hf")
        else:
            detector = detector_cls()
        _AUX_DEPTH_KIND = "aux-depth-anything"
    else:
        raise RuntimeError(
            f"Unknown/unsupported DEPTH_BACKEND '{kind}' for controlnet-aux. "
            "Use 'aux-midas' or 'aux-zoe' (or fallback 'midas')."
        )

    if hasattr(detector, "to"):
        detector = detector.to(device)

    _AUX_DEPTH_DETECTOR = detector
    return detector


def compute_depth(image: Image.Image, device: str = DEVICE):
    backend = os.getenv("DEPTH_BACKEND", "midas").lower().strip()
    size = image.size

    if backend in {"midas", "hub-midas"}:
        return _ensure_3c_uint8_pil(compute_depth_midas(image, device=device), size)

    if backend.startswith("aux") or backend in {"zoe", "depth-anything"}:
        detector = _get_aux_depth_detector(device=device, kind=backend)
        depth = detector(image)
        return _ensure_3c_uint8_pil(depth, size)

    raise RuntimeError(
        f"Unknown DEPTH_BACKEND '{backend}'. Use 'midas', 'aux-midas', or 'aux-zoe'."
    )

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

controlnet_depth = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_DEPTH,
    torch_dtype=torch.float16,
)

if USE_CANNY:
    controlnet_canny = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_CANNY,
        torch_dtype=torch.float16,
    )
    controlnet_for_pipe = [controlnet_canny, controlnet_depth]
else:
    controlnet_for_pipe = controlnet_depth

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet_for_pipe,
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.load_lora_weights(LORA_PATH, weight=1.6)
pipe.fuse_lora()

pipe = pipe.to(DEVICE)

# ------------------------------------------------------------
# Inferenz
# ------------------------------------------------------------

def sim2real(
    sim_image_path: str,
    out_path: str,
    seed: int = 42,
    save_control_maps: bool = True,
):
    init_image = load_image(sim_image_path)

    depth_image = compute_depth(init_image)

    if USE_CANNY:
        canny_image = compute_canny(init_image)
        control_images = [canny_image, depth_image]
        conditioning_scale = [CANNY_SCALE, DEPTH_SCALE]
    else:
        canny_image = None
        control_images = depth_image
        conditioning_scale = DEPTH_SCALE

    if save_control_maps:
        out_dir = os.path.dirname(out_path) or "."
        base = os.path.splitext(os.path.basename(out_path))[0]
        depth_path = os.path.join(out_dir, f"{base}_depth.png")
        if USE_CANNY and canny_image is not None:
            canny_path = os.path.join(out_dir, f"{base}_canny.png")
            canny_image.save(canny_path)
            print(f"✓ Gespeichert: {canny_path}")
        depth_image.save(depth_path)
        print(f"✓ Gespeichert: {depth_path}")

    mask_image = create_mask(init_image.size)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=(
            "wet asphalt, reflective road surface, specular highlights, road reflections"
        ),
        negative_prompt="car details, vehicle focus, flat lighting, matte surface",
        image=init_image,
        mask_image=mask_image, 
        control_image=control_images,
        strength=0.85,
        guidance_scale=4.0,
        controlnet_conditioning_scale=conditioning_scale,
        num_inference_steps=40,
        generator=generator,
    )

    result.images[0].save(out_path)
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
