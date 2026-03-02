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

_controlnet_aux_import_error = None
_HEDdetector_cls = None


def _load_hed_detector_class_safely():
    """Load HEDdetector without importing controlnet_aux/__init__.py.

    controlnet_aux's package __init__ imports multiple optional detectors (e.g. mediapipe/openpose).
    On some clusters those optional deps are broken, which prevents using HED even though it doesn't need them.
    This loader injects a minimal `controlnet_aux` package into `sys.modules` and loads only the HED module.
    """

    import importlib.util
    import os
    import sys
    import types

    global _controlnet_aux_import_error

    # Fast path: if a healthy import works, use it.
    try:
        from controlnet_aux.hed import HEDdetector as HEDdetectorImported  # type: ignore

        return HEDdetectorImported
    except Exception as e:
        _controlnet_aux_import_error = e

    # Slow path: load module code directly from installed files, without executing controlnet_aux/__init__.py.
    for base in list(sys.path):
        if not base:
            continue
        pkg_dir = os.path.join(base, "controlnet_aux")
        hed_init = os.path.join(pkg_dir, "hed", "__init__.py")
        if not os.path.isfile(hed_init):
            continue

        # Ensure we don't reuse a partially-imported, broken package.
        sys.modules.pop("controlnet_aux", None)
        sys.modules.pop("controlnet_aux.hed", None)

        pkg = types.ModuleType("controlnet_aux")
        pkg.__path__ = [pkg_dir]
        pkg.__package__ = "controlnet_aux"
        sys.modules["controlnet_aux"] = pkg

        spec = importlib.util.spec_from_file_location("controlnet_aux.hed", hed_init)
        if spec is None or spec.loader is None:
            continue

        mod = importlib.util.module_from_spec(spec)
        sys.modules["controlnet_aux.hed"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            _controlnet_aux_import_error = e
            continue

        hed_cls = getattr(mod, "HEDdetector", None)
        if hed_cls is None:
            _controlnet_aux_import_error = RuntimeError("controlnet_aux.hed loaded, but HEDdetector not found")
            continue
        return hed_cls

    return None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
CONTROLNET_MODEL_HED = "lllyasviel/sd-controlnet-hed"
LORA_PATH = "output_lora/sim2real_dashcam.safetensors"

CANNY_CONDITIONING_SCALE = 0.9
HED_CONDITIONING_SCALE = 0.9

# Auto Bounding Box (geschützt)
x1, y1, x2, y2 = 110, 200, 380, 512 #0,0,0,0#50, 230, 460, 512

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


_hed_detector = None


def compute_softedge_hed(image: Image.Image, image_resolution: int = 512):
    """Compute a soft-edge map using the HED preprocessor (ControlNet Aux).

    Returns a 3-channel PIL image suitable for `sd-controlnet-hed`.
    """

    global _hed_detector, _HEDdetector_cls
    if _HEDdetector_cls is None:
        _HEDdetector_cls = _load_hed_detector_class_safely()

    if _HEDdetector_cls is None:
        raise RuntimeError(
            "HEDdetector is not available (controlnet-aux import failed). "
            "Ensure `controlnet-aux` is installed in the job environment. "
            f"Original error: {_controlnet_aux_import_error!r}"
        )

    if _hed_detector is None:
        # `controlnet_aux` downloads weights on first use.
        # Try the most common repo first, then fall back.
        last_err = None
        for repo_id in ("lllyasviel/Annotators", "lllyasviel/ControlNet"):
            try:
                _hed_detector = _HEDdetector_cls.from_pretrained(repo_id)
                break
            except Exception as e:
                last_err = e
                _hed_detector = None
        if _hed_detector is None:
            raise RuntimeError(f"Failed to load HEDdetector weights: {last_err}")

    # controlnet_aux expects an RGB PIL image
    hed = _hed_detector(
        image,
        detect_resolution=image_resolution,
        image_resolution=image_resolution,
    )

    # Ensure 3-channel RGB
    if isinstance(hed, Image.Image):
        return hed.convert("RGB")

    # Some versions may return numpy arrays
    hed_arr = np.array(hed)
    if hed_arr.ndim == 2:
        hed_arr = np.stack([hed_arr] * 3, axis=-1)
    return Image.fromarray(hed_arr).convert("RGB")

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

print("[debug] Loading ControlNet canny model...")
controlnet_canny = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
)
print("[debug] ControlNet canny loaded.")

print("[debug] Loading ControlNet HED model...")
controlnet_hed = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_HED,
    torch_dtype=torch.float16,
)
print("[debug] ControlNet HED loaded.")

print("[debug] Loading base SD pipeline...")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=[controlnet_canny, controlnet_hed],
    torch_dtype=torch.float16,
    safety_checker=None,
)
print("[debug] Base SD pipeline loaded.")

print("[debug] Loading LoRA weights...")
pipe.load_lora_weights(LORA_PATH, weight=1.6)
pipe.fuse_lora()
print("[debug] LoRA fused.")

pipe = pipe.to(DEVICE)

# ------------------------------------------------------------
# Inferenz
# ------------------------------------------------------------

def sim2real(
    sim_image_path: str,
    out_path: str,
    hed_out_path: str = None,
    seed: int = 42,
):
    init_image = load_image(sim_image_path)
    control_image_canny = compute_canny(init_image)
    hed_scale = HED_CONDITIONING_SCALE
    try:
        control_image_hed = compute_softedge_hed(init_image, image_resolution=init_image.size[0])
    except Exception as e:
        print(f"! HED/SoftEdge deaktiviert (Fallback auf nur Canny): {e}")
        control_image_hed = control_image_canny
        hed_scale = 0.0
    mask_image = create_mask(init_image.size)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=(
            "realistic dashcam photo, natural lighting, realistic asphalt texture, "
            "road surface details, soft shadows, realistic reflections, high dynamic range"
        ),
        negative_prompt="car details, vehicle focus, flat lighting, overexposed, underexposed",
        image=init_image,
        mask_image=mask_image, 
        control_image=[control_image_canny, control_image_hed],
        strength=0.85,
        guidance_scale=6.5,
        controlnet_conditioning_scale=[CANNY_CONDITIONING_SCALE, hed_scale],
        num_inference_steps=40,
        generator=generator,
    )

    out_img = result.images[0]

    out_img.save(out_path)
    print(f"✓ Gespeichert: {out_path}")
    
    # Speichere HED/Softedge-Map
    if hed_out_path:
        control_image_hed.save(hed_out_path)
        print(f"✓ HED-Map gespeichert: {hed_out_path}")


if __name__ == "__main__":
    input_dir = "inputs"
    output_dir = "outputs"
    hed_output_dir = "outputs/hed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hed_output_dir, exist_ok=True)

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
            hed_out_path = os.path.join(hed_output_dir, os.path.basename(f))
            try:
                print(f"→ Verarbeite: {f}  ->  {out_path}")
                sim2real(sim_image_path=f, out_path=out_path, hed_out_path=hed_out_path)
            except Exception as e:
                print(f"✗ Fehler beim Verarbeiten von {f}: {e}")
