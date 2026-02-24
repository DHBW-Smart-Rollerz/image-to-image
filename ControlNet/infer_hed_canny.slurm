#!/bin/bash
#SBATCH --job-name=infer_i2i_controlnet
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Umgebungsvariablen für Container-GPU-Support (wie im Pyxis-Beispiel)
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Pfad zum Repo (angepasst, falls dein Repo an anderem Ort liegt)
CONTEXT_DIR="$HOME/image-to-image/ControlNet"

# Container Image für Pyxis (Default wie im Beispiel; kann per ENV überschrieben werden)
# Beispiele:
#   export CONTAINER_IMAGE=pytorch/pytorch:latest
#   export CONTAINER_IMAGE=docker://nvcr.io/nvidia/pytorch:24.01-py3
CONTAINER_IMAGE="${CONTAINER_IMAGE:-pytorch/pytorch:latest}"

# Python deps werden in ein persistentes Verzeichnis (auf dem Host) installiert,
# damit Jobs nicht jedes Mal alles neu laden müssen.
# Kann bei Bedarf überschrieben werden: export PY_DEPS_DIR=$HOME/.cache/controlnet_py_deps
PY_DEPS_DIR="${PY_DEPS_DIR:-$HOME/.cache/controlnet_py_deps}"

# Falls du die Installation überspringen willst (wenn alles schon im Image ist):
#   export INSTALL_DEPS=0
INSTALL_DEPS="${INSTALL_DEPS:-1}"
export INSTALL_DEPS

# Depth backend for infer.py:
# - "midas" (default, no controlnet-aux)
# - "aux-midas" / "aux-zoe" (requires controlnet-aux)
export DEPTH_BACKEND="${DEPTH_BACKEND:-aux-zoe}"
export DEPTH_BACKEND

export PY_DEPS_DIR

echo "Überprüfe GPU-Sichtbarkeit im Container..."
srun --export=ALL --container-image="${CONTAINER_IMAGE}" nvidia-smi -L

echo "Starte Inferenz via Pyxis..."
srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/ControlNet,${HOME}:/workspace/home" \
  --container-workdir=/workspace/ControlNet \
  bash -lc 'set -euo pipefail

  # Defaults (wichtig, weil -u sonst bei nicht gesetzten Variablen abbricht)
  INSTALL_DEPS="${INSTALL_DEPS:-1}"
  DEPTH_BACKEND="${DEPTH_BACKEND:-aux-zoe}"

  # Wichtig für Pyxis/Container: HOME vom Host zeigt oft auf einen Pfad,
  # der im Container nicht (schreibbar) gemountet ist. Wir nutzen das gemountete HOME.
  export HOME="/workspace/home"
  export XDG_CACHE_HOME="$HOME/.cache"
  export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
  mkdir -p "$PIP_CACHE_DIR"

  # Persistentes Zielverzeichnis im Container (gemountet vom Host)
  PY_DEPS_DIR_IN_CONTAINER="/workspace/home/.cache/controlnet_py_deps"
  mkdir -p "$PY_DEPS_DIR_IN_CONTAINER"

  export PYTHONPATH="$PY_DEPS_DIR_IN_CONTAINER:${PYTHONPATH:-}"

  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    # Wichtig: Wenn DEPTH_BACKEND auf aux-* steht, braucht infer.py zusätzlich controlnet_aux.
    if [[ "${DEPTH_BACKEND}" == aux* || "${DEPTH_BACKEND}" == "zoe" || "${DEPTH_BACKEND}" == "depth-anything" ]]; then
      NEED_INSTALL=0
      python3 -c "import cv2, diffusers, transformers, accelerate" >/dev/null 2>&1 || NEED_INSTALL=1
      [[ -f "$PY_DEPS_DIR_IN_CONTAINER/controlnet_aux/__init__.py" ]] || NEED_INSTALL=1
      [[ "$NEED_INSTALL" == "0" ]] || (
        echo "Installiere Python-Abhängigkeiten (einmalig im Cache)..."
        # Nicht ins (evtl. read-only) User-Site installieren; alles in den Target-Ordner legen.
        python3 -m pip install --upgrade --target "$PY_DEPS_DIR_IN_CONTAINER" pip setuptools wheel

        # Stale OpenCV (z.B. opencv-python statt headless) führt zu libxcb/libX*-Fehlern.
        # Da wir mit --target arbeiten, räumen wir das Paketverzeichnis vor der Neuinstallation weg.
        rm -rf \
          "$PY_DEPS_DIR_IN_CONTAINER/cv2" \
          "$PY_DEPS_DIR_IN_CONTAINER/opencv_python" \
          "$PY_DEPS_DIR_IN_CONTAINER/opencv_contrib_python" \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python*-*.dist-info \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python*-*.dist-info \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python_headless*-*.dist-info || true

        python3 -m pip install --upgrade --force-reinstall --target "$PY_DEPS_DIR_IN_CONTAINER" -r /workspace/ControlNet/requirements.txt

        # mediapipe zieht opencv-contrib-python nach, das oft GUI/X11 Dependencies (libxcb) mitbringt.
        # Wir nutzen daher die headless-contrib Variante (mediapipe/controlnet-aux kompatibel, ohne X11).
        rm -rf \
          "$PY_DEPS_DIR_IN_CONTAINER/cv2" \
          "$PY_DEPS_DIR_IN_CONTAINER/opencv_python" \
          "$PY_DEPS_DIR_IN_CONTAINER/opencv_contrib_python" \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python*-*.dist-info \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python*-*.dist-info \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python_headless*-*.dist-info \
          "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python_headless*-*.dist-info || true
        python3 -m pip install --upgrade --force-reinstall --no-deps --target "$PY_DEPS_DIR_IN_CONTAINER" opencv-contrib-python-headless
      )
    else
      python3 -c "import cv2, diffusers, transformers, accelerate" >/dev/null 2>&1 || (
      echo "Installiere Python-Abhängigkeiten (einmalig im Cache)..."
      # Nicht ins (evtl. read-only) User-Site installieren; alles in den Target-Ordner legen.
      python3 -m pip install --upgrade --target "$PY_DEPS_DIR_IN_CONTAINER" pip setuptools wheel

      # Stale OpenCV (z.B. opencv-python statt headless) führt zu libxcb/libX*-Fehlern.
      # Da wir mit --target arbeiten, räumen wir das Paketverzeichnis vor der Neuinstallation weg.
      rm -rf \
        "$PY_DEPS_DIR_IN_CONTAINER/cv2" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_contrib_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python_headless*-*.dist-info || true

      python3 -m pip install --upgrade --force-reinstall --target "$PY_DEPS_DIR_IN_CONTAINER" -r /workspace/ControlNet/requirements.txt

      # mediapipe zieht opencv-contrib-python nach, das oft GUI/X11 Dependencies (libxcb) mitbringt.
      # Wir nutzen daher die headless-contrib Variante (mediapipe/controlnet-aux kompatibel, ohne X11).
      rm -rf \
        "$PY_DEPS_DIR_IN_CONTAINER/cv2" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_contrib_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python_headless*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python_headless*-*.dist-info || true
      python3 -m pip install --upgrade --force-reinstall --no-deps --target "$PY_DEPS_DIR_IN_CONTAINER" opencv-contrib-python-headless
      )
    fi
  fi

  python3 /workspace/ControlNet/infer_hed.py'