#!/bin/bash
#SBATCH --job-name=infer_i2i_canny
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Pyxis/Enroot Runtime-Dir Fix (für Permission-Fehler bei /run/user/*)
export XDG_RUNTIME_DIR="/tmp/enroot-runtime-${UID}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Pyxis/Enroot GPU support
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Repo-Pfad (kann per ENV überschrieben werden)
CONTEXT_DIR="${CONTEXT_DIR:-$HOME/image-to-image/ControlNet}"

# Container Image für Pyxis (Default ist Platzhalter; am besten beim sbatch-Aufruf setzen)
# Beispiele:
#   export CONTAINER_IMAGE=pytorch/pytorch:latest
#   export CONTAINER_IMAGE=docker://nvcr.io/nvidia/pytorch:24.01-py3
CONTAINER_IMAGE="${CONTAINER_IMAGE:-pytorch/pytorch:latest}"

# Python deps werden in ein persistentes Verzeichnis (auf dem Host) installiert,
# damit Jobs nicht jedes Mal alles neu installieren müssen.
PY_DEPS_DIR="${PY_DEPS_DIR:-$HOME/.cache/controlnet_py_deps}"

# Falls du die Installation überspringen willst (wenn alles schon im Image ist):
#   export INSTALL_DEPS=0
INSTALL_DEPS="${INSTALL_DEPS:-1}"
export PY_DEPS_DIR INSTALL_DEPS

echo "Überprüfe GPU-Sichtbarkeit im Container..."
srun --export=ALL --container-image="${CONTAINER_IMAGE}" nvidia-smi -L

echo "Starte Canny-Inferenz via Pyxis..."
srun \
  --export=ALL,XDG_RUNTIME_DIR \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/ControlNet,${HOME}:/workspace/home,${XDG_RUNTIME_DIR}:${XDG_RUNTIME_DIR}" \
  --container-workdir=/workspace/ControlNet \
  bash -lc 'set -euo pipefail

  # Defaults (wichtig, weil -u sonst bei nicht gesetzten Variablen abbricht)
  INSTALL_DEPS="${INSTALL_DEPS:-1}"

  # Wichtig für Pyxis/Container: Host-HOME ist im Container oft nicht schreibbar.
  export HOME="/workspace/home"
  export XDG_CACHE_HOME="$HOME/.cache"
  export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
  mkdir -p "$PIP_CACHE_DIR"

  # Persistentes Zielverzeichnis im Container (gemountet vom Host)
  PY_DEPS_DIR_IN_CONTAINER="/workspace/home/.cache/controlnet_py_deps"
  mkdir -p "$PY_DEPS_DIR_IN_CONTAINER"
  export PYTHONPATH="$PY_DEPS_DIR_IN_CONTAINER:${PYTHONPATH:-}"

  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    # Wichtig: Dieses Inferenz-Skript nutzt zusätzlich den HED Preprocessor aus controlnet-aux.
    # Achtung: `import controlnet_aux` kann (je nach Installation) über mediapipe/openpose crashen.
    # Daher prüfen wir hier nur auf Dateiexistenz im --target Cache, nicht per Python-Import.
    NEED_INSTALL=0
    python3 -c "import cv2, diffusers, transformers, accelerate" >/dev/null 2>&1 || NEED_INSTALL=1
    [[ -f "$PY_DEPS_DIR_IN_CONTAINER/controlnet_aux/hed/__init__.py" ]] || NEED_INSTALL=1
    [[ "$NEED_INSTALL" == "0" ]] || (
      echo "Installiere Python-Abhängigkeiten (einmalig im Cache)..."
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
      # Für diese Inferenz wollen wir ein headless-cv2, aber weiterhin "contrib" (mediapipe/controlnet-aux kompatibel).
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

  # Führt Canny-ControlNet Inpaint-Inferenz aus (Eingaben: ./inputs, Ausgaben: ./outputs)
  python3 "/workspace/ControlNet/infer.py"
'