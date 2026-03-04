#!/bin/bash
#SBATCH --job-name=inf23153-crop
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=35GB
#SBATCH --time=02:00:00
#SBATCH --output=/raid/homes/inf23153/image-to-image/ControlNet/slurm-crop-%j.out
#SBATCH --error=/raid/homes/inf23153/image-to-image/ControlNet/slurm-crop-%j.err
#SBATCH --chdir=/raid/homes/inf23153/image-to-image/ControlNet

set -euo pipefail

echo "========================================"
echo "SLURM Job Started: ${SLURM_JOB_ID:-unknown}"
echo "========================================"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Total CPUs available: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
nvidia-smi -L || true
echo "========================================"
echo ""

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

echo "Starte Cropping-Preprocessing via Pyxis..."

srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/ControlNet,${HOME}:/workspace/home" \
  --container-workdir=/workspace/ControlNet \
  bash -lc 'set -euo pipefail

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

  if [[ "$INSTALL_DEPS" == "1" ]]; then
    python3 -c "import cv2" >/dev/null 2>&1 || (
      echo "Installiere OpenCV (headless) in den Cache..."
      python3 -m pip install --upgrade --target "$PY_DEPS_DIR_IN_CONTAINER" pip setuptools wheel

      # GUI-linked OpenCV kann libxcb/libX*-Fehler verursachen; wir erzwingen headless.
      rm -rf \
        "$PY_DEPS_DIR_IN_CONTAINER/cv2" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_contrib_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python_headless*-*.dist-info || true

      python3 -m pip install --upgrade --force-reinstall --target "$PY_DEPS_DIR_IN_CONTAINER" opencv-python-headless
    )
  fi

  # Führt Crop aus.
  python3 scripts/crop_upper_half_40pct.py --dir scripts/data/1_dashcam --fraction 0.2 --seed 42
'
