#!/bin/bash
#SBATCH --job-name=inf23153_controlnet_train
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Pyxis/Enroot Container Support
# Hinweis: --container-image muss für Pyxis erreichbar sein (z.B. docker://REGISTRY/IMAGE:TAG oder ein .sqsh Image).
# Beispiel:
#   export CONTAINER_IMAGE=docker://ghcr.io/<org>/<image>:latest
# Wichtig: Nicht pauschal "all" setzen. Wenn im Container >1 GPU sichtbar ist,
# aktiviert train_network.py DDP (torch.cuda.device_count()>1) und erwartet dann
# WORLD_SIZE/RANK Env-Variablen. Wir lassen Slurm/pyxis die GPU-Bindung setzen.
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Repo-Pfad (kann per ENV überschrieben werden)
CONTEXT_DIR="${CONTEXT_DIR:-$HOME/image-to-image/ControlNet}"

# Default ist nur ein Platzhalter. Setze CONTAINER_IMAGE beim sbatch-Aufruf passend.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-pytorch/pytorch:latest}"

# Optional: Python deps in ein persistentes Host-Verzeichnis installieren (wie in infer.slurm),
# falls das Container-Image nicht schon alles enthält.
PY_DEPS_DIR="${PY_DEPS_DIR:-$HOME/.cache/controlnet_py_deps}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
export PY_DEPS_DIR INSTALL_DEPS

echo "Überprüfe GPU-Sichtbarkeit im Container..."
srun --export=ALL --container-image="${CONTAINER_IMAGE}" nvidia-smi -L

echo "Starte Training via Pyxis..."
srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/ControlNet,${HOME}:/workspace/home" \
  --container-workdir=/workspace/ControlNet \
  bash -lc 'set -euo pipefail

  # Wichtig für Pyxis/Container: Host-HOME ist im Container oft nicht schreibbar.
  export HOME="/workspace/home"
  export XDG_CACHE_HOME="$HOME/.cache"
  export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
  mkdir -p "$PIP_CACHE_DIR"

  # GPU-Sichtbarkeit auf die zugewiesene(n) GPU(s) beschränken.
  # Falls Slurm CUDA_VISIBLE_DEVICES schon setzt: auf erstes Device reduzieren (1-GPU-Job).
  GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}}"
  if [[ -n "${GPU_LIST}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(echo "${GPU_LIST}" | cut -d, -f1)"
  else
    export CUDA_VISIBLE_DEVICES="0"
  fi
  export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

  # Robust gegen Fälle, in denen Accelerate/Slurm dennoch torch.distributed via env:// initialisiert:
  # env:// erwartet mindestens WORLD_SIZE und (implizit) RANK sowie MASTER_ADDR/PORT.
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export RANK="${RANK:-0}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"

  INSTALL_DEPS="${INSTALL_DEPS:-1}"

  # Persistentes Zielverzeichnis im Container (gemountet vom Host)
  PY_DEPS_DIR_IN_CONTAINER="/workspace/home/.cache/controlnet_py_deps"
  mkdir -p "$PY_DEPS_DIR_IN_CONTAINER"
  export PYTHONPATH="$PY_DEPS_DIR_IN_CONTAINER:${PYTHONPATH:-}"

  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    python3 -c "import torch, diffusers, transformers, accelerate" >/dev/null 2>&1 || (
      echo "Installiere Python-Abhängigkeiten (einmalig im Cache)..."
      python3 -m pip install --upgrade --target "$PY_DEPS_DIR_IN_CONTAINER" pip setuptools wheel

      # Wenn OpenCV als GUI-Variante installiert wird, kann das zu libxcb/libX*-Fehlern führen.
      rm -rf \
        "$PY_DEPS_DIR_IN_CONTAINER/cv2" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/opencv_contrib_python" \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_contrib_python*-*.dist-info \
        "$PY_DEPS_DIR_IN_CONTAINER/"opencv_python_headless*-*.dist-info || true

      python3 -m pip install --upgrade --force-reinstall --target "$PY_DEPS_DIR_IN_CONTAINER" -r /workspace/ControlNet/requirements.txt
      python3 -m pip install --upgrade --force-reinstall --no-deps --target "$PY_DEPS_DIR_IN_CONTAINER" opencv-python-headless
    )
  fi

  python3 - <<PY
import os, torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"), "RANK:", os.environ.get("RANK"), "LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"), "MASTER_PORT:", os.environ.get("MASTER_PORT"))
PY

  python3 -m accelerate.commands.launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    /workspace/ControlNet/scripts/train_network.py \
      --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
      --train_data_dir /workspace/ControlNet/scripts/data/ \
      --resolution 512 \
      --enable_bucket \
      --max_data_loader_n_workers 0 \
      --network_module networks.lora \
      --network_dim 16 \
      --network_alpha 16 \
      --train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-4 \
      --text_encoder_lr 5e-5 \
      --max_train_steps 8000 \
      --lr_scheduler cosine \
      --output_dir /workspace/ControlNet/output_lora \
      --output_name sim2real_dashcam
  '
