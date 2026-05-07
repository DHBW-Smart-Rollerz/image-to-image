#!/bin/bash
#SBATCH --job-name=inf23153-captions
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=35GB
#SBATCH --time=00:30:00
#SBATCH --output=/raid/homes/inf23153/image-to-image/ControlNet/slurm-captions-%j.out
#SBATCH --error=/raid/homes/inf23153/image-to-image/ControlNet/slurm-captions-%j.err
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

# Optionen für generate_captions.py über ENV steuerbar
#   export CAPTIONS_DIR="scripts/data/1_dashcam"
#   export CAPTIONS_EXT=".jpg .png"            # Leerzeichen-getrennt
#   export CAPTIONS_SEQUENTIAL=1                # 1 => --sequential
#   export CAPTIONS_START=1
#   export CAPTIONS_DIGITS=4
#   export CAPTIONS_KEEP_EXT=1                  # 1 => --keep-ext
#   export CAPTIONS_DRY_RUN=1                   # 1 => --dry-run
CAPTIONS_DIR="${CAPTIONS_DIR:-}"
CAPTIONS_EXT="${CAPTIONS_EXT:-}"
CAPTIONS_SEQUENTIAL="${CAPTIONS_SEQUENTIAL:-0}"
CAPTIONS_START="${CAPTIONS_START:-1}"
CAPTIONS_DIGITS="${CAPTIONS_DIGITS:-4}"
CAPTIONS_KEEP_EXT="${CAPTIONS_KEEP_EXT:-0}"
CAPTIONS_DRY_RUN="${CAPTIONS_DRY_RUN:-0}"

export CAPTIONS_DIR CAPTIONS_EXT CAPTIONS_SEQUENTIAL CAPTIONS_START CAPTIONS_DIGITS CAPTIONS_KEEP_EXT CAPTIONS_DRY_RUN

echo "Starte Caption-Generierung via Pyxis..."

srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/ControlNet,${HOME}:/workspace/home" \
  --container-workdir=/workspace/ControlNet \
  bash -lc 'set -euo pipefail

  # Wichtig für Pyxis/Container: Host-HOME ist im Container oft nicht schreibbar.
  export HOME="/workspace/home"
  export XDG_CACHE_HOME="$HOME/.cache"
  mkdir -p "$XDG_CACHE_HOME"

  args=()

  if [[ -n "${CAPTIONS_DIR:-}" ]]; then
    args+=("--dir" "${CAPTIONS_DIR}")
  fi

  if [[ -n "${CAPTIONS_EXT:-}" ]]; then
    # CAPTIONS_EXT ist leerzeichen-getrennt (z.B. ".jpg .png")
    args+=("--ext")
    # shellcheck disable=SC2206
    exts=( ${CAPTIONS_EXT} )
    args+=("${exts[@]}")
  fi

  if [[ "${CAPTIONS_SEQUENTIAL:-0}" == "1" ]]; then
    args+=("--sequential" "--start" "${CAPTIONS_START:-1}" "--digits" "${CAPTIONS_DIGITS:-4}")
    if [[ "${CAPTIONS_KEEP_EXT:-0}" == "1" ]]; then
      args+=("--keep-ext")
    fi
  fi

  if [[ "${CAPTIONS_DRY_RUN:-0}" == "1" ]]; then
    args+=("--dry-run")
  fi

  python3 /workspace/ControlNet/scripts/generate_captions.py --dir scripts/data_car/1_car --sequential --caption "model car, miniature vehicale, detailed, realistic, high quality, sharp focus, studio lighting, dark background"
'
