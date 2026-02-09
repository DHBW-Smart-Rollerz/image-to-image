#!/bin/bash
#SBATCH --job-name=inf23153_controlnet_dataset
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Pfad zum Context (angepasst, falls dein Repo an anderem Ort liegt)
CONTEXT_DIR="$HOME/image-to-image/ControlNet"
IMAGE_NAME="controlnet_image"
CONTAINER_NAME="${IMAGE_NAME}_${SLURM_JOB_ID:-$(date +%s)}"

cleanup() {
  rc=$?
  echo "Cleaning up container..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  exit $rc
}
trap cleanup EXIT INT TERM

echo "Building Docker image from ${CONTEXT_DIR}..."
docker build -t "${IMAGE_NAME}:latest" "${CONTEXT_DIR}"

echo "Running Docker container..."

docker run --rm --gpus all \
  --name "$CONTAINER_NAME" \
  -v "${CONTEXT_DIR}":/workspace/ControlNet \
  -w /workspace/ControlNet \
  "${IMAGE_NAME}:latest" \

  bash -lc 'set -euo pipefail; python scripts/generate_captions.py && python scripts/blur.py'

# Bei normalem Ende wird cleanup durch trap ausgeführt
