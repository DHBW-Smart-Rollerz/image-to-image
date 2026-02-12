#!/bin/bash
#SBATCH --job-name=infer_i2i_controlnet
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
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

# Depth backend for infer.py:
# - "midas" (default, no controlnet-aux)
# - "aux-midas" / "aux-zoe" (requires controlnet-aux)
DEPTH_BACKEND="${DEPTH_BACKEND:-aux-zoe}"

docker run --rm --gpus "device=5" \
  --name "$CONTAINER_NAME" \
  -e DEPTH_BACKEND="$DEPTH_BACKEND" \
  -v "${CONTEXT_DIR}":/workspace/ControlNet \
  -w /workspace/ControlNet \
  "${IMAGE_NAME}:latest" \
  python3 /workspace/ControlNet/infer.py

# Bei normalem Ende wird cleanup durch trap ausgeführt
