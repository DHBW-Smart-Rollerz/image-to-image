#!/bin/bash
#SBATCH --job-name=inf23153_controlnet_train
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

docker run --rm --gpus "device=4" \
  --name "$CONTAINER_NAME" \
  -v "${CONTEXT_DIR}":/workspace/ControlNet \
  -w /workspace/ControlNet \
  "${IMAGE_NAME}:latest" \
  accelerate launch --num_processes 1 --mixed_precision fp16 /workspace/ControlNet/scripts/train_network.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --train_data_dir /workspace/ControlNet/scripts/data/ --resolution 512 --enable_bucket --max_data_loader_n_workers 0 --network_module networks.lora --network_dim 16 --network_alpha 16 --train_batch_size 2 --gradient_accumulation_steps 4 --learning_rate 1e-4 --text_encoder_lr 5e-5 --max_train_steps 12000 --lr_scheduler cosine --output_dir /workspace/ControlNet/output_lora --output_name sim2real_dashcam

# Bei normalem Ende wird cleanup durch trap ausgeführt
