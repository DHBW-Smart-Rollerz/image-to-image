#!/bin/bash
#SBATCH --job-name=cyclegan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

# Docker Image und Container Variablen
IMAGE_NAME="pytorch-cyclegan-pix2pix"
CONTAINER_NAME="${IMAGE_NAME}_$(date +%s)"
DOCKERFILE_PATH="./docs"
PROJECT_DIR="$HOME/I2I/CycleGAN/pytorch-CycleGAN-and-pix2pix"

# Cleanup-Funktion (wichtig für SLURM+Docker!)
cleanup() {
    echo "Stopping and removing Docker container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null
    docker rm "$CONTAINER_NAME" 2>/dev/null
    exit 0
}

# Signals für sauberes Herunterfahren
trap cleanup SIGTERM SIGINT SIGHUP EXIT

cd "$PROJECT_DIR"

# Docker-Image bauen
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:latest" "$DOCKERFILE_PATH"

if [ $? -ne 0 ]; then
    echo "Docker build failed! Exiting..."
    exit 1
fi

# Docker Container mit GPU starten und Training ausführen
echo "Running Docker container with training..."
docker run --rm \
    --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$PROJECT_DIR/datasets:/workspace/pytorch-CycleGAN-and-pix2pix/datasets" \
    -v "$PROJECT_DIR/checkpoints:/workspace/pytorch-CycleGAN-and-pix2pix/checkpoints" \
    -v "$PROJECT_DIR/results:/workspace/pytorch-CycleGAN-and-pix2pix/results" \
    "$IMAGE_NAME:latest" \
    python train.py \
        --dataroot ./datasets/sim2real \
        --name sim2real_docker \
        --model cycle_gan \
        --batch_size 8 \
        --n_epochs 100 \
        --n_epochs_decay 100 \
        --save_epoch_freq 5 \
        --display_freq 100 \
        --print_freq 50 \
        --update_html_freq 500 \
        --gpu_ids 0

echo "Training completed!"
cleanup
