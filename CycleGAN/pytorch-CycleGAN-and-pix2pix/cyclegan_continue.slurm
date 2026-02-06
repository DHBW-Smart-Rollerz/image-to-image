#!/bin/bash
#SBATCH --job-name=cyclegan_continue
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

# Docker Image und Container Variablen
IMAGE_NAME="pytorch-cyclegan-pix2pix"
CONTAINER_NAME="${IMAGE_NAME}_$(date +%s)"
DOCKERFILE_PATH="./docs"
PROJECT_DIR="$HOME/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix"

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

# Checkpoint von sim2real_run1 nach sim2real_docker kopieren
echo "Copying checkpoint 300 from sim2real_run1 to sim2real_docker..."
mkdir -p checkpoints/sim2real_docker
cp checkpoints/sim2real_run1/300_net_*.pth checkpoints/sim2real_docker/
cp checkpoints/sim2real_run1/train_opt.txt checkpoints/sim2real_docker/ 2>/dev/null || true

# Docker-Image bauen
echo "Building Docker image..."
docker build --no-cache -t "$IMAGE_NAME:latest" "$DOCKERFILE_PATH"

if [ $? -ne 0 ]; then
    echo "Docker build failed! Exiting..."
    exit 1
fi

# Docker Container mit GPU starten und Training fortsetzen
echo "Running Docker container with training from checkpoint..."
docker run --rm \
    --gpus all \
    --shm-size=8g \
    --name "$CONTAINER_NAME" \
    -v "$PROJECT_DIR/datasets:/workspace/pytorch-CycleGAN-and-pix2pix/datasets" \
    -v "$PROJECT_DIR/checkpoints:/workspace/pytorch-CycleGAN-and-pix2pix/checkpoints" \
    -v "$PROJECT_DIR/results:/workspace/pytorch-CycleGAN-and-pix2pix/results" \
    "$IMAGE_NAME:latest" \
    python train.py \
        --dataroot ./datasets/sim2real \
        --name sim2real_docker \
        --model cycle_gan \
        --continue_train \
        --epoch 300 \
        --epoch_count 301 \
        --batch_size 8 \
        --lr 0.0001 \
        --lambda_A 13.4 \
        --lambda_B 13.4 \
        --lambda_identity 0.6 \
        --n_epochs 600 \
        --n_epochs_decay 400 \
        --save_epoch_freq 5 \
        --display_freq 100 \
        --print_freq 50 \
        --update_html_freq 500

echo "Training completed!"
cleanup
