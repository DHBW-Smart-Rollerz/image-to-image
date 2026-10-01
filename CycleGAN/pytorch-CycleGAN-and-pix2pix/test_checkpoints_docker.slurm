#!/bin/bash
#SBATCH --job-name=test_checkpoints
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-test-%j.out

# Docker Image und Container Variablen
IMAGE_NAME="pytorch-cyclegan-pix2pix"
CONTAINER_NAME="${IMAGE_NAME}_test_$(date +%s)"
DOCKERFILE_PATH="./docs"
PROJECT_DIR="$HOME/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix"

# Cleanup-Funktion
cleanup() {
    echo "Stopping and removing Docker container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null
    docker rm "$CONTAINER_NAME" 2>/dev/null
    exit 0
}

trap cleanup SIGTERM SIGINT SIGHUP EXIT

cd "$PROJECT_DIR"

# Docker-Image bauen (falls noch nicht vorhanden)
echo "Building Docker image..."
docker build --no-cache -t "$IMAGE_NAME:latest" "$DOCKERFILE_PATH"

if [ $? -ne 0 ]; then
    echo "Docker build failed! Exiting..."
    exit 1
fi

# Checkpoints zum Testen
CHECKPOINT_NAME="sim2real_run4"
TEST_DATA="./datasets/sim2real"
NUM_TEST=50
EPOCHS=(300 400 500 600 700 800 900 1000)

echo "=== Testing Multiple Checkpoints with Docker ==="
echo ""

for epoch in "${EPOCHS[@]}"; do
    echo "========================================="
    echo "Testing Epoch $epoch..."
    echo "========================================="
    
    # Prüfe ob Checkpoint existiert
    if [ ! -f "checkpoints/$CHECKPOINT_NAME/${epoch}_net_G_A.pth" ]; then
        echo "⚠️  Checkpoint $epoch not found, skipping..."
        continue
    fi
    
    # Test mit Docker
    docker run --rm \
        --gpus all \
        --name "${CONTAINER_NAME}_${epoch}" \
        -v "$PROJECT_DIR/datasets:/workspace/pytorch-CycleGAN-and-pix2pix/datasets" \
        -v "$PROJECT_DIR/checkpoints:/workspace/pytorch-CycleGAN-and-pix2pix/checkpoints" \
        -v "$PROJECT_DIR/results:/workspace/pytorch-CycleGAN-and-pix2pix/results" \
        "$IMAGE_NAME:latest" \
        python test.py \
            --dataroot "$TEST_DATA" \
            --name "$CHECKPOINT_NAME" \
            --model cycle_gan \
            --epoch "$epoch" \
            --num_test "$NUM_TEST" \
            --no_dropout
    
    if [ $? -eq 0 ]; then
        echo "✓ Epoch $epoch completed"
    else
        echo "✗ Epoch $epoch failed"
    fi
    echo ""
done

echo "========================================="
echo "All tests completed!"
echo ""
echo "Results are saved in: results/$CHECKPOINT_NAME/"
echo ""
echo "To view results, check these HTML files:"
for epoch in "${EPOCHS[@]}"; do
    html_file="results/$CHECKPOINT_NAME/test_${epoch}/index.html"
    if [ -f "$html_file" ]; then
        echo "  Epoch $epoch: $html_file"
    fi
done

cleanup
