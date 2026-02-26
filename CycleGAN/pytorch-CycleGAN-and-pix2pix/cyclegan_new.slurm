#!/bin/bash
#SBATCH --job-name=inf23017-sim2real
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir=/raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix

set -euo pipefail

echo "========================================"
echo "SLURM Job Started: ${SLURM_JOB_ID:-unknown}"
echo "========================================"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Total CPUs available: $(nproc)"
nvidia-smi -L || true
echo "========================================"

# Repo-Pfad
CONTEXT_DIR="${CONTEXT_DIR:-/raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix}"

# Container Image
CONTAINER_IMAGE="${CONTAINER_IMAGE:-pytorch/pytorch:latest}"

echo "Starte CycleGAN Training via Pyxis..."

srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/CycleGAN,/raid/homes/inf23017:/workspace/home" \
  --container-workdir=/workspace/CycleGAN \
  bash -lc '
  set -euo pipefail

  export HOME="/workspace/home"
  export XDG_CACHE_HOME="$HOME/.cache"
  mkdir -p "$XDG_CACHE_HOME"
  
  # Stelle sicher, dass keine DDP Umgebungsvariablen gesetzt sind
  unset LOCAL_RANK
  unset RANK
  unset WORLD_SIZE
  
  echo "Installiere Dependencies..."
  pip install --no-cache-dir dominate visdom wandb || echo "Warnung: Installation fehlgeschlagen"

  echo "Starte Training..."
  
  # Kopiere Checkpoint 300 von sim2real_run1 für Finetuning
  mkdir -p ./checkpoints/sim2real_run1_finetune
  cp ./checkpoints/sim2real_run1/300_net_*.pth ./checkpoints/sim2real_run1_finetune/ || echo "Warnung: Checkpoints nicht gefunden"
  
  python3 train.py \
    --dataroot ./datasets/sim2real \
    --name sim2real_run1_finetune \
    --model cycle_gan \
    --continue_train \
    --epoch 300 \
    --epoch_count 301 \
    --batch_size 8 \
    --lr 0.00008 \
    --lambda_A 6.0 \
    --lambda_B 6.0 \
    --lambda_identity 2.5 \
    --n_epochs 400 \
    --n_epochs_decay 100 \
    --save_epoch_freq 5 \
    --display_freq 100 \
    --print_freq 50
'