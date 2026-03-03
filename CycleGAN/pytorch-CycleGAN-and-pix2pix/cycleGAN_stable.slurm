#!/bin/bash
#SBATCH --job-name=inf23017-sim2real_stable_finetune
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir=/raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix

set -euo pipefail

echo "========================================"
echo "SLURM Job Started: ${SLURM_JOB_ID:-unknown}"
echo "========================================"

CONTEXT_DIR="/raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix"
CONTAINER_IMAGE="pytorch/pytorch:latest"

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

  unset LOCAL_RANK
  unset RANK
  unset WORLD_SIZE

  echo "Installiere Dependencies..."
  pip install --no-cache-dir dominate visdom wandb || true

  echo "Starte Stabilitäts-Finetuning ab Epoche 569..."

  python3 train.py \
    --dataroot ./datasets/sim2real \
    --name sim2real_stable_finetune2 \
    --model cycle_gan \
    --continue_train \
    --epoch 569 \
    --epoch_count 570 \
    --batch_size 8 \
    --lr 0.00001 \
    --netG resnet_9blocks \
    --netD basic \
    --gan_mode lsgan \
    --norm instance \
    --lambda_A 11.5 \
    --lambda_B 11.5 \
    --lambda_identity 2 \
    --n_epochs 900 \
    --n_epochs_decay 50 \
    --no_flip \
    --save_epoch_freq 1 \
    --display_freq 200 \
    --print_freq 100
'