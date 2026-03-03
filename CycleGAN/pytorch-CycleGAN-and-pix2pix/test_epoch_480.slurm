#!/bin/bash
#SBATCH --job-name=inf23017-569_test100
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
# Anzahl Testbilder (für Debug zunächst klein halten; beim Submit überschreibbar)
NUM_TEST="${NUM_TEST:-350}"
export NUM_TEST

echo "Starte CycleGAN Test via Pyxis..."

srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/CycleGAN,/raid/homes/inf23017:/workspace/home" \
  --container-workdir=/workspace/CycleGAN \
  bash -lc '
  set -euo pipefail

  export HOME="/workspace/home"
  export XDG_CACHE_HOME="$HOME/.cache"
  export WANDB_MODE="offline"
  export PYTHONUNBUFFERED=1
  mkdir -p "$XDG_CACHE_HOME"

  unset LOCAL_RANK
  unset RANK
  unset WORLD_SIZE

  echo "Installiere Dependencies..."
  pip install --no-cache-dir dominate visdom wandb || echo "Warnung: Installation fehlgeschlagen"

  echo "Preflight: Dataset und Checkpoints prüfen..."
  echo "testA files: $(find ./datasets/sim2real/testA -type f 2>/dev/null | wc -l)"
  echo "testB files: $(find ./datasets/sim2real/testB -type f 2>/dev/null | wc -l)"
  ls -lh ./checkpoints/sim2real_stable_finetune/569_net_G_A.pth ./checkpoints/sim2real_stable_finetune/569_net_G_B.pth 2>/dev/null || echo "Warnung: Checkpoints nicht gefunden"

  echo "Teste Checkpoint 569 von sim2real_stable_finetune (num_test=${NUM_TEST})..."

  python3 -u test.py \
    --dataroot ./datasets/sim2real \
    --name sim2real_stable_finetune \
    --model cycle_gan \
    --epoch 569 \
    --num_test "${NUM_TEST}" \
    --results_dir ./results/sim2real_test100_epoch569 \
    --phase test
'
