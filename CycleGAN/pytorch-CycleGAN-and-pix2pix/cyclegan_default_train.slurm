#!/bin/bash
#SBATCH --job-name=inf23017-cyclegan_default
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --chdir=/raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix

set -euo pipefail

JOB_START_EPOCH="$(date +%s)"
trap 'rc=$?; job_end_epoch=$(date +%s); echo "========================================"; echo "SLURM Job Finished: ${SLURM_JOB_ID:-unknown} (rc=${rc})"; echo "Gesamtlaufzeit: $((job_end_epoch - JOB_START_EPOCH)) Sekunden"; echo "========================================"' EXIT

echo "========================================"
echo "SLURM Job Started: ${SLURM_JOB_ID:-unknown}"
echo "========================================"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Total CPUs available: $(nproc)"
nvidia-smi -L || true
echo "========================================"

CONTEXT_DIR="${CONTEXT_DIR:-/raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-pytorch/pytorch:latest}"

# Fresh training run with default training hyperparameters from train.py / TrainOptions.
# Required values are limited to dataset, experiment name, and model.
TRAIN_NAME="${TRAIN_NAME:-sim2real_default_train}"
export TRAIN_NAME


echo "Starte CycleGAN Training via Pyxis..."

srun \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTEXT_DIR}:/workspace/CycleGAN,/raid/homes/inf23017:/workspace/home" \
  --container-workdir=/workspace/CycleGAN \
  bash -s <<'EOF'
set -euo pipefail

export HOME="/workspace/home"
export XDG_CACHE_HOME="$HOME/.cache"
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME"

unset LOCAL_RANK
unset RANK
unset WORLD_SIZE

echo "Installiere Dependencies..."
pip install --no-cache-dir dominate visdom wandb || echo "Warnung: Installation fehlgeschlagen"

echo "Prüfe Trainingsdaten..."
echo "trainA files: $(find ./datasets/sim2real/trainA -type f 2>/dev/null | wc -l)"
echo "trainB files: $(find ./datasets/sim2real/trainB -type f 2>/dev/null | wc -l)"

LOG_DIR="./checkpoints/${TRAIN_NAME}"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/train_${SLURM_JOB_ID:-manual}.log"

echo "Starte Training ..."
TIME_FMT="Training-Walltime: %E | CPU-Auslastung: %P | Max-RAM: %M KB"
TRAIN_CMD=(
  python3 -u train.py
  --dataroot ./datasets/sim2real
  --name "${TRAIN_NAME}"
  --model cycle_gan
  --batch_size 8
  --lr 0.00001
  --netG resnet_9blocks
  --netD basic
  --gan_mode lsgan
  --norm instance
  --lambda_A 11.5
  --lambda_B 11.5
  --lambda_identity 2
  --n_epochs 600
  --n_epochs_decay 50
  --no_flip
  --save_epoch_freq 1
)

if [ -x /usr/bin/time ]; then
  /usr/bin/time -f "$TIME_FMT" "${TRAIN_CMD[@]}" 2>&1 | tee "$TRAIN_LOG"
elif [ -x /bin/time ]; then
  /bin/time -f "$TIME_FMT" "${TRAIN_CMD[@]}" 2>&1 | tee "$TRAIN_LOG"
else
  echo "Warnung: GNU time nicht gefunden, starte Training ohne detailed time-Statistik"
  "${TRAIN_CMD[@]}" 2>&1 | tee "$TRAIN_LOG"
fi

echo "========================================"
echo "Epoch-Analyse (ohne Änderungen an train.py)"
echo "Log-Datei: $TRAIN_LOG"
awk '
  BEGIN {
    ds = 0;
    printf "epoch\ttime_sec\tsamples\tsec_per_sample\tsamples_per_sec\timages_per_sec(A+B)\n";
  }
  /The number of training images =/ {
    if (match($0, /[0-9]+/)) {
      ds = substr($0, RSTART, RLENGTH) + 0;
    }
  }
  /End of epoch/ {
    epoch = "";
    sec = "";

    if (match($0, /End of epoch[[:space:]]+[0-9]+/)) {
      epoch = substr($0, RSTART, RLENGTH);
      sub(/End of epoch[[:space:]]+/, "", epoch);
      epoch += 0;
    }

    if (match($0, /Time Taken:[[:space:]]*[0-9]+(\.[0-9]+)?/)) {
      sec = substr($0, RSTART, RLENGTH);
      sub(/Time Taken:[[:space:]]*/, "", sec);
      sec += 0;
    }

    if (epoch != "" && sec != "") {
      if (ds > 0 && sec > 0) {
        sps = ds / sec;
        sps_inv = sec / ds;
        ips = (2 * ds) / sec;
        printf "%d\t%.3f\t%d\t%.6f\t%.3f\t%.3f\n", epoch, sec, ds, sps_inv, sps, ips;
      } else {
        printf "%d\t%.3f\tNA\tNA\tNA\tNA\n", epoch, sec;
      }
    }
  }
' "$TRAIN_LOG"
echo "========================================"
EOF