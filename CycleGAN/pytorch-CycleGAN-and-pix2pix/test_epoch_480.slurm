#!/bin/bash
#SBATCH --job-name=inf23017-661_test100
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=3:00:00
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
  bash -s <<'EOF'
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
  ls -lh ./checkpoints/sim2real_stable_finetune3/719_net_G_A.pth ./checkpoints/sim2real_stable_finetune3/719_net_G_B.pth 2>/dev/null || echo "Warnung: Checkpoints nicht gefunden"

  echo "Teste Checkpoint 719 von sim2real_stable_finetune3 (num_test=${NUM_TEST})..."

  TEST_START_NS=$(date +%s%N)

  python3 -u test.py \
    --dataroot ./datasets/sim2real \
    --name sim2real_stable_finetune3 \
    --model cycle_gan \
    --epoch 719 \
    --num_test "${NUM_TEST}" \
    --results_dir ./results/sim2real_test100_epoch719 \
    --phase test

  TEST_END_NS=$(date +%s%N)
  ELAPSED_NS=$((TEST_END_NS - TEST_START_NS))

  ACTUAL_TEST_COUNT=$(find ./datasets/sim2real/testA -type f 2>/dev/null | wc -l)
  if [ "${ACTUAL_TEST_COUNT}" -gt "${NUM_TEST}" ]; then
    ACTUAL_TEST_COUNT="${NUM_TEST}"
  fi

  ELAPSED_SEC=$(awk -v ns="${ELAPSED_NS}" 'BEGIN { printf "%.3f", ns / 1000000000 }')
  IMAGES_PER_SEC=$(awk -v n="${ACTUAL_TEST_COUNT}" -v s="${ELAPSED_NS}" 'BEGIN { if (s > 0) printf "%.3f", n / (s / 1000000000); else print "nan" }')
  AVG_SEC_PER_IMAGE=$(awk -v n="${ACTUAL_TEST_COUNT}" -v s="${ELAPSED_NS}" 'BEGIN { if (n > 0) printf "%.3f", (s / 1000000000) / n; else print "nan" }')

  echo "========================================"
  echo "Inferenz abgeschlossen"
  echo "Verarbeitete Bilder: ${ACTUAL_TEST_COUNT}"
  echo "Gesamt-Inferenzzeit: ${ELAPSED_SEC} s"
  echo "Bilder pro Sekunde: ${IMAGES_PER_SEC}"
  echo "Durchschnitt pro Bild: ${AVG_SEC_PER_IMAGE} s"
  echo "========================================"
EOF
