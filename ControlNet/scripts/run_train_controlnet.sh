#!/usr/bin/env bash
set -e
# Minimal wrapper to run the diffusers ControlNet training example from this repo.
# Assumes WSL / Linux environment with CUDA available.

REPO_DIR=$(pwd)
DIFFUSERS_DIR="$REPO_DIR/diffusers"

if [ ! -d "$DIFFUSERS_DIR" ]; then
  echo "Cloning diffusers repository (this may take a while)..."
  git clone https://github.com/huggingface/diffusers.git "$DIFFUSERS_DIR"
fi

cd "$DIFFUSERS_DIR/examples/controlnet"

echo "Installing python deps (may require GPU build tools)..."
python -m pip install -r requirements.txt || true
python -m pip install -e . || true

echo "Launching training via accelerate..."
# Adjust paths and hyperparams below as needed
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
  --train_data_dir "$REPO_DIR/ControlNet/data/prepared" \
  --resolution 512 \
  --output_dir "$REPO_DIR/ControlNet/checkpoints/lora" \
  --learning_rate 2e-4 \
  --max_train_steps 1000 \
  --train_batch_size 1 \
  --mixed_precision fp16 \
  --gradient_accumulation_steps 1 \
  --use_lora

echo "Training launched. Checkpoints will be stored in ControlNet/checkpoints/lora"
