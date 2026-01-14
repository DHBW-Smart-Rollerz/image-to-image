"""
train_unpaired_controlnet.py

Starter scaffold for Direct Unpaired ControlNet training.

This script sets up dataset loading, a simple PatchGAN discriminator and
the overall training scaffold for unpaired Sim->Real ControlNet finetuning.

IMPORTANT:
- True end-to-end adversarial training with a diffusion-based generator
  (Stable Diffusion + ControlNet) requires implementing the denoising
  training step (noise prediction) and backpropagating through the U-Net
  denoiser. That portion is non-trivial and is left as a clear TODO in the
  function `generator_update()` below. This scaffold provides the dataset,
  discriminator and training loop wiring so you can plug in the noise-pred
  training or a differentiable sampler.

Usage (example):
  accelerate launch ControlNet/train_unpaired_controlnet.py \
    --sim_dir data/sim/train --real_dir data/real/train \
    --sim_controls data/sim/train/controls --real_controls data/real/train/controls \
    --pretrained_model_name_or_path runwaysml/stable-diffusion-v1-5 \
    --pretrained_controlnet_path ./pretrained_controlnet.safetensors \
    --output_dir model/controlnet_unpaired --resolution 512 --batch_size 4 --epochs 10

This file is intended to be adapted to your ControlNet/SD training utilities
in this repository. See the comments marked TODO for the critical implementation
points required for a working unpaired adversarial training.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UnpairedControlDataset(Dataset):
    """Yields unpaired samples: (sim_img, sim_control, real_img, real_control)
    Expects directory structure like:
      sim_dir/{train,val,test}/*.png
      sim_controls/{train,val,test}/*.png
      real_dir/... and real_controls/...
    Filenames need NOT match across domains (unpaired).
    """

    def __init__(self, sim_dir, real_dir, sim_controls=None, real_controls=None, resolution=512):
        self.sim_images = sorted(Path(sim_dir).glob('**/*'))
        self.real_images = sorted(Path(real_dir).glob('**/*'))
        self.sim_controls = None if sim_controls is None else sorted(Path(sim_controls).glob('**/*'))
        self.real_controls = None if real_controls is None else sorted(Path(real_controls).glob('**/*'))
        self.resolution = resolution

        self.transform_img = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.transform_ctrl = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

        if len(self.sim_images) == 0 or len(self.real_images) == 0:
            raise RuntimeError('Empty sim_dir or real_dir passed to UnpairedControlDataset')

    def __len__(self):
        return max(len(self.sim_images), len(self.real_images))

    def _load_img(self, path, is_control=False):
        img = Image.open(path).convert('RGB')
        if is_control:
            return self.transform_ctrl(img)
        return self.transform_img(img)

    def __getitem__(self, idx):
        sim_path = self.sim_images[idx % len(self.sim_images)]
        real_path = self.real_images[random.randrange(len(self.real_images))]

        # controls fallback to nearest image if not provided
        if self.sim_controls:
            sim_ctrl_path = self.sim_controls[idx % len(self.sim_controls)]
        else:
            sim_ctrl_path = sim_path
        if self.real_controls:
            real_ctrl_path = self.real_controls[random.randrange(len(self.real_controls))]
        else:
            real_ctrl_path = real_path

        sim_img = self._load_img(sim_path, is_control=False)
        sim_ctrl = self._load_img(sim_ctrl_path, is_control=True)
        real_img = self._load_img(real_path, is_control=False)
        real_ctrl = self._load_img(real_ctrl_path, is_control=True)

        return {
            'sim_img': sim_img,
            'sim_ctrl': sim_ctrl,
            'real_img': real_img,
            'real_ctrl': real_ctrl,
        }


class PatchDiscriminator(nn.Module):
    """A small PatchGAN-like discriminator for RGB images 3xHxW -> patch logits."""

    def __init__(self, in_ch=3, base_features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_features, base_features*2, 4, 2, 1),
            nn.BatchNorm2d(base_features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_features*2, base_features*4, 4, 2, 1),
            nn.BatchNorm2d(base_features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_features*4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


def get_dataloaders(args):
    ds = UnpairedControlDataset(args.sim_dir, args.real_dir, args.sim_controls, args.real_controls, resolution=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dl


def generator_update(batch, generator, controlnet, unet, vae, tokenizer, text_encoder, optim_g, device, args):
    """
    Placeholder for generator update logic.

    TODO: Implement differentiable denoising training step here. Two main choices:
      - (A) Train using the diffusion noise-prediction objective: sample random noise, add to latents
        at a timestep, compute predicted noise via U-Net+ControlNet, compute MSE to ground-truth noise.
        Backprop through ControlNet and U-Net (or partially freeze U-Net) and optionally include
        adversarial/perceptual/cycle losses.
      - (B) Implement a differentiable sampling pipeline (e.g., differentiable DDIM steps) so that
        gradients from image-space discriminator/perceptual losses flow back into ControlNet.

    For this scaffold we raise NotImplementedError and document required components.
    """
    # Implement standard diffusion noise-prediction training (MSE on predicted noise).
    # Steps:
    #  - encode sim images to latents via VAE
    #  - sample random timesteps and noise
    #  - add noise via scheduler
    #  - get text embeddings (use provided prompt)
    #  - forward through ControlNet to get additional residuals
    #  - forward through UNet with controlnet residuals to predict noise
    #  - compute MSE loss between predicted and true noise and step optimizer

    from diffusers import DDPMScheduler
    mse_loss = torch.nn.MSELoss()

    sim_img = batch['sim_img'].to(device)
    sim_ctrl = batch['sim_ctrl'].to(device)
    batch_size = sim_img.size(0)

    # Text conditioning: replicate prompt for the batch
    tokens = tokenizer([args.prompt] * batch_size, padding='max_length', truncation=True,
                       max_length=tokenizer.model_max_length, return_tensors='pt')
    input_ids = tokens.input_ids.to(device)
    encoder_hidden_states = text_encoder(input_ids)[0]

    # Encode images to latents
    with torch.no_grad():
        # VAE encoding without grad for sampling latents
        enc = vae.encode(sim_img).latent_dist
        latents = enc.sample() * getattr(vae.config, 'scaling_factor', 0.18215)

    # Create scheduler (local instance) if not passed; use standard timesteps
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # sample random noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()

    # add noise to latents
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # Forward through ControlNet to obtain conditioning residuals
    # controlnet expects control images in same device/dtype
    controlnet_out = controlnet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                                 controlnet_cond=sim_ctrl, return_dict=True)

    # Forward through UNet with additional residuals
    unet_out = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=controlnet_out.down_block_additional_residuals,
                    mid_block_additional_residual=controlnet_out.mid_block_additional_residual,
                    return_dict=True)

    # Extract predicted noise
    if hasattr(unet_out, 'sample'):
        noise_pred = unet_out.sample
    else:
        # Some UNet variants return tuple (sample, ...)
        noise_pred = unet_out[0]

    # Compute loss and step optimizer
    optim_g.zero_grad()
    loss = mse_loss(noise_pred, noise)
    loss.backward()
    optim_g.step()

    # Optional: return loss value for logging
    return loss.item()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = get_dataloaders(args)

    # -- Discriminator
    disc = PatchDiscriminator(in_ch=3).to(device)
    optim_d = torch.optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    # -- Load pretrained Stable Diffusion + ControlNet (short references)
    # NOTE: we import here to avoid mandatory dependency if user only uses dataset utilities.
    try:
        from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
    except Exception as e:
        print('Missing diffusers/transformers. Install requirements to enable SD/ControlNet loading.')
        raise

    print('Loading pretrained models (ControlNet + SD components). This may take time...')
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet_path, torch_dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.float32).to(device)

    # Load SD components (UNet, VAE, text encoder/tokenizer) — adjust names to your checkpoint
    # TODO: adapt loading to repo conventions and checkpoint formats (diffusers vs ckpt)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet').to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae').to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder').to(device)

    # Optionally freeze parts to reduce memory / keep stability
    if args.freeze_unet:
        for p in unet.parameters():
            p.requires_grad = False
    if args.freeze_text_encoder:
        for p in text_encoder.parameters():
            p.requires_grad = False

    # Optimizer for generator side (ControlNet params + optionally part of UNet)
    trainable_params = [p for p in controlnet.parameters() if p.requires_grad]
    if not args.freeze_unet:
        trainable_params += [p for p in unet.parameters() if p.requires_grad]
    optim_g = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Perceptual net placeholder (VGG) if used later
    vgg = None

    # Training loop scaffold
    global_step = 0
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dl):
            # Move tensors to device
            sim_img = batch['sim_img'].to(device)
            sim_ctrl = batch['sim_ctrl'].to(device)
            real_img = batch['real_img'].to(device)
            real_ctrl = batch['real_ctrl'].to(device)

            # ------------------
            # 1) Discriminator update
            # ------------------
            disc.requires_grad_(True)
            optim_d.zero_grad()

            # Generate fake real images from sim control using the generator pipeline.
            # NOTE: to get gradients through generator you must implement generator_update;
            # here we produce samples in inference mode (no grad) to train the discriminator.
            with torch.no_grad():
                # TODO: replace with differentiable path when implementing generator_update
                # For now, we use the sim control as a proxy: (this is NOT training the generator)
                fake_img = sim_img  # placeholder; replace with generated images

            pred_real = disc(real_img).mean()
            pred_fake = disc(fake_img).mean()
            # hinge loss (discriminator)
            loss_d = torch.relu(1.0 - pred_real).mean() + torch.relu(1.0 + pred_fake).mean()
            loss_d.backward()
            optim_d.step()

            # ------------------
            # 2) Generator update (ControlNet + optionally UNet)
            # ------------------
            # This function must implement noise-prediction training or differentiable sampling
            try:
                generator_update(batch, None, controlnet, unet, vae, tokenizer, text_encoder, optim_g, device, args)
            except NotImplementedError as e:
                if global_step == 0:
                    print('\ngenerator_update() is not implemented in this scaffold. See TODO in file for details.')
                # stop early since generator training is required for real progress
                return

            global_step += 1

            if global_step % args.save_interval == 0:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save controlnet weights (diffusers format)
                print(f'Saving ControlNet weights to {out_dir}')
                controlnet.save_pretrained(out_dir / f'controlnet-step-{global_step}')

    print('Training finished (scaffold). Implement `generator_update()` to perform real training.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_dir', type=str, required=True)
    parser.add_argument('--real_dir', type=str, required=True)
    parser.add_argument('--sim_controls', type=str, default=None)
    parser.add_argument('--real_controls', type=str, default=None)
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--pretrained_controlnet_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='model/controlnet_unpaired')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=2e-5)
    parser.add_argument('--mixed_precision', choices=['no', 'fp16'], default='fp16')
    parser.add_argument('--freeze_unet', action='store_true')
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--save_interval', type=int, default=500)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
