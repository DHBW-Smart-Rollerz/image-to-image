"""
smoke_test_local.py

Generates a tiny synthetic unpaired sim/real dataset, runs the
`UnpairedControlDataset` and `PatchDiscriminator` for one training step
to validate data loading and discriminator updates locally.

Run:
  python ControlNet/smoke_test_local.py
"""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).parent
TMP = ROOT / 'tmp_smoke'

def make_image(path, size=(64,64)):
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype('uint8')
    Image.fromarray(arr).save(path)

def prepare_dirs():
    if TMP.exists():
        shutil.rmtree(TMP)
    (TMP / 'data' / 'sim' / 'train').mkdir(parents=True)
    (TMP / 'data' / 'sim' / 'train' / 'controls').mkdir(parents=True)
    (TMP / 'data' / 'real' / 'train').mkdir(parents=True)
    (TMP / 'data' / 'real' / 'train' / 'controls').mkdir(parents=True)

    # create 4 images per domain
    for i in range(4):
        make_image(TMP / 'data' / 'sim' / 'train' / f'sim_{i}.png', size=(64,64))
        make_image(TMP / 'data' / 'sim' / 'train' / 'controls' / f'sim_ctrl_{i}.png', size=(64,64))
        make_image(TMP / 'data' / 'real' / 'train' / f'real_{i}.png', size=(64,64))
        make_image(TMP / 'data' / 'real' / 'train' / 'controls' / f'real_ctrl_{i}.png', size=(64,64))


def run_smoke():
    try:
        from train_unpaired_controlnet import UnpairedControlDataset, PatchDiscriminator
    except Exception as e:
        print('Failed to import training utilities:', e)
        return

    ds = UnpairedControlDataset(
        sim_dir=str(TMP / 'data' / 'sim' / 'train'),
        real_dir=str(TMP / 'data' / 'real' / 'train'),
        sim_controls=str(TMP / 'data' / 'sim' / 'train' / 'controls'),
        real_controls=str(TMP / 'data' / 'real' / 'train' / 'controls'),
        resolution=64,
    )

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disc = PatchDiscriminator().to(device)
    optim_d = torch.optim.Adam(disc.parameters(), lr=1e-4)

    batch = next(iter(dl))
    sim_img = batch['sim_img'].to(device)
    real_img = batch['real_img'].to(device)

    # simple discriminator step: fake = sim_img (proxy)
    fake_img = sim_img

    pred_real = disc(real_img).mean()
    pred_fake = disc(fake_img).mean()
    loss_d = torch.relu(1.0 - pred_real).mean() + torch.relu(1.0 + pred_fake).mean()

    optim_d.zero_grad()
    loss_d.backward()
    optim_d.step()

    print('Smoke test passed: discriminator step executed. loss_d=', float(loss_d))


if __name__ == '__main__':
    prepare_dirs()
    run_smoke()