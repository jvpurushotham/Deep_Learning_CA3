"""DCGAN for minority-class augmentation. Trained ONLY on minority images of the TRAIN split (asserted)."""
import time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from .common import DEVICE, set_seed, save_json

NZ = 100


class Generator(nn.Module):
    def __init__(self, size=64, ngf=64):
        super().__init__()
        mults = {64: [8, 4, 2, 1], 128: [16, 8, 4, 2, 1]}[size]
        layers = [nn.ConvTranspose2d(NZ, ngf * mults[0], 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * mults[0]), nn.ReLU(True)]
        for a, b in zip(mults[:-1], mults[1:]):
            layers += [nn.ConvTranspose2d(ngf * a, ngf * b, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * b), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(ngf * mults[-1], 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, size=64, ndf=64):
        super().__init__()
        mults = {64: [1, 2, 4, 8], 128: [1, 2, 4, 8, 16]}[size]
        layers = [nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)]
        for a, b in zip(mults[:-1], mults[1:]):
            layers += [nn.Conv2d(ndf * a, ndf * b, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * b), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ndf * mults[-1], 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).view(-1)


def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d): nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


class _DS(Dataset):
    def __init__(self, paths, tf): self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(self.paths[i]).convert("RGB"))


def pairwise_l1(x, n=100):
    x = x[:n].flatten(1); d = torch.cdist(x, x, p=1) / x.shape[1]
    return float(d.sum() / (len(x) * (len(x) - 1)))


def train_gan(df, minority, out: Path, size=64, epochs=300, n_synthetic=500, seed=0):
    out = Path(out); gdir = out / "gan"; gdir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    d = df[(df.label == minority) & (df.split == "train")]
    assert set(d.split) == {"train"} and len(d) > 0, "GAN must see TRAIN minority images only"
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    dl = DataLoader(_DS(d.path.tolist(), tf), batch_size=min(32, len(d)), shuffle=True, drop_last=len(d) > 32)
    G, D = Generator(size).to(DEVICE), Discriminator(size).to(DEVICE); G.apply(dcgan_init); D.apply(dcgan_init)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); fixed = torch.randn(32, NZ, 1, 1, device=DEVICE); hist = []; t0 = time.time()
    for ep in range(1, epochs + 1):
        ld = lg = n = 0
        for real in dl:
            real = real.to(DEVICE); b = real.size(0); fake = G(torch.randn(b, NZ, 1, 1, device=DEVICE))
            lossD = bce(D(real), torch.full((b,), 0.9, device=DEVICE)) + bce(D(fake.detach()), torch.zeros(b, device=DEVICE))   # one-sided label smoothing
            oD.zero_grad(); lossD.backward(); oD.step()
            lossG = bce(D(fake), torch.ones(b, device=DEVICE))                                                                    # non-saturating G loss
            oG.zero_grad(); lossG.backward(); oG.step(); ld += lossD.item(); lg += lossG.item(); n += 1
        hist.append({"epoch": ep, "loss_D": ld / n, "loss_G": lg / n})
        if ep % 50 == 0 or ep == epochs: print(f"GAN epoch {ep}: loss_D {ld/n:.3f} loss_G {lg/n:.3f}")
    G.eval(); torch.save(G.state_dict(), gdir / "generator.pt"); pd.DataFrame(hist).to_csv(gdir / "gan_history.csv", index=False)
    h = pd.DataFrame(hist); plt.figure(figsize=(6, 3)); plt.plot(h.epoch, h.loss_D, label="discriminator"); plt.plot(h.epoch, h.loss_G, label="generator")
    plt.legend(); plt.grid(alpha=.3); plt.xlabel("epoch"); plt.title("GAN losses"); plt.tight_layout(); plt.savefig(out / "figures" / "gan_losses.png", dpi=150); plt.close()
    with torch.no_grad():
        fake = G(fixed) * 0.5 + 0.5
        real = torch.stack([tf(Image.open(p).convert("RGB")) for p in d.path.tolist()[:32]]) * 0.5 + 0.5
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].imshow(make_grid(real, nrow=8).permute(1, 2, 0)); ax[0].set_title("Real (train minority)")
        ax[1].imshow(make_grid(fake.cpu(), nrow=8).permute(1, 2, 0)); ax[1].set_title("Generated (random, UNCURATED)")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.savefig(out / "figures" / "gan_samples.png", dpi=150); plt.close()
        # simple mode-collapse indicator: diversity of fakes relative to diversity of reals (pixel-space mean pairwise L1)
        rd, fd = pairwise_l1(real.cpu()), pairwise_l1(fake.cpu())
        sdir = out / "synthetic" / minority; sdir.mkdir(parents=True, exist_ok=True)
        for f in sdir.glob("*.png"): f.unlink()
        for i in range(n_synthetic): save_image(G(torch.randn(1, NZ, 1, 1, device=DEVICE)) * 0.5 + 0.5, sdir / f"syn_{i:05d}.png")
    info = {"gan_train_images": len(d), "size": size, "epochs": epochs, "n_synthetic": n_synthetic, "real_diversity_l1": rd,
            "fake_diversity_l1": fd, "diversity_ratio_fake_over_real": fd / max(rd, 1e-8), "train_time_s": round(time.time() - t0, 1),
            "note": "ratio << 1 suggests mode collapse; also inspect gan_samples.png"}
    save_json(info, gdir / "gan_info.json"); print(info); return info
