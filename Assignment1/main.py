import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# --- Config ---
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
IMG_SIZE = 28 * 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Data ---
print("Loading data...")
df = pd.read_csv("archive/fashion-mnist_train.csv")
images = df.iloc[:, 1:].values.astype(np.float32)
images = (images / 127.5) - 1.0  # normalize to [-1, 1]

dataset = TensorDataset(torch.tensor(images))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Generator ---


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, IMG_SIZE),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# --- Init ---
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
criterion = nn.BCELoss()

os.makedirs("outputs", exist_ok=True)
fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE)

# --- Training ---
print(f"Training on {DEVICE} for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    d_losses, g_losses = [], []

    for (real,) in loader:
        real = real.to(DEVICE)
        bs = real.size(0)

        real_labels = torch.ones(bs, 1, device=DEVICE)
        fake_labels = torch.zeros(bs, 1, device=DEVICE)

        # Train Discriminator
        z = torch.randn(bs, LATENT_DIM, device=DEVICE)
        fake = G(z).detach()

        loss_D = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(bs, LATENT_DIM, device=DEVICE)
        fake = G(z)
        loss_G = criterion(D(fake), real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        d_losses.append(loss_D.item())
        g_losses.append(loss_G.item())

    print(f"Epoch {epoch}/{EPOCHS} | D loss: {np.mean(d_losses):.4f} | G loss: {np.mean(g_losses):.4f}")

    # Save sample images every 10 epochs
    if epoch % 10 == 0:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).cpu().numpy()
        G.train()

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            img = samples[i].reshape(28, 28)
            img = (img + 1) / 2  # back to [0, 1]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"outputs/epoch_{epoch:03d}.png")
        plt.close()
        print(f"  Saved outputs/epoch_{epoch:03d}.png")

torch.save(G.state_dict(), "generator.pth")
print("Generator saved to generator.pth")
print("Done.")
