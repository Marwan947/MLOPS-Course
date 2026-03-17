import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os

# --- Config ---
LATENT_DIM = 100
IMG_SIZE = 28 * 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Generator (must match main.py) ---


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


def generate(weights_path, n_images, output_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights found at '{weights_path}'. Train the model first with main.py.")

    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    G.eval()

    with torch.no_grad():
        z = torch.randn(n_images, LATENT_DIM, device=DEVICE)
        samples = G(z).cpu().numpy()

    cols = min(n_images, 8)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = [axes] if n_images == 1 else axes.flat

    for i, ax in enumerate(axes):
        if i < n_images:
            img = samples[i].reshape(28, 28)
            img = (img + 1) / 2  # [-1,1] -> [0,1]
            ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved {n_images} generated image(s) to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="generator.pth", help="Path to saved generator weights")
    parser.add_argument("--n", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output", default="outputs/generated.png", help="Output image path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate(args.weights, args.n, args.output)
