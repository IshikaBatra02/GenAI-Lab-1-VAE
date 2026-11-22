"""Train a 2-D Variational Autoencoder on MNIST."""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, 128), nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 400), nn.ReLU(),
            nn.Linear(400, 784), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z).view(-1, 1, 28, 28)
        return reconstruction, mu, logvar


def loss_function(recon, x, mu, logvar):
    reconstruction = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction + kl, reconstruction, kl


@torch.no_grad()
def save_latent_plot(model, loader, device, path):
    model.eval()
    points, labels = [], []
    for x, y in loader:
        mu, _ = model.encode(x.to(device))
        points.append(mu.cpu().numpy())
        labels.append(y.numpy())
    points = np.concatenate(points)
    labels = np.concatenate(labels)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=5, alpha=.65)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("MNIST latent space (encoder means)")
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@torch.no_grad()
def save_prior_samples(model, device, path, count=64):
    model.eval()
    z = torch.randn(count, 2, device=device)
    samples = model.decoder(z).view(-1, 1, 28, 28).cpu()
    save_image(samples, path, nrow=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="lab_outputs/vae")
    args = parser.parse_args()
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = VAE(latent_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = recon_total = kl_total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()
            recon_total += recon_loss.item()
            kl_total += kl_loss.item()
        n = len(train_data)
        print(f"Epoch {epoch}/{args.epochs} | loss {total/n:.4f} | recon {recon_total/n:.4f} | KL {kl_total/n:.4f}")

    model.eval()
    x, _ = next(iter(test_loader))
    with torch.no_grad():
        recon, _, _ = model(x.to(device))
    comparison = torch.cat([x[:8], recon.cpu()[:8]])
    save_image(comparison, out / "reconstructions.png", nrow=8)
    save_prior_samples(model, device, out / "prior_samples.png")
    save_latent_plot(model, test_loader, device, out / "latent_space.png")
    torch.save(model.state_dict(), out / "vae_mnist.pt")
    print(f"Saved model and visualizations to {out.resolve()}")


if __name__ == "__main__":
    main()
