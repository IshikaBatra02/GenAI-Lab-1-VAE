# Lab 1 — Variational Autoencoder (VAE) on MNIST

Implements the VAE experiment from the Generative AI lab manual.

## Experiment
- Dataset: MNIST (28×28 grayscale digits)
- Latent dimension: 2
- Objective: reconstruction loss + KL-divergence
- Default training: 5 epochs
- Outputs: reconstruction grid, prior samples, latent-space scatter plot

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python vae_mnist.py --epochs 5 --batch-size 128
```

Outputs are saved under `lab_outputs/vae/`. The script uses a fixed seed (42) for reproducibility. A CUDA device is used automatically when available.

## Model
The encoder maps an image to a mean and log-variance vector. The reparameterization trick samples a 2-D latent vector, and the decoder reconstructs the image. The loss is binary cross-entropy reconstruction loss plus the KL divergence to a unit Gaussian prior.

## Notes
The first run downloads MNIST. Training results are generated when you run the script; no precomputed metrics are claimed.
