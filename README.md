# MNIST Autoencoder — 784 → 32

A minimal PyTorch implementation of an Autoencoder trained on MNIST.

This project demonstrates neural network–based compression by encoding 28×28 handwritten digit images (784 pixels) into a 32-dimensional latent vector and reconstructing them back to the original image.

---

##  What This Project Does

- Loads the MNIST dataset
- Flattens 28×28 images into 784-dim vectors
- Trains an Autoencoder:
  - Encoder: 784 → 128 → 32
  - Decoder: 32 → 128 → 784
- Minimizes reconstruction loss (MSE)
- Visualizes original vs reconstructed digits

> You are literally watching a neural network learn to compress and redraw handwritten digits.

---

##  Architecture
```text
Input (784)
   ↓
Linear(784 → 128)
ReLU
Linear(128 → 32)
   ↓
Latent Space (32 dimensions)
   ↓
Linear(32 → 128)
ReLU
Linear(128 → 784)
Sigmoid
   ↓
Reconstructed Image
```

---

##  Installation

Make sure you have Python 3.8+. Install dependencies:
```bash
pip install torch torchvision matplotlib
```

---

##  How to Run

From the project directory:
```bash
python autoencoder.py
```

**What will happen:**

1. MNIST downloads (first run only)
2. Model trains for 10 epochs
3. Loss prints each epoch
4. A matplotlib window shows:
   - Top row → Original digits
   - Bottom row → Reconstructed digits

> If the bottom digits look blurry but recognizable, it worked.

---

##  Loss Function

**Mean Squared Error (MSE):**
```
MSE(original, reconstruction)
```

This measures how close the reconstructed image is to the original.

---

##  Experiments You Can Try

###  Increase Compression

Change latent size from 32 to 8:
```python
nn.Linear(128, 8)
```

How much information survives?

###  Make It Deeper

Add extra layers:
```python
nn.Linear(784, 256)
nn.ReLU()
nn.Linear(256, 128)
nn.ReLU()
nn.Linear(128, 32)
```

Compare reconstruction quality.

###  Visualize Latent Space

Change latent dimension to 2 and plot encodings by digit label. You'll see clustering emerge.

---

##  What You're Learning

This project teaches:

- Representation learning
- Neural network compression
- Latent space concepts
- Training loops in PyTorch
- Image reconstruction

This is the foundation for:

- Variational Autoencoders (VAEs)
- GANs
- Diffusion models
- Modern generative AI

---

##  Project Structure
```
.
├── autoencoder.py
├── README.md
└── data/          ← auto-downloaded MNIST
```

---

##  Next Steps

After this, try:

- Convolutional Autoencoder
- Denoising Autoencoder
- Variational Autoencoder (VAE)
- GAN on MNIST

---

