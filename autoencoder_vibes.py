import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================
# Load MNIST
# =========================
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# =========================
# Autoencoder Model
# =========================
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

model = Autoencoder().to(device)

# =========================
# Loss + Optimizer
# =========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =========================
# Training Loop
# =========================
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.6f}")

print("Training complete ✨")

# =========================
# Visualize Results
# =========================
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

data_iter = iter(test_loader)
images, _ = next(data_iter)
images = images.view(images.size(0), -1).to(device)

with torch.no_grad():
    reconstructed = model(images)

images = images.view(-1, 1, 28, 28).cpu()
reconstructed = reconstructed.view(-1, 1, 28, 28).cpu()

fig, axes = plt.subplots(2, 8, figsize=(12, 4))

for i in range(8):
    axes[0][i].imshow(images[i].squeeze(), cmap="gray")
    axes[0][i].axis("off")
    axes[1][i].imshow(reconstructed[i].squeeze(), cmap="gray")
    axes[1][i].axis("off")

axes[0][0].set_title("Original")
axes[1][0].set_title("Reconstructed")

plt.show()
