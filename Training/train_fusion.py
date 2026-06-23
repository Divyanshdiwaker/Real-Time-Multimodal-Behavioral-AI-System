import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from backend.models.fusion_model import FusionNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# Load real embedding dataset
# ------------------------------------------------
# Use augmented dataset if available, otherwise fall back to original
if os.path.exists("fusion_data/fusion_X_augmented.npy"):
    X = np.load("fusion_data/fusion_X_augmented.npy")
    y = np.load("fusion_data/fusion_y_augmented.npy")
    print("✅ Using augmented dataset")
else:
    X = np.load("fusion_data/fusion_X.npy")
    y = np.load("fusion_data/fusion_y.npy")
    print("⚠️ Augmented dataset not found — using original")

print(f"Dataset: {len(X)} samples, X shape: {X.shape}, y shape: {y.shape}")
print(f"Stress labels      — 1: {np.sum(y[:,0]==1)}, 0: {np.sum(y[:,0]==0)}")
print(f"Incongruence labels — 1: {np.sum(y[:,1]==1)}, 0: {np.sum(y[:,1]==0)}")
# Normalize X using training stats
train_size_temp = int(0.8 * len(X))
train_mean = X[:train_size_temp].mean(axis=0)
train_std = X[:train_size_temp].std(axis=0)
X = (X - train_mean) / (train_std + 1e-8)

np.save("fusion_data/train_mean.npy", train_mean)
np.save("fusion_data/train_std.npy", train_std)
print("✅ Normalization applied and stats saved")
# ------------------------------------------------
# Dataset
# ------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = FusionDataset(X, y)

# 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"Train: {train_size} samples, Val: {val_size} samples")

# ------------------------------------------------
# Model — continue from best checkpoint
# ------------------------------------------------
fusion_model = FusionNet().to(DEVICE)

if os.path.exists("backend/models/fusion_model.pth"):
    fusion_model.load_state_dict(
        torch.load("backend/models/fusion_model.pth", map_location=DEVICE)
    )
    print("✅ Loaded existing model — continuing training...")
else:
    print("No existing model found — training from scratch...")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.0003)  # ✅ lower LR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 40  # ✅ more epochs

# ------------------------------------------------
# Training Loop
# ------------------------------------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):

    # ---- Train ----
    fusion_model.train()
    train_loss = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = fusion_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- Validate ----
    fusion_model.eval()
    stress_correct = 0
    align_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = torch.sigmoid(fusion_model(inputs))

            stress_preds = (outputs[:, 0] > 0.5).float()
            align_preds = (outputs[:, 1] > 0.5).float()

            stress_correct += (stress_preds == labels[:, 0]).sum().item()
            align_correct += (align_preds == labels[:, 1]).sum().item()
            total += labels.size(0)

    stress_acc = 100 * stress_correct / total
    align_acc = 100 * align_correct / total
    avg_acc = (stress_acc + align_acc) / 2

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss/len(train_loader):.4f} | "
        f"Stress Acc: {stress_acc:.1f}% | "
        f"Incongruence Acc: {align_acc:.1f}% | "
        f"Avg: {avg_acc:.1f}%"
    )

    # ---- Save best model ----
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        torch.save(fusion_model.state_dict(), "backend/models/fusion_model.pth")
        print(f"  ✅ Best model saved (avg acc: {avg_acc:.1f}%)")

    scheduler.step()

print(f"\nTraining complete! Best avg accuracy: {best_val_acc:.1f}%")
print("Model saved to backend/models/fusion_model.pth")

