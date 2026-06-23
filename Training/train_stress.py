import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from backend.models.stress_model import StressLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_PATH = "Data/audio"  # Change if needed

# ------------------------------------------------
# Dataset
# ------------------------------------------------
class RAVDESSDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []

        for actor in os.listdir(root_dir):
            actor_path = os.path.join(root_dir, actor)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith(".wav"):
                        self.files.append(os.path.join(actor_path, file))

    def __len__(self):
        return len(self.files)

    def extract_label(self, filename):
        try:
            parts = filename.split("-")
            if len(parts) < 3:
                print(f"⚠️ Skipping unexpected filename format: {filename}")
                return None
            emotion_id = int(parts[2])
            if emotion_id in [5, 6, 7]:  # angry, fear, disgust
                return 1
            return 0
        except (ValueError, IndexError):
            print(f"⚠️ Could not parse label from: {filename}")
            return None

    def __getitem__(self, idx):
        file_path = self.files[idx]
        filename = os.path.basename(file_path)

        y, sr = librosa.load(file_path, sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Pad or truncate to 173 frames
        if mfcc.shape[1] < 173:
            pad_width = 173 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :173]

        mfcc = mfcc.T  # (173, 40)

        label = self.extract_label(filename)

        if label is None:
            # Return zeros as fallback — will be filtered by DataLoader
            return torch.zeros(173, 40), torch.tensor([-1], dtype=torch.float32)

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)


# ------------------------------------------------
# Training
# ------------------------------------------------
dataset = RAVDESSDataset(AUDIO_PATH)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = StressLSTM().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for mfcc, labels in loader:
        # Skip invalid samples
        valid = (labels[:, 0] != -1)
        if not valid.any():
            continue
        mfcc = mfcc[valid].to(DEVICE)
        labels = labels[valid].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels.bool()).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {total_loss/len(loader):.4f} "
          f"Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "backend/models/stress_model.pth")
print("Stress model saved successfully.")
