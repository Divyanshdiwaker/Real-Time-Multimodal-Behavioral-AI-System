import torch
import torch.nn as nn
import numpy as np

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        # 🔹 Project stress (64 → 128)
        self.stress_proj = nn.Linear(64, 128)

        # 🔹 Project emotion (138 → 128)
        self.emotion_proj = nn.Linear(138, 128)

        # 🔹 Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(330, 3),
            nn.Softmax(dim=1)
        )

        # 🔹 Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        # Split features
        emotion = x[:, :138]            # 128 CNN + 10 landmarks
        stress = x[:, 138:202]          # 64
        text = x[:, 202:]               # 128

        # Project emotion to 128
        emotion = self.emotion_proj(emotion)

        # Project stress to 128
        stress = self.stress_proj(stress)

        # Compute gates
        gates = self.gate(x)

        g1 = gates[:, 0].unsqueeze(1)
        g2 = gates[:, 1].unsqueeze(1)
        g3 = gates[:, 2].unsqueeze(1)

        # Gated fusion (all 128 now)
        fused = g1 * emotion + g2 * stress + g3 * text

        output = self.classifier(fused)

        return output

print("\n" + "=" * 50)
print("FUSION LABEL BALANCE (per column)")
print("=" * 50)

for fname, path in [("fusion_y", "fusion_data/fusion_y.npy"), 
                     ("fusion_y_augmented", "fusion_data/fusion_y_augmented.npy")]:
    try:
        y = np.load(path)
        print(f"\n{fname}:")
        
        # Stress (column 0)
        stress_0 = np.sum(y[:, 0] == 0)
        stress_1 = np.sum(y[:, 0] == 1)
        stress_pct = stress_1 / len(y) * 100
        print(f"  Stress       — 0: {stress_0} | 1: {stress_1} | {stress_pct:.1f}% stressed")
        
        # Incongruence (column 1)
        inc_0 = np.sum(y[:, 1] == 0)
        inc_1 = np.sum(y[:, 1] == 1)
        inc_pct = inc_1 / len(y) * 100
        print(f"  Incongruence — 0: {inc_0} | 1: {inc_1} | {inc_pct:.1f}% incongruent")
        
        # Both high
        both = np.sum((y[:, 0] == 1) & (y[:, 1] == 1))
        print(f"  Both high    — {both} samples ({both/len(y)*100:.1f}%)")
        
        # Both low
        neither = np.sum((y[:, 0] == 0) & (y[:, 1] == 0))
        print(f"  Both low     — {neither} samples ({neither/len(y)*100:.1f}%)")

    except FileNotFoundError:
        print(f"❌ {fname} — NOT FOUND")