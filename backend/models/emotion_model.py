import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        # 48 → 24 → 12 → 6
        # 128 * 6 * 6 = 4608
        self.embedding = nn.Linear(4608, 128)

        self.classifier = nn.Linear(128, 7)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.flatten(x)

        features = self.embedding(x)
        output = self.classifier(features)

        if return_features:
            return output, features

        return output




