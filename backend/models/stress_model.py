import torch
import torch.nn as nn

class StressLSTM(nn.Module):
    def __init__(self):
        super(StressLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=40,     # MFCC features
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x, return_features=False):
        _, (h, _) = self.lstm(x)

        features = h[-1]  # 64-dim embedding
        output = self.fc(features)

        if return_features:
            return output, features

        return output


