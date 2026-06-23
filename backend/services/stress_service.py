import torch
from backend.models.stress_model import StressLSTM
from backend.utils.audio_utils import extract_mfcc

from backend.utils.config import DEVICE

# Load model once at startup
model = StressLSTM().to(DEVICE)
model.load_state_dict(torch.load("backend/models/stress_model.pth", map_location=DEVICE))
model.eval()

def get_stress_features(path):
    mfcc = extract_mfcc(path)  # (173, 40)

    tensor = torch.tensor(mfcc).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, features = model(tensor, return_features=True)

    return features.cpu().numpy()[0]  # (64,)
