import cv2
import torch
import numpy as np
from backend.models.emotion_model import EmotionCNN

from backend.utils.config import DEVICE

# Load model once at startup
model = EmotionCNN().to(DEVICE)
model.load_state_dict(torch.load("backend/models/emotion_model.pth", map_location=DEVICE))
model.eval()



