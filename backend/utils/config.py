import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
FACE_FEATURES = 138  # 128 CNN + 10 MediaPipe landmarks
LANDMARK_FEATURES = 10
AUDIO_FEATURES = 64
TEXT_FEATURES = 128
TEMPERATURE = 1.5
MAX_MFCC_FRAMES = 173
N_MFCC = 40