import librosa
import numpy as np

from backend.utils.config import SAMPLE_RATE, N_MFCC, MAX_MFCC_FRAMES

def extract_mfcc(path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_frames=MAX_MFCC_FRAMES):
    """
    Loads audio from path and extracts MFCC features.
    Returns shape: (max_frames, n_mfcc) — ready for StressLSTM
    """
    y, sr = librosa.load(path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or trim to fixed length
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc.T  # (173, 40)