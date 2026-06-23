import numpy as np
import torch
from faster_whisper import WhisperModel
import tempfile
import os
from sentence_transformers import SentenceTransformer

from backend.utils.video_utils import extract_face_features
from backend.services.stress_service import get_stress_features
from backend.services.sentiment_service import get_sentiment_score
from backend.models.fusion_model import FusionNet

from backend.utils.config import DEVICE, TEMPERATURE

# ------------------------------------------------
# Load models once at startup
# ------------------------------------------------
fusion_model = FusionNet().to(DEVICE)
fusion_model.load_state_dict(torch.load("backend/models/fusion_model.pth", map_location=DEVICE))
fusion_model.eval()

whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")  # ✅ medium for accent handling
text_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# ✅ Load training stats once at startup
train_mean = np.load("fusion_data/train_mean.npy")
train_std = np.load("fusion_data/train_std.npy")



async def analyze_session(audio_bytes: bytes, frame, last_known_emotion: list = None):

    # ✅ Safe temp file — bytes written directly, no stream re-read risk
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # ------------------------------------------------
        # Extract Emotion Features (with face detection)
        # ------------------------------------------------
        emotion_features = extract_face_features(frame)  # (128,) or None

        if emotion_features is not None:
            last_known_emotion = emotion_features.copy()
        elif last_known_emotion is not None:
            emotion_features = last_known_emotion
        else:
            emotion_features = np.zeros(128) # ✅ only zeros if nothing ever detected

        # ------------------------------------------------
        # Extract Stress Features
        # ------------------------------------------------
        stress_features = get_stress_features(tmp_path)  # (64,)

        # ------------------------------------------------
        # Whisper Transcription
        # ------------------------------------------------
        segments, _ = whisper_model.transcribe(
            tmp_path,
            language="en",
            task="transcribe",
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=1,
            vad_filter=True
        )
        text = " ".join([s.text for s in segments]).strip()

        # ------------------------------------------------
        # Sentiment Score
        # ------------------------------------------------
        sentiment_score = get_sentiment_score(text)

        # ------------------------------------------------
        # Text Embedding
        # ------------------------------------------------
        if text:
            text_embedding = text_model.encode(text)
            text_features = text_embedding[:128]   # (128,)
        else:
            text_features = np.zeros(128)

        # ------------------------------------------------
        # Combine Features
        # ------------------------------------------------
        fused = np.concatenate([
            emotion_features,
            stress_features,
            text_features
        ])  # (320,)

        # ✅ Normalize using training stats — same as app.py
        fused = (fused - train_mean) / (train_std + 1e-8)

        fused_tensor = torch.from_numpy(fused).float().unsqueeze(0).to(DEVICE)

        # ------------------------------------------------
        # Fusion Model Inference + Temperature Scaling
        # ------------------------------------------------
        with torch.no_grad():
            output = fusion_model(fused_tensor)           # (1, 2)
            output_scaled = output / TEMPERATURE          # ✅ temperature scaling
            probs = torch.sigmoid(output_scaled)[0]       # sigmoid for multi-label

        stress_score = float(probs[0])                    # output 0 = stress
        incongruency = float(probs[1])                    # output 1 = incongruence

        # ✅ Use sentiment to adjust stress score
        # negative sentiment → boost stress slightly
        # positive sentiment → reduce stress slightly
        sentiment_adjustment = (0.5 - sentiment_score) * 0.2
        stress_score = float(np.clip(stress_score + sentiment_adjustment, 0.0, 1.0))
        # Confidence = how decisive the model was, not how stressed the subject is
        # Use distance from 0.5 for both outputs — further from 0.5 = more confident
        stress_confidence = abs(stress_score - 0.5) * 2      # 0 = uncertain, 1 = certain
        incongruency_confidence = abs(incongruency - 0.5) * 2
        confidence = round(float(0.6 * stress_confidence + 0.4 * incongruency_confidence), 4)
        return {
            "stress": stress_score,
            "confidence": confidence,
            "behavioral_incongruency": incongruency,
            "transcript": text,
            "sentiment": sentiment_score               # ✅ now included
        }

    finally:
        os.remove(tmp_path)
