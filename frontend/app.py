import streamlit as st
import cv2
import torch
import numpy as np
import librosa
import sounddevice as sd
import threading
from faster_whisper import WhisperModel
import time
import mediapipe as mp
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from sentence_transformers import SentenceTransformer
from backend.models.emotion_model import EmotionCNN
from backend.models.stress_model import StressLSTM
from backend.models.fusion_model import FusionNet
from backend.session.session_manager import SessionManager
from backend.session.session_logger import SessionLogger
from backend.session.report_generator import generate_report
from backend.services.sentiment_service import get_sentiment_score

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Real-Time Behavioral AI System")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 1.5
BASELINE_FRAMES = 100
VAD_THRESHOLD = 0.03  # ✅ voice activity detection threshold
DEEPGRAM_API_KEY = "885f2abfae5e89addbfa25e84a74186803984f3e"

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# ✅ Common Whisper hallucinations to filter
HALLUCINATIONS = [
    # Single words
    "you", "thanks", "bye", "goodbye", "okay", "ok",
    "hmm", "uh", "um", "ah", "oh", "the", "a",
    "i", "he", "she", "we", "they", "it",

    # Classic Whisper silence hallucinations
    "thank you", "thank you.",
    "thanks for watching", "thanks for watching.",
    "so this is it", "so this is it.",
    "so this is it, thanks for watching.",
    "like and subscribe", "please subscribe",
    "don't forget to subscribe",
    "see you next time", "see you in the next video",
    "thank you for watching", "thank you for watching.",
    "i'll see you in the next one",
    "thanks for watching and i'll see you in the next one",
    "subscribe", "subtitles by", "captions by",
    "translation by", "transcription by",
]

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def is_speech(audio, threshold=VAD_THRESHOLD):
    """Returns True only if audio contains actual speech"""
    rms = np.sqrt(np.mean(audio ** 2))
    return rms > threshold

def is_valid_transcription(text):
    """Filter out Whisper hallucinations"""
    cleaned = text.strip().lower().strip(".,!?")

    if len(cleaned) < 3:
        return False

    if len(cleaned.split()) < 2:
        return False

    # ✅ Exact match check
    if cleaned in HALLUCINATIONS:
        return False

    # ✅ Partial match check — catches variations
    for h in HALLUCINATIONS:
        if h in cleaned and len(h) > 10:  # only partial match long phrases
            return False

    # ✅ Filter repetitive single word repeated
    words = cleaned.split()
    if len(set(words)) == 1:  # all same word
        return False

    return True

def extract_landmark_features(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return np.zeros(10)

    lm = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    left_eye_w = max(abs(pt(133)[0] - pt(33)[0]), 1e-6)
    right_eye_w = max(abs(pt(263)[0] - pt(362)[0]), 1e-6)

    gaze_left_h = (pt(133)[0] - pt(33)[0]) / left_eye_w
    gaze_left_v = (pt(145)[1] - pt(159)[1]) / left_eye_w
    gaze_right_h = (pt(263)[0] - pt(362)[0]) / right_eye_w
    gaze_right_v = (pt(374)[1] - pt(386)[1]) / right_eye_w

    nose_tip = pt(1)
    chin = pt(152)
    left_ear = pt(234)
    right_ear = pt(454)
    left_eye_pt = pt(159)
    right_eye_pt = pt(386)

    face_center_x = (left_ear[0] + right_ear[0]) / 2
    yaw = (nose_tip[0] - face_center_x) / max(abs(right_ear[0] - left_ear[0]), 1e-6)
    eye_mid_y = (left_eye_pt[1] + right_eye_pt[1]) / 2
    pitch = (nose_tip[1] - eye_mid_y) / max(abs(chin[1] - eye_mid_y), 1e-6)
    roll = (right_eye_pt[1] - left_eye_pt[1]) / max(abs(right_eye_pt[0] - left_eye_pt[0]), 1e-6)

    mouth_top = pt(13)
    mouth_bottom = pt(14)
    mouth_left = pt(78)
    mouth_right = pt(308)
    mouth_open = abs(mouth_bottom[1] - mouth_top[1]) / max(abs(mouth_right[0] - mouth_left[0]), 1e-6)

    left_brow = pt(105)
    right_brow = pt(334)
    brow_left = (left_eye_pt[1] - left_brow[1]) / max(h, 1e-6)
    brow_right = (right_eye_pt[1] - right_brow[1]) / max(h, 1e-6)

    features = np.array([
        gaze_left_h, gaze_left_v,
        gaze_right_h, gaze_right_v,
        yaw, pitch, roll,
        mouth_open,
        brow_left, brow_right
    ], dtype=np.float32)

    return np.clip(features, -1.0, 1.0)


def deepgram_worker(stop_event, session_state):
    """Dedicated thread for Deepgram real-time transcription display only"""
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.live.v("1")

        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if sentence.strip():
                session_state.deepgram_transcript = sentence

        def on_error(self, error, **kwargs):
            print(f"[DEEPGRAM] Error: {error}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-IN",  # Indian English
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
        )

        dg_connection.start(options)

        samplerate = 16000
        chunk_duration = 0.25  # send 250ms chunks for low latency
        chunk_samples = int(samplerate * chunk_duration)

        while not stop_event.is_set():
            audio = sd.rec(chunk_samples, samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()
            dg_connection.send(audio.tobytes())

        dg_connection.finish()

    except Exception as e:
        print(f"[DEEPGRAM] Worker error: {e}")

# -------------------------------------------------
# LOAD MODELS (cached — only loads once)
# -------------------------------------------------
@st.cache_resource
def load_models():
    emotion = EmotionCNN().to(DEVICE)
    emotion.load_state_dict(torch.load("backend/models/emotion_model.pth", map_location=DEVICE))
    emotion.eval()

    stress = StressLSTM().to(DEVICE)
    stress.load_state_dict(torch.load("backend/models/stress_model.pth", map_location=DEVICE))
    stress.eval()

    fusion = FusionNet().to(DEVICE)
    fusion.load_state_dict(torch.load("backend/models/fusion_model.pth", map_location=DEVICE))
    fusion.eval()

    text_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8") # ✅ medium for better accent handling

    train_mean = np.load("fusion_data/train_mean.npy")
    train_std = np.load("fusion_data/train_std.npy")

    return emotion, stress, fusion, text_model, whisper_model, train_mean, train_std 

emotion_model, stress_model, fusion_model, text_model, whisper_model, train_mean, train_std = load_models()

# -------------------------------------------------
# FACE DETECTOR
# -------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------------------------
# THREAD SAFE AUDIO STATE
# -------------------------------------------------
class AudioState:
    def __init__(self):
        self.latest_text = ""
        self.latest_stress_feat = np.zeros(64)
        self.lock = threading.Lock()

    def update(self, text, stress_feat):
        with self.lock:
            self.latest_text = text
            self.latest_stress_feat = stress_feat.copy()

    def get(self):
        with self.lock:
            return self.latest_text, self.latest_stress_feat.copy()

# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "smooth_stress" not in st.session_state:
    st.session_state.smooth_stress = 0.0

if "smooth_align" not in st.session_state:
    st.session_state.smooth_align = 0.0

if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()
    st.session_state.session_logger = SessionLogger(st.session_state.session_manager)
    st.session_state.session_active = False

if "audio_state" not in st.session_state:
    st.session_state.audio_state = AudioState()

if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

if "last_emotion_feat" not in st.session_state:
    st.session_state.last_emotion_feat = None
   

if "baseline_stress" not in st.session_state:
    st.session_state.baseline_stress = None

if "baseline_align" not in st.session_state:
    st.session_state.baseline_align = None

if "baseline_samples" not in st.session_state:
    st.session_state.baseline_samples = []

if "baseline_ready" not in st.session_state:
    st.session_state.baseline_ready = False

if "deepgram_transcript" not in st.session_state:
    st.session_state.deepgram_transcript = ""

# -------------------------------------------------
# AUDIO WORKER
# -------------------------------------------------


def audio_worker(audio_state, stop_event):
    while not stop_event.is_set():
        try:
            duration = 3
            samplerate = 16000

            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1
            )
            sd.wait()

            y_audio = audio.flatten().astype(np.float32)
            y_audio = np.nan_to_num(y_audio)

            max_val = np.max(np.abs(y_audio))
            if max_val > 0:
                y_audio = y_audio / max_val

            # ✅ Always extract stress features regardless of speech
            mfcc = librosa.feature.mfcc(y=y_audio, sr=samplerate, n_mfcc=40)
            if mfcc.shape[1] < 173:
                pad = 173 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
            else:
                mfcc = mfcc[:, :173]

            audio_tensor = torch.tensor(mfcc.T).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, stress_feat_tensor = stress_model(audio_tensor, return_features=True)
            stress_feat = stress_feat_tensor.cpu().numpy()[0]

            # ✅ Transcribe only if speech detected
            segments, _ = whisper_model.transcribe(
                y_audio,
                language="en",
                task="transcribe",
                condition_on_previous_text=False,
                temperature=0.0,
                beam_size=1,
                vad_filter=True,
                initial_prompt="This is a behavioral analysis interview. The subject is speaking clearly in English."
            )
            text = " ".join([s.text for s in segments]).strip()

            # ✅ Filter hallucinations
            if is_valid_transcription(text):
                print(f"[AUDIO] Transcribed: '{text}'")
                audio_state.update(text, stress_feat)
            else:
                if text.strip():  # only log if something was actually filtered
                    print(f"[AUDIO] Filtered: '{text}'")
                old_text, _ = audio_state.get()
                audio_state.update(old_text, stress_feat)

        except Exception as e:
            import traceback
            print("Audio error:", e)
            traceback.print_exc()

# -------------------------------------------------
# UI LAYOUT
# -------------------------------------------------
col1, col2 = st.columns([1, 1.5])

with col1:
    camera_toggle = st.toggle("Enable Camera")
    session_mode = st.toggle("Enable Session Logging")

frame_placeholder = col2.empty()

metric1, metric2, metric3 = col1.columns(3)
stress_metric = metric1.empty()
confidence_metric = metric2.empty()
align_metric = metric3.empty()

alignment_color_placeholder = col1.empty()
baseline_placeholder = col1.empty()
speech_placeholder = col1.empty()


# -------------------------------------------------
# CAMERA CONTROL
# -------------------------------------------------
if camera_toggle and not st.session_state.camera_running:
    st.session_state.camera_running = True
    st.session_state.stop_event.clear()

    # Reset baseline on every new camera start
    st.session_state.baseline_stress = None
    st.session_state.baseline_align = None
    st.session_state.baseline_samples = []
    st.session_state.baseline_ready = False
    st.session_state.smooth_stress = 0.0
    st.session_state.smooth_align = 0.0

    threading.Thread(
        target=audio_worker,
        args=(st.session_state.audio_state, st.session_state.stop_event),
        daemon=True
    ).start()
    print("Audio thread started!")

    threading.Thread(
        target=deepgram_worker,
        args=(st.session_state.stop_event, st.session_state),
        daemon=True
    ).start()
    print("Deepgram thread started!")

if not camera_toggle and st.session_state.camera_running:
    st.session_state.camera_running = False
    st.session_state.stop_event.set()
    st.session_state.last_emotion_feat = None

# -------------------------------------------------
# SESSION CONTROL
# -------------------------------------------------
if session_mode and not st.session_state.session_active:
    st.session_state.session_manager.start_session()
    st.session_state.session_active = True

if not session_mode and st.session_state.session_active:
    session = st.session_state.session_manager.end_session()
    if session is not None:
        generate_report(session)
    st.session_state.session_active = False

# -------------------------------------------------
# CAMERA LOOP
# -------------------------------------------------
if st.session_state.camera_running:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Could not open camera. Please check your camera connection.")
        st.session_state.camera_running = False

    else:
        while st.session_state.camera_running:

            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from camera.")
                break

            frame = cv2.resize(frame, (480, 360))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ✅ Sensitive face detection
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(15, 15)
            )

            emotion_feat = np.zeros(138)

            for (x, y, w, h) in faces:
                face_crop = gray[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (48, 48))

                img_tensor = torch.from_numpy(face_crop).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

                with torch.no_grad():
                    emotion_output, emotion_feat_tensor = emotion_model(
                        img_tensor, return_features=True
                    )

                cnn_feat = emotion_feat_tensor.cpu().numpy()[0]  # 128
                landmark_feat = extract_landmark_features(frame)  # 10
                emotion_feat = np.concatenate([cnn_feat, landmark_feat])  # 138
                st.session_state.last_emotion_feat = emotion_feat.copy()

                pred_class = torch.argmax(emotion_output, dim=1).item()
                emotion_text = emotion_labels[pred_class]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

            # ✅ Use last known emotion if no face detected
            if emotion_feat.std() == 0 and st.session_state.last_emotion_feat is not None:
                emotion_feat = st.session_state.last_emotion_feat

            # ------------------------------------------------
            # Read from thread safe audio state
            # ------------------------------------------------
            latest_text, latest_stress_feat = st.session_state.audio_state.get()

            # ------------------------------------------------
            # Text Features
            # ------------------------------------------------
            if latest_text.strip() != "":
                text_embedding = text_model.encode(latest_text)
                text_feat = text_embedding[:128]
            else:
                text_feat = np.zeros(128)

            # ------------------------------------------------
            # Fuse + Normalize
            # ------------------------------------------------
            fused = np.concatenate([
                emotion_feat,
                latest_stress_feat,
                text_feat
            ])

            fused = (fused - train_mean) / (train_std + 1e-8)
            fused_tensor = torch.tensor(fused).float().unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # Fusion Inference + Temperature Scaling
            # ------------------------------------------------
            with torch.no_grad():
                output = fusion_model(fused_tensor)
                output_scaled = output / TEMPERATURE
                probs = torch.sigmoid(output_scaled)[0]

            stress_raw = float(probs[0])

            sentiment_score = get_sentiment_score(latest_text)
            sentiment_adjustment = (0.5 - sentiment_score) * 0.2
            stress_raw = float(np.clip(stress_raw + sentiment_adjustment, 0.0, 1.0))

            align_raw = float(probs[1])

            # ------------------------------------------------
            # Dynamic Baseline
            # ------------------------------------------------
            if not st.session_state.baseline_ready:
                st.session_state.baseline_samples.append((stress_raw, align_raw))

                remaining = BASELINE_FRAMES - len(st.session_state.baseline_samples)
                baseline_placeholder.info(
                    f"🔄 Calibrating baseline... {remaining} frames remaining. "
                    f"Please sit calm and relaxed."
                )

                if len(st.session_state.baseline_samples) >= BASELINE_FRAMES:
                    samples = st.session_state.baseline_samples
                    st.session_state.baseline_stress = np.mean([s[0] for s in samples])
                    st.session_state.baseline_align = np.mean([s[1] for s in samples])
                    st.session_state.baseline_ready = True
                    print(f"[BASELINE] stress={st.session_state.baseline_stress:.3f} align={st.session_state.baseline_align:.3f}")

            else:
                baseline_placeholder.success("✅ Baseline calibrated — monitoring active")

                stress_raw = max(0.0, min(1.0,
                    stress_raw - (st.session_state.baseline_stress - 0.3)
                ))
                align_raw = max(0.0, min(1.0,
                    align_raw - (st.session_state.baseline_align - 0.3)
                ))

            # ------------------------------------------------
            # Smoothing
            # ------------------------------------------------
            alpha = 0.2
            st.session_state.smooth_stress = (
                alpha * stress_raw + (1 - alpha) * st.session_state.smooth_stress
            )
            st.session_state.smooth_align = (
                alpha * align_raw + (1 - alpha) * st.session_state.smooth_align
            )

            stress_val = st.session_state.smooth_stress
            align_val = st.session_state.smooth_align
            stress_confidence = abs(stress_val - 0.5) * 2
            incongruency_confidence = abs(align_val - 0.5) * 2
            confidence_val = round(0.6 * stress_confidence + 0.4 * incongruency_confidence, 4)

            # ------------------------------------------------
            # Session Logging
            # ------------------------------------------------
            if st.session_state.session_active and st.session_state.baseline_ready:
                st.session_state.session_logger.log({
                    "timestamp": time.time(),
                    "stress_score": stress_val,
                    "align_score": align_val,
                    "confidence": confidence_val,
                    "text": latest_text
                })

            # ------------------------------------------------
            # Incongruency Color
            # ------------------------------------------------
            if align_val < 0.30:
                color = "green"
                status = "Low Incongruence"
            elif align_val < 0.50:
                color = "orange"
                status = "Moderate Incongruence"
            else:
                color = "red"
                status = "High Incongruence"

            # ------------------------------------------------
            # UI Updates
            # ------------------------------------------------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            stress_metric.metric("Stress", round(stress_val, 3))
            confidence_metric.metric("Confidence", round(confidence_val, 3))
            align_metric.metric("Behavioral Incongruence", round(align_val, 3))

            alignment_color_placeholder.markdown(
                f"<h3 style='color:{color};'>{status}</h3>",
                unsafe_allow_html=True
            )

            deepgram_text = st.session_state.get("deepgram_transcript", "")
            if deepgram_text.strip():
                speech_placeholder.markdown(
                    f"**🎤 Detected Speech:** {deepgram_text}"
                )
            elif latest_text.strip():
                speech_placeholder.markdown(
                    f"**🎤 Detected Speech:** {latest_text}"
                )
            else:
                speech_placeholder.markdown("🎤 *Waiting for speech...*")


        cap.release()







