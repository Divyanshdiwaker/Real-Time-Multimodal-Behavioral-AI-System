import os
import mediapipe as mp
import cv2
import torch
import numpy as np
import librosa
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from backend.models.emotion_model import EmotionCNN
from backend.models.stress_model import StressLSTM
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------
# Load Models
# ------------------------------------------------
emotion_model = EmotionCNN().to(DEVICE)
emotion_model.load_state_dict(
    torch.load("backend/models/emotion_model.pth", map_location=DEVICE)
)
emotion_model.eval()

stress_model = StressLSTM().to(DEVICE)
stress_model.load_state_dict(
    torch.load("backend/models/stress_model.pth", map_location=DEVICE)
)
stress_model.eval()

text_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def extract_landmark_features_from_tensor(image_tensor):
    """
    Takes a (1, 48, 48) grayscale tensor, returns 10 landmark features.
    Returns zeros if no face detected.
    """
    # Convert to BGR for MediaPipe
    img = (image_tensor.squeeze().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return np.zeros(10)

    lm = results.multi_face_landmarks[0].landmark
    h, w = 48, 48

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

# ------------------------------------------------
# Emotion Embeddings
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

emotion_dataset = datasets.ImageFolder("Data/emotion/train", transform=transform)
emotion_loader = DataLoader(emotion_dataset, batch_size=64)

emotion_embeddings = []
emotion_labels = []

with torch.no_grad():
    for images, labels in emotion_loader:
        images = images.to(DEVICE)
        _, feats = emotion_model(images, return_features=True)
        cnn_feats = feats.cpu().numpy()

        # Extract landmark features for each image in batch
        landmark_feats = np.array([
            extract_landmark_features_from_tensor(images[i].cpu())
            for i in range(images.shape[0])
        ])

        # Concatenate CNN + landmarks
        combined = np.concatenate([cnn_feats, landmark_feats], axis=1)
        emotion_embeddings.append(combined)
        emotion_labels.append(labels.numpy())

emotion_embeddings = np.vstack(emotion_embeddings)
emotion_labels = np.hstack(emotion_labels)

print(f"Emotion embeddings: {emotion_embeddings.shape}")
print(f"Emotion label distribution: {np.bincount(emotion_labels)}")

# ------------------------------------------------
# Stress Embeddings (RAVDESS)
# ------------------------------------------------
stress_embeddings = []
stress_labels = []

AUDIO_PATH = "Data/audio"

for actor in os.listdir(AUDIO_PATH):
    actor_path = os.path.join(AUDIO_PATH, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        emotion_id = int(file.split("-")[2])
        stress_label = 1 if emotion_id in [5, 6, 7] else 0

        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < 173:
            pad = 173 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
        else:
            mfcc = mfcc[:, :173]

        tensor = torch.from_numpy(mfcc.T).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, feat = stress_model(tensor, return_features=True)

        stress_embeddings.append(feat.cpu().numpy()[0])
        stress_labels.append(stress_label)

stress_embeddings = np.array(stress_embeddings)
stress_labels = np.array(stress_labels)

print(f"Stress embeddings: {stress_embeddings.shape}")
print(f"Stress label distribution: {np.bincount(stress_labels)}")

# ------------------------------------------------
# Text Embeddings — rich dataset with labels
# ------------------------------------------------
stressed_texts = [
    "I am extremely stressed and overwhelmed.",
    "Everything is going wrong today.",
    "I can't handle this pressure anymore.",
    "I feel very anxious and nervous.",
    "This situation is terrible and frightening.",
    "I am angry and frustrated right now.",
    "I feel like I'm going to break down.",
    "Nothing is working out for me.",
    "I am exhausted and burnt out.",
    "I feel hopeless and desperate.",
    "This is too much for me to handle.",
    "I am scared and don't know what to do.",
    "I feel completely out of control.",
    "Everything feels overwhelming right now.",
    "I am panicking and can't calm down.",
    "I feel tense and on edge.",
    "I am worried about everything.",
    "I feel like crying right now.",
    "I am under enormous pressure.",
    "I feel trapped and helpless.",
    "My heart is racing and I can't breathe.",
    "I feel like everything is falling apart.",
    "I am dreading what comes next.",
    "I can't stop shaking, I'm so nervous.",
    "I feel sick with anxiety.",
    "Nothing I do seems to be good enough.",
    "I am overwhelmed by all my responsibilities.",
    "I feel like I'm drowning in stress.",
    "I can't think straight because of the pressure.",
    "Everything feels urgent and I can't cope.",
    "I am so frustrated I want to scream.",
    "I feel paralyzed by fear.",
    "I don't know how much more I can take.",
    "I am on the verge of a breakdown.",
    "My mind won't stop racing.",
    "I feel completely defeated.",
    "I am terrified of what might happen.",
    "I feel like no one understands my stress.",
    "The pressure is unbearable right now.",
    "I am struggling to keep it together.",
]

calm_texts = [
    "I feel happy and relaxed today.",
    "Everything is going well.",
    "I am calm and at peace.",
    "Life is good and I feel great.",
    "I feel confident and in control.",
    "Today is a wonderful day.",
    "I am comfortable and content.",
    "I feel positive and energetic.",
    "Things are going smoothly.",
    "I am grateful and satisfied.",
    "I feel focused and productive.",
    "Everything is under control.",
    "I am enjoying this moment.",
    "I feel balanced and centered.",
    "I am doing well and feeling good.",
    "I feel refreshed and motivated.",
    "Everything is peaceful and quiet.",
    "I am happy with how things are going.",
    "I feel safe and secure.",
    "I am in a good mood today.",
    "I feel light and free.",
    "Today has been a really good day.",
    "I am at ease with everything around me.",
    "I feel cheerful and optimistic.",
    "Everything is working out perfectly.",
    "I am enjoying the calmness of this moment.",
    "I feel strong and capable.",
    "I am looking forward to what comes next.",
    "I feel content with where I am in life.",
    "Everything feels manageable and clear.",
    "I am relaxed and breathing easily.",
    "I feel a deep sense of satisfaction.",
    "I am comfortable in my own skin.",
    "I feel joy and gratitude right now.",
    "Things are exactly as they should be.",
    "I am smiling and feeling great.",
    "I feel steady and grounded.",
    "I am peaceful and undisturbed.",
    "I feel clarity and purpose today.",
    "I am thriving and doing well.",
]

neutral_texts = [
    "Today is Wednesday.",
    "The meeting starts at 10.",
    "I am speaking right now.",
    "The weather is cloudy today.",
    "I need to buy some groceries.",
    "The report is due tomorrow.",
    "I will call you later.",
    "The project is in progress.",
    "I had lunch an hour ago.",
    "The traffic was normal today.",
    "I am sitting at my desk.",
    "The file has been saved.",
    "I need to send an email.",
    "The door is on the left.",
    "I will be there in five minutes.",
    "The screen is showing the results.",
    "I finished the task earlier.",
    "The system is running normally.",
    "I have a meeting at three.",
    "The document needs a signature.",
]

# Labels: 1 = stressed, 0 = calm/neutral
text_sentences = stressed_texts + calm_texts + neutral_texts
text_labels = (
    [1] * len(stressed_texts) +
    [0] * len(calm_texts) +
    [0] * len(neutral_texts)
)

text_embeddings = text_model.encode(text_sentences)
text_embeddings = text_embeddings[:, :128]
text_labels = np.array(text_labels)

print(f"Text embeddings: {text_embeddings.shape}")
print(f"Text labels — stressed={np.sum(text_labels==1)}, calm={np.sum(text_labels==0)}")

# ------------------------------------------------
# Save All Embeddings
# ------------------------------------------------
os.makedirs("fusion_data", exist_ok=True)

np.save("fusion_data/emotion_embeddings.npy", emotion_embeddings)
np.save("fusion_data/emotion_labels.npy", emotion_labels)

np.save("fusion_data/stress_embeddings.npy", stress_embeddings)
np.save("fusion_data/stress_labels.npy", stress_labels)

np.save("fusion_data/text_embeddings.npy", text_embeddings)
np.save("fusion_data/text_labels.npy", text_labels)  # ✅ now saved with labels

print("\nAll embeddings generated and saved successfully.")