import cv2
import os
import numpy as np
import torch
import torch
from backend.models.emotion_model import EmotionCNN
import mediapipe as mp

from backend.utils.config import DEVICE

model = EmotionCNN().to(DEVICE)

if not os.path.exists("backend/models/emotion_model.pth"):
    raise RuntimeError("emotion_model.pth not found — run train_emotion.py first")

model.load_state_dict(torch.load("backend/models/emotion_model.pth", map_location=DEVICE))
model.eval()

# Load OpenCV's built-in face detector
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

def extract_landmark_features(frame):
    """
    Extracts 10 landmark features from face:
    eye gaze (4), head pose (3), mouth openness (1), brow raise (2)
    Returns np.zeros(10) if no face detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return np.zeros(10)

    lm = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])

    # ---- Eye gaze (horizontal + vertical for each eye) ----
    # Left eye: outer=33, inner=133, top=159, bottom=145
    left_h = (pt(133)[0] - pt(33)[0])
    left_v = (pt(145)[1] - pt(159)[1])
    # Right eye: outer=362, inner=263, top=386, bottom=374
    right_h = (pt(263)[0] - pt(362)[0])
    right_v = (pt(374)[1] - pt(386)[1])

    # Normalize by eye width
    left_eye_w = max(abs(pt(133)[0] - pt(33)[0]), 1e-6)
    right_eye_w = max(abs(pt(263)[0] - pt(362)[0]), 1e-6)

    gaze_left_h = left_h / left_eye_w
    gaze_left_v = left_v / left_eye_w
    gaze_right_h = right_h / right_eye_w
    gaze_right_v = right_v / right_eye_w

    # ---- Head pose (pitch, yaw, roll approximation) ----
    nose_tip = pt(1)
    chin = pt(152)
    left_ear = pt(234)
    right_ear = pt(454)
    left_eye_pt = pt(159)
    right_eye_pt = pt(386)

    # Yaw — horizontal nose offset from face center
    face_center_x = (left_ear[0] + right_ear[0]) / 2
    yaw = (nose_tip[0] - face_center_x) / max(abs(right_ear[0] - left_ear[0]), 1e-6)

    # Pitch — vertical nose offset from eye-chin midpoint
    eye_mid_y = (left_eye_pt[1] + right_eye_pt[1]) / 2
    pitch = (nose_tip[1] - eye_mid_y) / max(abs(chin[1] - eye_mid_y), 1e-6)

    # Roll — eye line angle
    roll = (right_eye_pt[1] - left_eye_pt[1]) / max(abs(right_eye_pt[0] - left_eye_pt[0]), 1e-6)

    # ---- Mouth openness ----
    # Top lip=13, bottom lip=14
    mouth_top = pt(13)
    mouth_bottom = pt(14)
    mouth_left = pt(78)
    mouth_right = pt(308)
    mouth_open = abs(mouth_bottom[1] - mouth_top[1]) / max(abs(mouth_right[0] - mouth_left[0]), 1e-6)

    # ---- Brow raise ----
    # Left brow top=105, left eye top=159
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

    # Clip to reasonable range
    features = np.clip(features, -1.0, 1.0)

    return features  # shape (10,)



def extract_face_features(frame):
    """
    Detects face in frame, crops it, and returns
    128-dim feature vector from EmotionCNN.
    Returns None if no face detected.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------
    # Face Detection
    # ------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None  # ✅ Explicitly signal no face found

    # Take the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # ------------------------------------------------
    # Crop & Preprocess
    # ------------------------------------------------
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))

    tensor = torch.tensor(face).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    # ------------------------------------------------
    # Extract Features via EmotionCNN
    # ------------------------------------------------
    with torch.no_grad():
        _, features = model(tensor, return_features=True)

    cnn_features = features.cpu().numpy()[0]  # shape: (128,)
    landmark_features = extract_landmark_features(frame)  # shape: (10,)
    return np.concatenate([cnn_features, landmark_features])  # shape: (138,)  # shape: (128,)
