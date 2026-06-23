import numpy as np
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------
# Load embeddings
# ------------------------------------------------
emotion_embeddings = np.load("fusion_data/emotion_embeddings.npy")
emotion_labels = np.load("fusion_data/emotion_labels.npy")

stress_embeddings = np.load("fusion_data/stress_embeddings.npy")
stress_labels = np.load("fusion_data/stress_labels.npy")

text_embeddings = np.load("fusion_data/text_embeddings.npy")
text_labels = np.load("fusion_data/text_labels.npy")  # ✅ load text labels

print(f"Emotion embeddings: {emotion_embeddings.shape}")
print(f"Stress embeddings:  {stress_embeddings.shape}")
print(f"Text embeddings:    {text_embeddings.shape}")

# ------------------------------------------------
# High arousal emotion indices
# Angry=0, Disgust=1, Fear=2
# ------------------------------------------------
high_arousal = [0, 1, 2]

fusion_X = []
fusion_y = []

num_samples = 20000
target_stress_0 = num_samples // 2
target_stress_1 = num_samples // 2
target_inc_0 = num_samples // 2
target_inc_1 = num_samples // 2

stress_0_count = 0
stress_1_count = 0

# Target 50/50 split for incongruence
half = num_samples // 2
congruent_count = 0
incongruent_count = 0

for _ in range(num_samples * 10):  # extra iterations to fill balanced classes

    # Randomly sample embeddings
    e_idx = random.randint(0, len(emotion_embeddings) - 1)
    s_idx = random.randint(0, len(stress_embeddings) - 1)
    t_idx = random.randint(0, len(text_embeddings) - 1)

    e_embed = emotion_embeddings[e_idx]
    e_label = int(emotion_labels[e_idx])

    s_embed = stress_embeddings[s_idx]
    s_label = int(stress_labels[s_idx])

    t_embed = text_embeddings[t_idx]
    t_label = int(text_labels[t_idx])  # ✅ use real text label

    # ------------------------------------------------
    # Stress label
    # stressed if audio OR text is stressed
    # ------------------------------------------------
    stress = 1 if (s_label == 1 or t_label == 1) else 0

    # ------------------------------------------------
    # Incongruence label
    # all 3 modalities must agree to be congruent
    # any disagreement = incongruent
    # ------------------------------------------------
    # ------------------------------------------------
    # Incongruence label
    # all 3 modalities must agree to be congruent
    # any disagreement = incongruent
    # ------------------------------------------------
    emotion_stressed = 1 if e_label in high_arousal else 0
    signals = [emotion_stressed, s_label, t_label]
    incongruency = 0 if (signals[0] == signals[1] == signals[2]) else 1

    # Skip to balance both stress and incongruence labels
    if stress == 0 and stress_0_count >= target_stress_0:
        continue
    if stress == 1 and stress_1_count >= target_stress_1:
        continue
    if incongruency == 0 and congruent_count >= target_inc_0:
        continue
    if incongruency == 1 and incongruent_count >= target_inc_1:
        continue

    if stress == 0:
        stress_0_count += 1
    else:
        stress_1_count += 1

    if incongruency == 0:
        congruent_count += 1
    else:
        incongruent_count += 1

    fused = np.concatenate([e_embed, s_embed, t_embed])  # 138 + 64 + 128 = 330

    fusion_X.append(fused)
    fusion_y.append([stress, incongruency])

fusion_X = np.array(fusion_X)
fusion_y = np.array(fusion_y, dtype=np.int32)

os.makedirs("fusion_data", exist_ok=True)

np.save("fusion_data/fusion_X.npy", fusion_X)
np.save("fusion_data/fusion_y.npy", fusion_y)

print("\nFusion dataset generated successfully!")
print(f"X shape: {fusion_X.shape}")
print(f"Y shape: {fusion_y.shape}")
print(f"Stress       — 1: {np.sum(fusion_y[:,0]==1)}, 0: {np.sum(fusion_y[:,0]==0)}")
print(f"Incongruence — 1: {np.sum(fusion_y[:,1]==1)}, 0: {np.sum(fusion_y[:,1]==0)}")