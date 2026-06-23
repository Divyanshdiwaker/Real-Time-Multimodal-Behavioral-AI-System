import numpy as np

X = np.load("fusion_data/fusion_X.npy")
y = np.load("fusion_data/fusion_y.npy")

print(f"Original dataset: {len(X)} samples")
np.save("fusion_data/fusion_X_backup.npy", X)
np.save("fusion_data/fusion_y_backup.npy", y)

augmented_X = []
augmented_y = []

for i in range(len(X)):
    # ---- Original sample ----
    augmented_X.append(X[i])
    augmented_y.append(y[i])

    # ---- Simulate webcam noise on emotion features ----
    noisy1 = X[i].copy()
    noisy1[:128] += np.random.normal(0, 0.3, 128)
    augmented_X.append(noisy1)
    augmented_y.append(y[i])

    # ---- Simulate mic noise on stress features ----
    noisy2 = X[i].copy()
    noisy2[128:192] += np.random.normal(0, 0.2, 64)
    augmented_X.append(noisy2)
    augmented_y.append(y[i])

    # ---- Simulate no face detected — zero emotion ----
    noisy3 = X[i].copy()
    noisy3[:128] = 0
    augmented_X.append(noisy3)
    augmented_y.append(y[i])

    # ---- Simulate silence — zero stress ----
    noisy4 = X[i].copy()
    noisy4[128:192] = 0
    augmented_X.append(noisy4)
    augmented_y.append(y[i])

    # ---- Simulate both zero — worst case ----
    noisy5 = X[i].copy()
    noisy5[:192] = 0
    augmented_X.append(noisy5)
    augmented_y.append(y[i])

X_aug = np.array(augmented_X)
y_aug = np.array(augmented_y)

# Shuffle
idx = np.random.permutation(len(X_aug))
X_aug = X_aug[idx]
y_aug = y_aug[idx]

# Save
np.save("fusion_data/fusion_X_augmented.npy", X_aug)
np.save("fusion_data/fusion_y_augmented.npy", y_aug)

print(f"Augmented dataset: {len(X_aug)} samples")
print(f"Label distribution:")
print(f"  Stressed:      {np.sum(y_aug[:, 0] == 1)}")
print(f"  Not stressed:  {np.sum(y_aug[:, 0] == 0)}")
print(f"  Incongruent:   {np.sum(y_aug[:, 1] == 1)}")
print(f"  Not incongruent: {np.sum(y_aug[:, 1] == 0)}")