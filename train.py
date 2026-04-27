import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------- SETUP ----------------
data_dir = "./processed_data"
model_output = "./model.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ---------------- EXTRACT LANDMARKS ----------------
data = []
labels = []

gesture_labels = sorted(os.listdir(data_dir))
print(f"[INFO] Found gestures: {gesture_labels}")

for label in gesture_labels:
    label_path = os.path.join(data_dir, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            continue

        hand_landmarks = result.multi_hand_landmarks[0]

        # Flatten all 21 landmarks (x, y) into a 42-length feature vector
        # Normalize relative to wrist (landmark 0) so position doesn't matter
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y

        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist_x)
            features.append(lm.y - wrist_y)

        data.append(features)
        labels.append(label)

hands.close()

# ---------------- TRAIN ----------------
data = np.array(data)
labels = np.array(labels)

print(f"[INFO] Total samples: {len(data)}")
print(f"[INFO] Label distribution: { {l: (labels == l).sum() for l in gesture_labels} }")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Test accuracy: {acc * 100:.1f}%")
print("\n[RESULT] Per-gesture breakdown:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE ----------------
with open(model_output, "wb") as f:
    pickle.dump({"model": model, "labels": gesture_labels}, f)

print(f"\n[SAVED] Model saved to {model_output}")