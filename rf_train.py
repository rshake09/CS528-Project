import os
import pickle
import numpy as np
import mediapipe as mp
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# set up paths and mediapipe
data_dir = "./data"
model_output = "./model.pkl"

mp_hands = mp.solutions.hands # using mediapipe to extract hand landmarks from images
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# extract landmarks from each image in the dataset
data = []
labels = []
gesture_labels = set()

# iterating through each gesture folder + process each image to get features and labels for training
for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if not os.path.isdir(label_path):
        continue

    print(f"[INFO] Loading gesture: {label}") # for debugging, can remove later

    # iterating through each image in the order of filename for consistency
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None: # if image can't be read, skip it
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        # check if any hand gestures were detected in the image, if not, then skip
        if not result.multi_hand_landmarks:
            continue

        hand_landmarks = result.multi_hand_landmarks[0] # only processing the first detected hand for simplicity

        # normalize landmarks relative to wrist so position doesn't matter
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y

        features = [] # storing features for prediction
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist_x)
            features.append(lm.y - wrist_y)

        data.append(features)
        labels.append(label)
        gesture_labels.add(label)

hands.close()

gesture_labels = sorted(gesture_labels)

# convert to arrays and print summary
data = np.array(data)
labels = np.array(labels)

# printing summary of loaded data 
print(f"\n[INFO] Total samples: {len(data)}")
print(f"[INFO] Gestures found: {gesture_labels}")
print(f"[INFO] Label distribution: { {l: (labels == l).sum() for l in gesture_labels} }")

# train random forest classifier
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

# evaluate and print results
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[RESULT] Test accuracy: {acc * 100:.1f}%")
print("\n[RESULT] Per-gesture breakdown:")
print(classification_report(y_test, y_pred))

# save model to disk
with open(model_output, "wb") as f:
    pickle.dump({"model": model, "labels": gesture_labels}, f)

print(f"\n[SAVED] Model saved to {model_output}")