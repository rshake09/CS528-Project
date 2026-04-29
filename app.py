import cv2
import pickle
import numpy as np
import mediapipe as mp
import time

# ---------------- LOAD MODEL ----------------
with open("./model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
gesture_labels = saved["labels"]

print(f"[INFO] Loaded model with gestures: {gesture_labels}")

# ---------------- SETUP ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ---------------- STATE ----------------
sentence = []
current_prediction = None
hold_start = None
HOLD_DURATION = 1.5            # seconds to hold before confirming
last_confirmed = None
last_confirmed_time = 0
COOLDOWN = 2.0                 # seconds before same word can be added again

print("[INFO] Running. Controls:")
print("  Hold gesture 1.5s -> adds to sentence")
print("  Press 'c' -> clear sentence")
print("  Press 'q' -> quit")

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    prediction = None
    confidence = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract features (same normalization as training)
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y

        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist_x)
            features.append(lm.y - wrist_y)

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = max(probs)

        # Only show prediction if confident enough
        if confidence < 0.6:
            prediction = None

    # ---------------- HOLD LOGIC ----------------
    now = time.time()

    if prediction is not None:
        if prediction == current_prediction:
            held = now - hold_start
            progress = min(held / HOLD_DURATION, 1.0)

            if held >= HOLD_DURATION:
                if prediction != last_confirmed or (now - last_confirmed_time) > COOLDOWN:
                    sentence.append(prediction.upper())
                    last_confirmed = prediction
                    last_confirmed_time = now
                    hold_start = now
        else:
            current_prediction = prediction
            hold_start = now
            progress = 0.0
    else:
        current_prediction = None
        hold_start = None
        progress = 0.0

    # ---------------- DRAW UI ----------------
    overlay = frame.copy()

    # Dark bar at top for sentence
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    sentence_text = " ".join(sentence) if sentence else "..."
    cv2.putText(frame, sentence_text, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # Current prediction + confidence
    if prediction:
        label_text = f"{prediction.upper()}  ({int(confidence * 100)}%)"
        cv2.putText(frame, label_text, (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 3)

        # Progress bar for hold
        bar_w = int(w * progress)
        cv2.rectangle(frame, (0, h - 20), (bar_w, h), (0, 255, 100), -1)
        cv2.rectangle(frame, (0, h - 20), (w, h), (80, 80, 80), 2)

        hint = "Hold steady..." if progress < 1.0 else "Added!"
        cv2.putText(frame, hint, (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "No gesture detected", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Controls hint
    cv2.putText(frame, "C: clear  |  Q: quit", (w - 220, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("ASL Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = []
        print("[INFO] Sentence cleared.")

cap.release()
cv2.destroyAllWindows()