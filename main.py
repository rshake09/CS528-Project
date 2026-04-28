import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import threading
import subprocess
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from gtts import gTTS

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model('./models/asl_model.keras')

# IMPORTANT: make sure this matches training class_names
labels = ['hello', 'goodbye', 'thank_you', 'hungry', 'a', 'b', 'c', 'd']

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

# ---------------- SETTINGS ----------------
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.75
SMOOTHING_FRAMES = 5

cap = cv2.VideoCapture(0)

prediction_buffer = []
last_spoken = None
last_spoken_time = 0

# ---------------- TTS ----------------
def speak(text):
    def run():
        tts = gTTS(text=text, lang='en')
        filename = "temp.mp3"
        tts.save(filename)
        subprocess.run(["afplay", filename])
    threading.Thread(target=run).start()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    display_text = "No hand detected"
    confidence_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = frame.shape

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)

            # margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # draw box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, IMG_SIZE)
            hand_img = preprocess_input(hand_img)
            hand_img = np.expand_dims(hand_img, axis=0)

            preds = model.predict(hand_img, verbose=0)
            idx = np.argmax(preds)
            conf = preds[0][idx]

            prediction_buffer.append(idx)

            # keep buffer small
            if len(prediction_buffer) > SMOOTHING_FRAMES:
                prediction_buffer.pop(0)

            # majority vote
            if len(prediction_buffer) == SMOOTHING_FRAMES:
                final_idx = max(set(prediction_buffer), key=prediction_buffer.count)
                final_label = labels[final_idx]
                final_conf = conf

                display_text = final_label.replace('_', ' ')
                confidence_text = f"{final_conf*100:.1f}%"

                # speak only if confident and stable
                current_time = time.time()
                if (
                    final_conf > CONF_THRESHOLD and
                    final_label != last_spoken and
                    current_time - last_spoken_time > 2
                ):
                    speak(display_text)
                    last_spoken = final_label
                    last_spoken_time = current_time

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------------- UI ----------------
    cv2.putText(frame, f"Gesture: {display_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence_text}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ASL Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
hands.close()