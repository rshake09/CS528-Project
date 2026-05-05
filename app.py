import cv2
import pickle
import numpy as np
import mediapipe as mp
import time

# loading model and labels
with open("./model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
gesture_labels = saved["labels"]

print(f"[INFO] Loaded model with gestures: {gesture_labels}") # for debugging, can remove later

# set up mediapipe and webcam
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# state for hold logic and sentence building
sentence = []
current_prediction = None
hold_start = None
HOLD_DURATION = 0.8
last_confirmed = None
last_confirmed_time = 0
COOLDOWN = 1.2

# instructions
print("[INFO] Running. Controls:")
print("  Hold gesture 0.8s -> adds to sentence")
print("  Press 'c' -> clear sentence")
print("  Press 'q' -> quit")

# main loop for real-time gesture recognition
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

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # normalize keypoints relative to wrist
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y

        features = [] # storing features for prediction
        for lm in hand_landmarks.landmark: # iterating through landmarks to create feature vector
            features.append(lm.x - wrist_x)
            features.append(lm.y - wrist_y)

        # predict gesture using the trained model
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = max(probs)

        # filter out low confidence predictions
        if confidence < 0.6:
            prediction = None

    # check how long the gesture has been held
    now = time.time()

    # checking if the current prediction is the same as the last one to determine if we should start or reset the hold timer
    if prediction is not None:
        if prediction == current_prediction:
            held = now - hold_start
            progress = min(held / HOLD_DURATION, 1.0)

            # checking if the gesture has been held long enough to be added to the sentence, and also checking cooldown to prevent repetitions of the same gesture
            if held >= HOLD_DURATION:
                # add to sentence if not a repeat within cooldown
                if prediction != last_confirmed or (now - last_confirmed_time) > COOLDOWN:
                    sentence.append(prediction.upper())
                    last_confirmed = prediction
                    last_confirmed_time = now
                    hold_start = now
        else:
            # new gesture detected, reset hold timer
            current_prediction = prediction
            hold_start = now
            progress = 0.0
    else:
        current_prediction = None
        hold_start = None
        progress = 0.0

    # create sentence bar at top
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    # build full sentence text and measure its width to scroll if it overflows
    full_text = " ".join(sentence) if sentence else "..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    font_thickness = 2
    text_size, _ = cv2.getTextSize(full_text, font, font_scale, font_thickness)
    text_w = text_size[0]
    bar_padding = 10

    # if text fits in the bar, just draw it normally
    # if it overflows, draw onto a wide canvas and crop to show newest words
    if text_w <= w - bar_padding * 2:
        cv2.putText(frame, full_text, (bar_padding, 48),
                    font, font_scale, (255, 255, 255), font_thickness)
    else:
        canvas_w = text_w + bar_padding * 2
        canvas = np.full((70, canvas_w, 3), (20, 20, 20), dtype=np.uint8)
        cv2.putText(canvas, full_text, (bar_padding, 48),
                    font, font_scale, (255, 255, 255), font_thickness)
        # crop from the right so newest words are always visible
        frame[0:70, 0:w] = canvas[:, canvas_w - w:canvas_w]

    # draw prediction and progress bar
    if prediction:
        label_text = f"{prediction.upper()}  ({int(confidence * 100)}%)"
        cv2.putText(frame, label_text, (10, h - 80),
                    font, 1.3, (0, 255, 100), 3)

        bar_w = int(w * progress)
        cv2.rectangle(frame, (0, h - 20), (bar_w, h), (0, 255, 100), -1)
        cv2.rectangle(frame, (0, h - 20), (w, h), (80, 80, 80), 2)

        hint = "Hold steady..." if progress < 1.0 else "Added!"
        cv2.putText(frame, hint, (10, h - 35),
                    font, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "No gesture detected", (10, h - 80),
                    font, 0.8, (100, 100, 100), 2)

    cv2.putText(frame, "C: clear  |  Q: quit", (w - 220, h - 35),
                font, 0.5, (150, 150, 150), 1)

    cv2.imshow("ASL Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = []
        print("[INFO] Sentence cleared.")

# cleanup
cap.release()
cv2.destroyAllWindows()