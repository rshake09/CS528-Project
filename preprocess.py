import cv2
import os
import mediapipe as mp

# ---------------- SETUP ----------------
input_dir = "./data"
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

image_size = (224, 224)

# ---------------- LOOP ----------------
for label in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label)
    save_label_path = os.path.join(output_dir, label)
    os.makedirs(save_label_path, exist_ok=True)

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            print(f"No hand: {img_name}")
            continue

        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = image.shape

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_list) * w)
            x_max = int(max(x_list) * w)
            y_min = int(min(y_list) * h)
            y_max = int(max(y_list) * h)

            # margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            crop = image[y_min:y_max, x_min:x_max]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, image_size)

            save_path = os.path.join(save_label_path, img_name)
            cv2.imwrite(save_path, crop)

            print(f"Processed: {save_path}")

hands.close()