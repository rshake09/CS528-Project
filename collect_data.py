import cv2
import os
import argparse

# ------------------ ARGUMENT PARSER ------------------
parser = argparse.ArgumentParser(description="ASL Data Collection Tool")
parser.add_argument('--label', type=str, required=True,
                    help='Gesture label (e.g., hello, goodbye, thank_you, A, B, etc.)')
parser.add_argument('--count', type=int, default=150,
                    help='Number of images to collect (default: 150)')

args = parser.parse_args()
label = args.label.lower()
max_images = args.count

# ------------------ SETUP ------------------
output_dir = './data'
output_path = os.path.join(output_dir, label)
os.makedirs(output_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print(f"[INFO] Collecting data for label: {label}")
print("[INFO] Press 's' to save, 'q' to quit")

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define ROI (center box)
    x1, y1 = int(w * 0.3), int(h * 0.3)
    x2, y2 = int(w * 0.7), int(h * 0.7)

    roi = frame[y1:y2, x1:x2]

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay text
    cv2.putText(frame, f'Label: {label} | Count: {count}/{max_images}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, "Press 's' to save | 'q' to quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('ASL Data Capture', frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        resized = cv2.resize(roi, (224, 224))
        image_path = os.path.join(output_path, f'{count}.jpg')
        cv2.imwrite(image_path, resized)

        count += 1
        print(f"[SAVED] {image_path}")

        if count >= max_images:
            print("[INFO] Finished collecting data.")
            break

    elif key == ord('q'):
        print("[INFO] Exiting early.")
        break

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()