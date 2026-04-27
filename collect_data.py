import cv2
import os
import argparse

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, required=True)
parser.add_argument('--count', type=int, default=150)
args = parser.parse_args()

label = args.label.lower()
max_images = args.count

# ---------------- SETUP ----------------
base_dir = "./data"
save_dir = os.path.join(base_dir, label)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print(f"[INFO] Collecting: {label}")
print("[INFO] Press 's' to save, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural selfie view
    frame = cv2.flip(frame, 1)

    cv2.putText(frame,
                f"{label} | {count}/{max_images}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("RAW DATA COLLECTION", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        file_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Saved {file_path}")
        count += 1

        if count >= max_images:
            print("[DONE] Collected enough samples.")
            break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()