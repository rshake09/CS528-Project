# CS528-Project - ASL Gesture Recognition System
CS528 Final Project
Members: Athena Yung, Chau Tran, Rizma Shaikh
___

## Overview
Our project aims to build a **real-time American Sign Language (ASL) gesture recognition** system to assist individuals who are non-verbal or have lost the ability to speak. This system uses a webcam to classify them into meaningful outputs such as letters and common expressions. Our goal is to create a simple and accessible tool that translates ASL gestures into readable text, helping bridge communication gaps in basic interactions. 

## Gestures
To ensure reliability and accuracy within the project timeline, we focus on a small and high-impact set of gestures:
- Greetings: Hello, Goodbye
- Expressions: Thank You, Hungry, Yes, No
- Alphabet A-Z

## Pipeline
Webcam Input:
- Hand detection (MediaPipe)
- Landmark extraction
- Feature processing
- Gesture classification (ML Model)
- Output display (text)

## Teck Stack
Computer vision:
- MediaPipe Hands
- OpenCV

Machine Learning:
- Scikit-learn (k-NN, Random Forest)

Programming Language:
- Python


___
## How It Works

1. Webcam captures live video frames
2. MediaPipe Hands detects the hand and extracts 21 keypoints
3. Keypoints are normalized relative to the wrist
4. A trained Random Forest classifier predicts the gesture
5. Prediction and confidence are displayed on screen in real time
6. Hold a gesture for 0.8 seconds to add it to the sentence bar

---

## Setup

### 1. Install Miniconda

Download and install Miniconda from https://docs.conda.io/en/latest/miniconda.html

Then initialize it:
```bash
conda init zsh
source ~/.zshrc
```

### 2. Create the environment

```bash
conda create -n asl_env python=3.10
conda activate asl_env
```

### 3. Install dependencies

```bash
pip install mediapipe==0.10.9 opencv-python scikit-learn
```

---

## Running the Project

Make sure you have activated the conda environment before running any script:

```bash
conda activate asl_env
```

### Step 1 — Collect gesture data

Run this once per gesture, replacing `<your_name>` and `<gesture>` accordingly:

```bash
python3 collect_data.py --user <your_name> --label <gesture>
```

Example — collecting 50 samples of "hello":
```bash
python3 collect_data.py --user athena --label hello
```

Controls while collecting:
- Press `s` to save a frame
- Press `q` to quit early

Full collection commands:
```bash
python3 collect_data.py --user athena --label hello
python3 collect_data.py --user athena --label bye
python3 collect_data.py --user athena --label thank_you
python3 collect_data.py --user athena --label hungry
python3 collect_data.py --user athena --label yes
python3 collect_data.py --user athena --label no
python3 collect_data.py --user athena --label a
python3 collect_data.py --user athena --label b
python3 collect_data.py --user athena --label c
python3 collect_data.py --user athena --label d
```

Images are saved to `data/<label>/` with the naming convention `<user>_<label>_<n>.jpg`.

### Step 2 — Train the model

Once data is collected, run:

```bash
python3 rf_train.py
```

This extracts keypoints from all images, trains a Random Forest classifier, evaluates accuracy, and saves the model to `model.pkl`. Training takes only a few seconds.

### Step 3 — Run the live app

```bash
python3 app.py
```

Controls while running:
- **Hold a gesture for 0.8s** → adds to sentence bar
- Press `c` → clears the sentence
- Press `q` → quits

---

## Project Structure

```
CS528-Project/
├── collect_data.py     # webcam data collection script
├── preprocess.py       # image preprocessing (crop + resize)
├── rf_train.py         # keypoint extraction + model training
├── app.py              # real-time gesture recognition app
├── model.pkl           # trained model (generated after running rf_train.py)
├── data/               # raw collected images, organized by gesture label
│   ├── hello/
│   ├── bye/
│   ├── thank_you/
│   └── ...
└── README.md
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10 | runtime (via conda) |
| mediapipe | 0.10.9 | hand keypoint detection |
| opencv-python | latest | webcam capture + UI |
| scikit-learn | latest | Random Forest classifier |
| numpy | latest | feature vector processing |

---

## Notes

- Use Python 3.10 via conda — newer versions (3.11+) have compatibility issues with mediapipe
- Always activate `asl_env` before running any script
- `model.pkl` is excluded from the repo via `.gitignore` — each person needs to run `rf_train.py` to generate it locally
- `data/` is also excluded from the repo — collect your own data locally or share via another method

This is an ongoing project, so features and implementation details will continue to change as development progresses. Last update: May 4th, 2026
