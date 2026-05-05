# CS528-Project - ASL Gesture Recognition System
CS528 Final Project

## Overview
Our project aims to build a real-time American Sign Language (ASL) gesture recognition system to assist individuals who are non-verbal or have lost the ability to speak. This system uses a webcam to classify them into meaningful outputs such as letters and common expressions. Our goal is to create a simple and accessible tool that translates ASL gestures into readable text, helping bridge communication gaps in basic interactions. 

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


## Notes:
This is an ongoing project, so features and implementation details will continue to change as development progresses. Last update: May 4th, 2026
