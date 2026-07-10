"""
realtime_emotion.py
--------------------
Real-time facial emotion recognition from a webcam feed.

Pipeline per frame:
    1. Capture frame from webcam
    2. Detect face(s) with OpenCV Haar Cascade (or optional DNN detector)
    3. Crop, convert to grayscale, resize to 48x48, normalize
    4. Predict emotion probabilities with the trained CNN
    5. Smooth predictions over recent frames (per face position)
    6. Draw bounding box + label + confidence, show FPS

Usage:
    python realtime_emotion.py --model best_model.h5
    python realtime_emotion.py --model best_model.h5 --camera 0 --detector dnn
"""

import argparse
import json
import os
import time
from collections import deque, defaultdict

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import EMOTION_LABELS as DEFAULT_EMOTION_LABELS

IMG_SIZE = 48


def load_emotion_labels(labels_path="class_names.json"):
    """Uses the class order saved by train.py if available (recommended,
    guarantees a match with however the trained model's folders were
    ordered), otherwise falls back to the default label list."""
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            names = json.load(f)
        return [name.capitalize() for name in names]
    print(f"Warning: {labels_path} not found, using default label order. "
          "This may mismatch your trained model's classes.")
    return DEFAULT_EMOTION_LABELS
SMOOTHING_WINDOW = 8  # number of recent frames to average predictions over


def parse_args():
    p = argparse.ArgumentParser(description="Real-time facial emotion recognition")
    p.add_argument("--model", type=str, default="best_model.keras",
                    help="Path to trained Keras model (.keras or .h5)")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--detector", type=str, default="haar",
                    choices=["haar", "dnn"],
                    help="Face detector backend")
    p.add_argument("--min_confidence", type=float, default=0.5,
                    help="Minimum DNN detector confidence (only used if --detector dnn)")
    return p.parse_args()


def build_haar_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces_haar(detector, gray_frame):
    faces = detector.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return [(x, y, w, h) for (x, y, w, h) in faces]


def build_dnn_detector():
    # Uses OpenCV's bundled SSD face detector (res10_300x300).
    # Requires 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel'
    # to be downloaded separately and placed next to this script.
    proto = "deploy.prototxt"
    weights = "res10_300x300_ssd_iter_140000.caffemodel"
    return cv2.dnn.readNetFromCaffe(proto, weights)


def detect_faces_dnn(net, frame, min_confidence):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def match_face_to_track(box, tracks, max_dist=60):
    """Assigns a detected box to the nearest existing track (by center distance),
    or creates a new track id if none is close enough. Keeps smoothing stable
    across frames for the same person."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2

    best_id, best_dist = None, max_dist
    for track_id, center in tracks.items():
        dist = np.hypot(cx - center[0], cy - center[1])
        if dist < best_dist:
            best_dist, best_id = dist, track_id

    if best_id is None:
        best_id = max(tracks.keys(), default=-1) + 1

    tracks[best_id] = (cx, cy)
    return best_id


def preprocess_face(gray_frame, box):
    x, y, w, h = box
    face = gray_frame[y:y + h, x:x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # shape (1, 48, 48, 1)
    return face


def main():
    args = parse_args()

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)
    emotion_labels = load_emotion_labels()
    print("Using emotion labels:", emotion_labels)

    if args.detector == "haar":
        detector = build_haar_detector()
    else:
        detector = build_dnn_detector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    # Per-track rolling prediction history for smoothing, and center tracking.
    history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
    tracks = {}

    prev_time = time.time()
    fps = 0.0

    print("Starting real-time recognition. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if args.detector == "haar":
            boxes = detect_faces_haar(detector, gray)
        else:
            boxes = detect_faces_dnn(detector, frame, args.min_confidence)

        for box in boxes:
            x, y, w, h = box
            track_id = match_face_to_track(box, tracks)

            face_input = preprocess_face(gray, box)
            probs = model.predict(face_input, verbose=0)[0]

            history[track_id].append(probs)
            smoothed = np.mean(history[track_id], axis=0)
            label_idx = int(np.argmax(smoothed))
            label = emotion_labels[label_idx]
            confidence = float(smoothed[label_idx])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            text = f"{label} ({confidence * 100:.0f}%)"
            cv2.putText(frame, text, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # FPS calculation
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
