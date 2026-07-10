# Real-Time Facial Emotion Recognition System — Design Document

## 0. Setup Notes & Known Gotchas (read this first)

- **Python version**: 3.11+ recommended. Older TensorFlow pins (e.g.
  `tensorflow<2.17`) don't have wheels for newer Python versions — this repo's
  `requirements.txt` uses `tensorflow>=2.16`, which resolves correctly on
  recent Python/pip.
- **Model checkpoints use `.keras` format**, not `.h5` — this is the format
  TensorFlow 2.16+/Keras 3 recommends. `train.py` saves `best_model.keras`
  and `realtime_emotion.py` expects that same filename by default.
- **Keras 3 `SeparableConv2D` API change**: it no longer accepts a single
  `kernel_regularizer` argument — use `depthwise_regularizer` and
  `pointwise_regularizer` instead (already handled in `model.py`, just
  flagging it in case you extend the architecture).
- **`opencv-python` pinning**: if `pip install opencv-python` ever resolves
  to something odd (e.g. a `5.x` version — OpenCV has no official 5.x
  release) and you hit `AttributeError: module 'cv2' has no attribute
  'CascadeClassifier'`, uninstall and pin a known-good version:
  ```
  pip uninstall opencv-python -y
  pip install opencv-python==4.10.0.84
  ```
- **No GPU on native Windows**: TensorFlow >= 2.11 does not support GPU
  acceleration on native Windows (CUDA/cuDNN won't be used even if
  installed). Training will run on CPU only unless you use WSL2 or the
  TensorFlow-DirectML plugin. Expect roughly 2-4 minutes per epoch on a
  typical CPU with the full FER2013 training set.
- **Dataset is not included in this package** — download FER2013 from
  Kaggle yourself (folder-structured version, see Section 2.3) and place
  it as described in Section 4.

## 1. Objective

Build a system that captures live video from a webcam, detects faces in each
frame, and classifies the facial expression into one of several basic
emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) — all with low
enough latency to feel "real time" (target: 15–30 FPS on a CPU, higher on GPU).

## 2. System Architecture

The pipeline has four stages, run once per frame:

```
Webcam Frame
     │
     ▼
[1] Face Detection   → locate face bounding box(es) in the frame
     │  (Haar Cascade / DNN face detector)
     ▼
[2] Preprocessing    → crop, convert to grayscale, resize to 48x48,
     │                  normalize pixel values to [0, 1]
     ▼
[3] Emotion CNN      → forward pass through trained convolutional
     │                  network → probability distribution over 7 classes
     ▼
[4] Post-processing  → argmax + confidence smoothing over recent frames
     │                  → overlay label + bounding box on frame
     ▼
Display / Output Stream
```

### 2.1 Face Detection
Two options are supported, selectable at runtime:
- **Haar Cascade (`haarcascade_frontalface_default.xml`)** — ships with
  OpenCV, very fast, no GPU needed, good enough for frontal faces at close
  range. Used as the default for real-time performance.
- **DNN face detector (OpenCV's `res10_300x300_ssd`)** — more robust to
  angle/lighting, slightly heavier. Used as an optional upgrade.

### 2.2 Emotion Classifier
A compact CNN (a "mini-Xception"-style architecture) is used rather than a
huge network like ResNet50, because:
- Input is small (48×48 grayscale) — a large backbone is unnecessary.
- Inference needs to run every frame, so parameter count / FLOPs must stay
  low for real-time throughput on CPU.
- Depthwise-separable convolutions + residual blocks give a good
  accuracy/speed trade-off (~60-66% accuracy on FER2013, ~50k-100k
  parameters, sub-5ms inference on modern CPUs).

Architecture (see `model.py`):
```
Input (48x48x1)
 → Conv2D(8) + BN + ReLU  → Conv2D(8) + BN + ReLU
 → [Residual separable-conv block, 16 filters] + MaxPool + Dropout
 → [Residual separable-conv block, 32 filters] + MaxPool + Dropout
 → [Residual separable-conv block, 64 filters] + MaxPool + Dropout
 → [Residual separable-conv block, 128 filters] + MaxPool + Dropout
 → Conv2D(num_classes, 3x3) → GlobalAveragePooling2D → Softmax
```

### 2.3 Dataset
**FER2013** (Facial Expression Recognition 2013, Kaggle) is used, in its
folder-structured form:
```
fer2013/
  train/
    angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/   (images)
  test/
    angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/   (images)
```
- ~35,887 grayscale 48×48 images total across 7 classes
- The `train` folder is further split 90/10 into train/validation at load
  time; `test` is held out entirely for final evaluation
- Class order is whatever Keras assigns alphabetically to the subfolders;
  `train.py` saves this exact order to `class_names.json` so the real-time
  app always matches the trained model regardless of folder naming

Class imbalance (Disgust has far fewer images than the other six classes)
is handled with class weighting, computed directly from folder image counts.

### 2.4 Training Strategy
- Data augmentation: random rotation (±10°), width/height shift (±10%),
  zoom (±10%), horizontal flip — applied only to the training split.
- Optimizer: Adam, initial LR 1e-3, with `ReduceLROnPlateau` and
  `EarlyStopping` on validation loss.
- Loss: categorical cross-entropy, with class weights to counter imbalance.
- Checkpointing: best model (by validation accuracy) saved to disk.

### 2.5 Real-Time Inference Loop
- Grab frame → detect faces → for each face, crop/resize/normalize →
  batch-predict → smooth predictions over the last N frames per tracked
  face (simple moving average) to avoid label flicker → draw overlay.
- FPS counter displayed on-screen to verify real-time performance.
- Runs on CPU by default; will automatically use GPU if TensorFlow detects
  one (CUDA-enabled).

## 3. File Layout

```
emotion_recognition/
├── design_document.md      This file
├── requirements.txt         Python dependencies
├── model.py                 CNN architecture definition
├── preprocess.py            FER2013 CSV → numpy arrays, augmentation setup
├── train.py                 Training script, saves best model to disk
└── realtime_emotion.py      Webcam real-time inference application
```

## 4. How to Run

```bash
pip install -r requirements.txt

# 1. Download FER2013 from Kaggle (folder version) and place/unzip it so you
#    have a 'fer2013' folder next to these scripts containing 'train' and
#    'test' subfolders (each with one folder of images per emotion).
# 2. Train the model
python train.py --data_dir fer2013 --epochs 50

# 3. Run real-time recognition (uses trained model + webcam)
python realtime_emotion.py --model best_model.keras
```

## 5. Design Trade-offs & Limitations

- **Accuracy ceiling**: FER2013 is a noisy, low-resolution dataset; even
  state-of-the-art models cap out around 73-76% test accuracy. The
  lightweight model here targets ~65-68%, favoring speed over peak accuracy.
- **Single dominant face assumption for smoothing**: temporal smoothing is
  keyed by face position, so it degrades gracefully with multiple faces but
  works best for one primary subject.
- **Frontal-face bias**: Haar cascades struggle with extreme head poses or
  poor lighting; the optional DNN detector mitigates this at some FPS cost.
- **Privacy**: no frames or predictions are stored or transmitted anywhere
  by default — everything runs locally in-process.

## 6. Possible Extensions
- Swap the CNN for a pretrained face/emotion embedding model for higher
  accuracy at the cost of latency.
- Add face tracking (e.g., simple centroid tracker or `dlib` correlation
  tracker) to keep stable per-person smoothing across frames.
- Export the trained model to ONNX / TensorFlow Lite for deployment on
  mobile or embedded devices.
- Add multi-face analytics dashboard (emotion trends over time, per person).
