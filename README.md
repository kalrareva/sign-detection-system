# Sign Language Detection & Translation System

An end-to-end Computer Vision and Deep Learning system that translates American Sign Language (ASL) fingerspelling into text in real time from a live webcam feed.

Built with **MediaPipe Hand Landmark Detection**, a **1D Convolutional Neural Network (CNN)** (PyTorch), a **Random Forest Classifier** (scikit-learn), **OpenCV** real-time video processing with a HUD overlay, and **Text-to-Speech (TTS)** voice synthesis.

---

## 🚀 Key Features

- **Real-Time Hand Landmark Tracking**: Tracks 21 3D hand landmarks per frame at 30+ FPS using MediaPipe's modern `mediapipe.tasks` API.
- **Dual Classifier Architecture**:
  - **1D-CNN (PyTorch)**: A 1D Convolutional Neural Network that captures spatial curvature and joint relations across kinematic finger chains with data augmentation (rotation, scaling, Gaussian jitter).
  - **Random Forest (scikit-learn)**: Fast ensemble baseline classifier.
  - **Live Model Switching**: Press `M` during live camera inference to hot-swap between CNN and Random Forest models on the fly.
- **Translation & Scale Invariance**: Normalizes landmarks relative to the wrist origin and hand size, ensuring recognition is invariant to hand distance and screen position.
- **Hold-to-Confirm Typing**: Stabilizes predictions with a dynamic hold buffer and visual progress meter to prevent accidental noise while moving hands.
- **Text-to-Speech (TTS)**: Press `ENTER` or `S` to speak the translated sentence aloud.
- **Browser Demo (Streamlit + WebRTC)**: Optional web app ready to deploy to Hugging Face Spaces or Streamlit Cloud.

---

## 🧠 System Architecture

```
Webcam Video Stream (OpenCV / WebRTC)
                  │
                  ▼
   MediaPipe HandLandmarker (Tasks API)
                  │
                  ▼ (21 3D Hand Landmarks: x, y, z)
       Coordinate Normalization
    (Wrist Origin Shift + Bone Scale)
                  │
                  ▼ (Normalized 63-dim Feature Vector)
      ┌───────────────────────────┐
      │  Classifier Selection [M] │
      ├─────────────┬─────────────┤
      ▼             ▼             ▼
   1D-CNN      RandomForest     (Custom)
  (PyTorch)   (scikit-learn)
      │             │
      └──────┬──────┘
             ▼ (Predicted Sign + Confidence Score)
   Hold-to-Confirm Temporal Buffer
             │
             ▼
  Real-Time HUD Overlay & Sentence Buffer
             │
             ▼
   Text-to-Speech (TTS) Voice Engine
```

---

## 📋 Project Structure

```
sign-detection-system/
├── app.py                      # Browser-based demo (Streamlit + WebRTC)
├── download_model.py           # Downloads MediaPipe hand landmark model
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation
├── data/
│   └── landmarks.csv           # Collected landmark training dataset
├── models/
│   ├── hand_landmarker.task    # MediaPipe hand tracking model (~7.8MB)
│   ├── sign_classifier.pkl     # Trained Random Forest classifier
│   ├── sign_classifier_cnn.pth # Trained 1D-CNN model weights (PyTorch)
│   ├── cnn_classes.json        # Class label mapping for CNN
│   ├── cnn_confusion_matrix.png# CNN confusion matrix plot
│   ├── cnn_training_curves.png # CNN loss & accuracy curves
│   └── confusion_matrix.png    # Random Forest confusion matrix plot
└── src/
    ├── config.py               # Hyperparameters, paths, and sign labels
    ├── utils.py                # Landmark extraction, normalization & drawing
    ├── model_cnn.py            # 1D-CNN PyTorch architecture & wrapper
    ├── collect_data.py         # Step 1: Webcam training data collector
    ├── train_model.py          # Step 2a: Train Random Forest classifier
    ├── train_cnn.py            # Step 2b: Train 1D-CNN classifier (Deep Learning)
    └── realtime_translate.py   # Step 3: Real-time OpenCV translator with HUD & TTS
```

---

## 🛠️ Installation & Setup

### 1. Create and Activate Virtual Environment

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Download Hand Landmarker Model (One-Time)

```powershell
python download_model.py
```

---

## 🎮 How to Run

### 1. Live Real-Time Translator (OpenCV)

```powershell
python src/realtime_translate.py
```

Optional arguments:
- `python src/realtime_translate.py --model cnn` (Force 1D-CNN model)
- `python src/realtime_translate.py --model rf`  (Force Random Forest model)

**Interactive Controls**:
| Key | Action |
|:---|:---|
| **Hold Sign** | Fills hold meter and types letter after ~0.5s |
| `SPACE` | Insert space between words |
| `BACKSPACE` | Delete last character |
| `C` | Clear sentence |
| `S` or `ENTER` | **Speak sentence aloud (Text-to-Speech)** |
| `M` | **Toggle between 1D-CNN and Random Forest live** |
| `Q` or `ESC` | Quit application |

---

### 2. Train the Classifiers

#### Option A: Train 1D-CNN (Deep Learning)
```powershell
python src/train_cnn.py
```
- Trains for 40 epochs with geometric augmentation (rotation, jitter, scaling).
- Outputs classification report, `models/cnn_training_curves.png`, and `models/cnn_confusion_matrix.png`.

#### Option B: Train Random Forest
```powershell
python src/train_model.py
```
- Trains Random Forest on landmark vectors and outputs `models/confusion_matrix.png`.

---

### 3. Collect More Training Data (Optional)

```powershell
python src/collect_data.py
```
- Hold hand sign in camera view and hold `SPACE` to record samples.
- Press `N` to move to the next sign early; `Q` to save and exit.

---

### 4. Run the Streamlit Browser App

```powershell
streamlit run app.py
```

---

## 🔬 Deep Learning Model Architecture (1D-CNN)

The `SignLandmarkCNN` is designed specifically for hand skeleton topology:
1. **Input Shape**: `(Batch, Channels=3, Length=21)` representing `(x, y, z)` spatial coordinates across 21 joint landmarks.
2. **Conv Block 1**: `Conv1d(3, 32)` → `BatchNorm1d` → `ReLU` → `Conv1d(32, 64)` → `BatchNorm1d` → `ReLU` → `MaxPool1d(2)`.
3. **Conv Block 2**: `Conv1d(64, 128)` → `BatchNorm1d` → `ReLU` → `Conv1d(128, 128)` → `BatchNorm1d` → `ReLU` → `AdaptiveAvgPool1d(1)`.
4. **Classification Head**: `Linear(128, 64)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)` → `Linear(64, num_classes)`.
