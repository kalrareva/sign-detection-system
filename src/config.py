"""
Shared configuration for the Real-Time Sign Language Translator.

Edit LABELS to change which signs the model learns. Start small (the
default 12) to keep a weekend timeline realistic, then expand later.

Note: J and Z are deliberately excluded. In ASL fingerspelling those two
letters require a *motion* (a trace/swipe), not a static hand pose, so a
single-frame landmark classifier can't represent them. Handling them
would require a sequence model (e.g. an LSTM/temporal classifier) over a
window of frames -- a great "v2" extension, not a weekend one.
"""

import os

# ---- Signs the model will learn ------------------------------------------
LABELS = ["A", "B", "C", "D", "E", "F", "I", "L", "O", "U", "V", "Y"]

# ---- Paths -----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

HAND_LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
LANDMARKS_CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "sign_classifier.pkl")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# CNN Paths
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "sign_classifier_cnn.pth")
CNN_CLASSES_PATH = os.path.join(MODEL_DIR, "cnn_classes.json")
CNN_CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "cnn_confusion_matrix.png")
CNN_TRAINING_CURVES_PATH = os.path.join(MODEL_DIR, "cnn_training_curves.png")

# ---- Landmark / model settings ---------------------------------------------
NUM_LANDMARKS = 21          # MediaPipe hand landmark count
NUM_FEATURES = NUM_LANDMARKS * 3  # x, y, z per landmark, flattened

# ---- Data collection settings ----------------------------------------------
SAMPLES_PER_LABEL = 150     # target samples to collect per sign

# ---- Real-time inference settings ------------------------------------------
CONFIDENCE_THRESHOLD = 0.75     # minimum model confidence to accept a prediction
HOLD_FRAMES_TO_CONFIRM = 15     # consecutive matching frames (~0.5s @ 30fps) to confirm a letter
CONFIRM_COOLDOWN_FRAMES = 20    # frames to wait after confirming before accepting a new letter
