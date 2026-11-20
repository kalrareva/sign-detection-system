"""
Web Browser Demo: Streamlit + WebRTC Sign Language Translator.

Runs in the browser via WebRTC with live MediaPipe hand landmark extraction
and support for both 1D-CNN (PyTorch) and Random Forest (scikit-learn) classifiers.

Run locally with:
    streamlit run app.py
"""

import os
import sys
import time
from collections import deque

import av
import cv2
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import (
    HAND_LANDMARKER_MODEL_PATH,
    CLASSIFIER_PATH,
    CNN_MODEL_PATH,
    CNN_CLASSES_PATH,
    CNN_CONFUSION_MATRIX_PATH,
    CONFUSION_MATRIX_PATH,
    CONFIDENCE_THRESHOLD,
    HOLD_FRAMES_TO_CONFIRM,
    CONFIRM_COOLDOWN_FRAMES,
)
from utils import HandLandmarker, normalize_landmarks, draw_landmarks


st.set_page_config(
    page_title="Sign Language Detection System",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Real-Time Sign Language Detection & Translation")
st.markdown(
    "Translates American Sign Language (ASL) fingerspelling into text live via your webcam. "
    "Features **MediaPipe Hand Landmark Detection**, **1D-CNN & Random Forest Classification**, and **Real-Time Video Processing**."
)

# Sidebar settings
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.radio(
    "Select Classifier Architecture",
    ["1D-CNN (PyTorch)", "Random Forest (scikit-learn)"]
)

confidence_slider = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=0.95,
    value=float(CONFIDENCE_THRESHOLD),
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Supported Alphabet")
st.sidebar.write("A, B, C, D, E, F, I, L, O, U, V, Y")


@st.cache_resource
def load_model(choice):
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        return None, None, "MediaPipe hand landmarker model missing. Run `python download_model.py`."

    if choice == "1D-CNN (PyTorch)":
        if not (os.path.exists(CNN_MODEL_PATH) and os.path.exists(CNN_CLASSES_PATH)):
            return None, None, "CNN model weights not found. Run `python src/train_cnn.py` first."
        from model_cnn import CNNModelWrapper
        model = CNNModelWrapper(CNN_MODEL_PATH, CNN_CLASSES_PATH)
        return HAND_LANDMARKER_MODEL_PATH, model, None
    else:
        if not os.path.exists(CLASSIFIER_PATH):
            return None, None, "Random Forest model file not found. Run `python src/train_model.py` first."
        model = joblib.load(CLASSIFIER_PATH)
        return HAND_LANDMARKER_MODEL_PATH, model, None


landmarker_path, classifier, error_msg = load_model(model_choice)

if error_msg:
    st.error(error_msg)
    st.stop()

if "sentence" not in st.session_state:
    st.session_state.sentence = ""


class SignProcessor:
    def __init__(self):
        self.landmarker = HandLandmarker(landmarker_path, running_mode="VIDEO")
        self.start_time = time.time()
        self.recent = deque(maxlen=HOLD_FRAMES_TO_CONFIRM)
        self.cooldown = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        hand = self.landmarker.process(img, timestamp_ms)

        label, conf = None, 0.0
        if hand is not None:
            draw_landmarks(img, hand)
            features = normalize_landmarks(hand).reshape(1, -1)
            probs = classifier.predict_proba(features)[0]
            best_idx = probs.argmax()
            label = classifier.classes_[best_idx]
            conf = float(probs[best_idx])
            self.recent.append(label if conf >= confidence_slider else None)

            # Bounding box
            xs = [int(lm.x * w) for lm in hand]
            ys = [int(lm.y * h) for lm in hand]
            x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
            x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)
            color = (0, 220, 0) if conf >= confidence_slider else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} ({conf:.0%})", (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            self.recent.append(None)

        if self.cooldown > 0:
            self.cooldown -= 1
        elif (
            len(self.recent) == HOLD_FRAMES_TO_CONFIRM
            and self.recent[0] is not None
            and len(set(self.recent)) == 1
        ):
            st.session_state.sentence += self.recent[0]
            self.recent.clear()
            self.cooldown = CONFIRM_COOLDOWN_FRAMES

        return av.VideoFrame.from_ndarray(img, format="bgr24")


col_cam, col_output = st.columns([1.6, 1.0])

with col_cam:
    st.subheader("📹 Live Video Stream")
    webrtc_streamer(
        key=f"sign-translator-{model_choice}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_output:
    st.subheader("📝 Translated Sentence")
    st.text_area(
        label="Live Output",
        value=st.session_state.sentence,
        height=140,
        disabled=True
    )

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("␣ Space", use_container_width=True):
            st.session_state.sentence += " "
            st.rerun()
    with btn_col2:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
            st.rerun()
    with btn_col3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.sentence = ""
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Model Evaluation")
    matrix_path = CNN_CONFUSION_MATRIX_PATH if "CNN" in model_choice else CONFUSION_MATRIX_PATH
    if os.path.exists(matrix_path):
        st.image(matrix_path, caption=f"Confusion Matrix ({model_choice})", use_container_width=True)
    else:
        st.info("Train the model to view evaluation confusion matrix.")
