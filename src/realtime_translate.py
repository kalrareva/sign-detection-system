"""
Real-time sign language translator with OpenCV, MediaPipe, and Dual Classifier (CNN & Random Forest).

Features:
- Dual-model support: 1D-CNN or Random Forest classifier with live hot-switch (press 'M')
- Visual Hold-to-Confirm meter & hand bounding box overlay
- Text-to-Speech (TTS) integration: speaks translated sentences on ENTER or 'S'
- Blinking cursor sentence buffer with editing controls

Usage:
    python src/realtime_translate.py [--model cnn|rf]
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HAND_LANDMARKER_MODEL_PATH,
    CLASSIFIER_PATH,
    CNN_MODEL_PATH,
    CNN_CLASSES_PATH,
    CONFIDENCE_THRESHOLD,
    HOLD_FRAMES_TO_CONFIRM,
    CONFIRM_COOLDOWN_FRAMES,
)
from utils import HandLandmarker, normalize_landmarks, draw_landmarks


def speak_text(text):
    """Vocalizes text in a non-blocking background thread."""
    clean_text = text.strip()
    if not clean_text:
        return

    def _worker():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(clean_text)
            engine.runAndWait()
        except Exception:
            # Native Windows fallback using SAPI
            escaped = clean_text.replace('"', "'")
            os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\\"{escaped}\\")"')

    threading.Thread(target=_worker, daemon=True).start()


def load_classifier(model_type="auto"):
    """Loads either CNN or Random Forest model, with auto-selection fallback."""
    cnn_available = os.path.exists(CNN_MODEL_PATH) and os.path.exists(CNN_CLASSES_PATH)
    rf_available = os.path.exists(CLASSIFIER_PATH)

    chosen_type = model_type
    if chosen_type == "auto":
        chosen_type = "cnn" if cnn_available else "rf"

    if chosen_type == "cnn":
        if not cnn_available:
            if rf_available:
                print("CNN model not found. Falling back to Random Forest.")
                chosen_type = "rf"
            else:
                print("No trained models found. Run `python src/train_cnn.py` or `python src/train_model.py` first.")
                sys.exit(1)
        else:
            from model_cnn import CNNModelWrapper
            model = CNNModelWrapper(CNN_MODEL_PATH, CNN_CLASSES_PATH)
            return model, "1D-CNN"

    if chosen_type == "rf":
        if not rf_available:
            print(f"Random Forest model not found at {CLASSIFIER_PATH}. Run `python src/train_model.py` first.")
            sys.exit(1)
        model = joblib.load(CLASSIFIER_PATH)
        return model, "RandomForest"

    raise ValueError(f"Unknown model_type: {model_type}")


def get_hand_bbox(landmarks, frame_w, frame_h, pad=25):
    """Computes bounding box around 21 hand landmarks with padding."""
    xs = [int(lm.x * frame_w) for lm in landmarks]
    ys = [int(lm.y * frame_h) for lm in landmarks]
    x_min = max(0, min(xs) - pad)
    y_min = max(0, min(ys) - pad)
    x_max = min(frame_w, max(xs) + pad)
    y_max = min(frame_h, max(ys) + pad)
    return x_min, y_min, x_max, y_max


def main():
    parser = argparse.ArgumentParser(description="Real-Time Sign Language Translator")
    parser.add_argument("--model", choices=["auto", "cnn", "rf"], default="auto",
                        help="Classifier model to use ('cnn', 'rf', or 'auto')")
    args = parser.parse_args()

    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        print("MediaPipe model not found. Run `python download_model.py` first.")
        sys.exit(1)

    model, model_name = load_classifier(args.model)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Please check camera connections/permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = HandLandmarker(HAND_LANDMARKER_MODEL_PATH, running_mode="VIDEO")
    start_time = time.time()

    recent_predictions = deque(maxlen=HOLD_FRAMES_TO_CONFIRM)
    sentence = ""
    cooldown = 0
    confirmed_flash = 0
    fps_history = deque(maxlen=30)
    last_frame_time = time.time()

    print(f"\n========================================================")
    print(f" Real-Time Sign Language Translator Started!")
    print(f" Active Classifier: [{model_name}]")
    print(f" Controls:")
    print(f"   [SPACE]       - Insert space")
    print(f"   [BACKSPACE]   - Delete last character")
    print(f"   [C]           - Clear sentence")
    print(f"   [S] / [ENTER] - Speak sentence aloud (TTS)")
    print(f"   [M]           - Toggle between CNN and Random Forest")
    print(f"   [Q] / [ESC]   - Quit")
    print(f"========================================================\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam disconnected.")
                break

            now = time.time()
            fps = 1.0 / max(now - last_frame_time, 1e-5)
            last_frame_time = now
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            timestamp_ms = int((now - start_time) * 1000)
            hand = landmarker.process(frame, timestamp_ms)

            current_label, current_conf = None, 0.0

            if hand is not None:
                draw_landmarks(frame, hand)
                x1, y1, x2, y2 = get_hand_bbox(hand, w, h)

                features = normalize_landmarks(hand).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                best_idx = probs.argmax()
                current_label = model.classes_[best_idx]
                current_conf = float(probs[best_idx])

                if current_conf >= CONFIDENCE_THRESHOLD:
                    recent_predictions.append(current_label)
                else:
                    recent_predictions.append(None)

                # Draw Bounding Box & Label Tag
                box_color = (0, 220, 0) if current_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                tag_text = f"{current_label} ({current_conf:.0%})" if current_conf >= CONFIDENCE_THRESHOLD else "..."
                tag_y = max(y1 - 10, 25)
                cv2.rectangle(frame, (x1, tag_y - 24), (x1 + 140, tag_y + 4), box_color, -1)
                cv2.putText(frame, tag_text, (x1 + 8, tag_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            else:
                recent_predictions.append(None)

            # Hold to confirm logic
            hold_progress = 0.0
            if len(recent_predictions) > 0 and recent_predictions[-1] is not None:
                active_sign = recent_predictions[-1]
                match_count = sum(1 for p in recent_predictions if p == active_sign)
                hold_progress = match_count / HOLD_FRAMES_TO_CONFIRM

            if cooldown > 0:
                cooldown -= 1
            elif (
                len(recent_predictions) == HOLD_FRAMES_TO_CONFIRM
                and recent_predictions[0] is not None
                and len(set(recent_predictions)) == 1
            ):
                typed_char = recent_predictions[0]
                sentence += typed_char
                recent_predictions.clear()
                cooldown = CONFIRM_COOLDOWN_FRAMES
                confirmed_flash = 12

            # ======================== UI HUD OVERLAY ========================
            # 1. Top Bar
            cv2.rectangle(frame, (0, 0), (w, 50), (25, 25, 25), -1)
            cv2.line(frame, (0, 50), (w, 50), (60, 60, 60), 1)

            cv2.putText(frame, "Sign Language Detection System", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)

            model_badge = f"Model: {model_name} [M]"
            cv2.putText(frame, model_badge, (w - 380, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2)

            fps_text = f"{avg_fps:.0f} FPS"
            cv2.putText(frame, fps_text, (w - 110, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

            # 2. Hold-to-Confirm Progress Bar
            if hold_progress > 0.0:
                meter_w = 260
                meter_x = (w - meter_w) // 2
                meter_y = 65
                cv2.rectangle(frame, (meter_x - 2, meter_y - 2), (meter_x + meter_w + 2, meter_y + 16), (40, 40, 40), -1)
                fill_w = int(meter_w * hold_progress)
                fill_color = (0, 255, 0) if hold_progress >= 0.95 else (0, 200, 255)
                cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_w, meter_y + 14), fill_color, -1)
                cv2.putText(frame, "HOLD TO TYPE", (meter_x + 75, meter_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

            # 3. Confirmation Flash Notification
            if confirmed_flash > 0:
                confirmed_flash -= 1
                cv2.putText(frame, "LETTER CONFIRMED!", (w // 2 - 140, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

            # 4. Bottom Sentence Board
            cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
            cv2.line(frame, (0, h - 90), (w, h - 90), (60, 60, 60), 1)

            # Help shortcuts
            controls_tip = "[SPACE]=Space  |  [BKSP]=Delete  |  [C]=Clear  |  [S/ENTER]=Speak (TTS)  |  [M]=Toggle Model  |  [Q]=Quit"
            cv2.putText(frame, controls_tip, (16, h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

            # Sentence with blinking cursor
            cursor = "_" if (int(now * 2) % 2 == 0) else " "
            display_sentence = sentence[-48:] + cursor
            cv2.putText(frame, f"Output:  {display_sentence}", (16, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 255), 2)

            cv2.imshow("Sign Language Detection & Translation", frame)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("c"), ord("C")):
                sentence = ""
            elif key == 32:  # SPACE
                sentence += " "
            elif key in (8, 127):  # Backspace
                sentence = sentence[:-1]
            elif key in (ord("s"), ord("S"), 10, 13):  # S or ENTER -> Speak
                speak_text(sentence)
            elif key in (ord("m"), ord("M")):  # Toggle Model
                target = "rf" if model_name == "1D-CNN" else "cnn"
                try:
                    model, model_name = load_classifier(target)
                    print(f"Switched model to: {model_name}")
                except Exception as e:
                    print(f"Could not switch model: {e}")

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
