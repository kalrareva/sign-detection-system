"""
Interactive webcam data collection for the sign language classifier.

For each label in config.LABELS, hold up that hand sign in front of the
webcam and hold [SPACE] to capture samples. Move your hand slightly
between captures (angle, distance, position in frame) so the classifier
learns a robust, not-memorized, version of each sign.

Controls (while running):
    SPACE (hold)  - capture samples continuously while held
    N             - skip to next label early
    Q / ESC       - quit and save whatever has been collected so far

Output: appends rows to data/landmarks.csv (created if missing), with
columns feature_0..feature_62 plus a label column.
"""

import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    LABELS,
    HAND_LANDMARKER_MODEL_PATH,
    LANDMARKS_CSV_PATH,
    DATA_DIR,
    NUM_FEATURES,
    SAMPLES_PER_LABEL,
)
from utils import HandLandmarker, normalize_landmarks, draw_landmarks  # noqa: E402


def ensure_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LANDMARKS_CSV_PATH):
        with open(LANDMARKS_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"feature_{i}" for i in range(NUM_FEATURES)] + ["label"])


def main():
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        print("Model file not found. Run `python download_model.py` first.")
        sys.exit(1)

    ensure_csv_header()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Check your camera connection/permissions.")
        sys.exit(1)

    landmarker = HandLandmarker(HAND_LANDMARKER_MODEL_PATH, running_mode="VIDEO")
    start_time = time.time()

    csv_file = open(LANDMARKS_CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    print("Starting data collection. Hold SPACE to capture, N for next label, Q to quit.\n")

    try:
        for label in LABELS:
            count = 0
            print(f"--- Label: {label}  (target {SAMPLES_PER_LABEL} samples) ---")

            while count < SAMPLES_PER_LABEL:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read from webcam.")
                    break

                frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
                timestamp_ms = int((time.time() - start_time) * 1000)
                hand = landmarker.process(frame, timestamp_ms)

                capturing = False
                if hand is not None:
                    draw_landmarks(frame, hand)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt
                if key in (ord("n"), ord("N")):
                    break
                if key == 32 and hand is not None:  # SPACE, and a hand is visible
                    capturing = True

                if capturing:
                    features = normalize_landmarks(hand)
                    writer.writerow(list(features) + [label])
                    count += 1

                status = "CAPTURING" if capturing else ("HAND OK" if hand is not None else "NO HAND")
                color = (0, 0, 255) if capturing else ((0, 200, 0) if hand is not None else (0, 165, 255))
                cv2.putText(frame, f"Label: {label}   Samples: {count}/{SAMPLES_PER_LABEL}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, "SPACE=capture  N=next label  Q=quit",
                            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Collect Sign Data", frame)

            print(f"  Collected {count} samples for '{label}'.")

    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        csv_file.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nData saved to {LANDMARKS_CSV_PATH}")


if __name__ == "__main__":
    main()
