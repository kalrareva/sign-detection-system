"""
Downloads the MediaPipe hand landmark detection model.

The mediapipe pip package no longer bundles model weights -- the Tasks
API loads them from a separate .task file at runtime. Run this once
before anything else:

    python download_model.py
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import HAND_LANDMARKER_MODEL_PATH, MODEL_DIR  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        print(f"Model already present at {HAND_LANDMARKER_MODEL_PATH}")
        return

    print(f"Downloading hand landmark model from:\n  {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print(
            "If this network blocks storage.googleapis.com, download the "
            "file manually from the URL above and save it to:\n  "
            f"{HAND_LANDMARKER_MODEL_PATH}"
        )
        sys.exit(1)

    print(f"Saved model to {HAND_LANDMARKER_MODEL_PATH}")


if __name__ == "__main__":
    main()
