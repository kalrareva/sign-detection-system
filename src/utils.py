"""
Hand landmark extraction utilities built on MediaPipe's Tasks API.

IMPORTANT: As of mediapipe 0.10.x, the old `mp.solutions.hands` /
`mp.solutions.drawing_utils` API that most tutorials show has been
removed. This module uses the current `mediapipe.tasks` API instead,
and implements its own landmark drawing since `drawing_utils` no
longer ships with the package.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Standard 21-point MediaPipe hand skeleton connections (index pairs).
# Used for drawing since mp.solutions.drawing_utils is no longer available.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


class HandLandmarker:
    """Thin wrapper around mediapipe.tasks.vision.HandLandmarker.

    running_mode:
        "IMAGE" - treat each call as an independent still image.
        "VIDEO" - treat calls as frames of a continuous video stream
                  (requires monotonically increasing timestamps; gives
                  better temporal consistency, used for the webcam
                  scripts in this project).
    """

    def __init__(self, model_path, running_mode="VIDEO", num_hands=1,
                 min_detection_confidence=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        mode_map = {
            "IMAGE": mp_vision.RunningMode.IMAGE,
            "VIDEO": mp_vision.RunningMode.VIDEO,
        }
        if running_mode not in mode_map:
            raise ValueError(f"running_mode must be one of {list(mode_map)}")
        self._mode = mode_map[running_mode]

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self._mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr, timestamp_ms=0):
        """Run detection on a BGR frame (as returned by cv2.VideoCapture).

        Returns the first detected hand's list of 21 landmarks (each with
        .x, .y, .z in normalized [0, 1] image coordinates), or None if no
        hand was detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == mp_vision.RunningMode.VIDEO:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def normalize_landmarks(hand_landmarks):
    """Convert 21 MediaPipe landmarks into a translation/scale-invariant
    63-length feature vector.

    - Translation invariance: subtract the wrist (landmark 0) so the
      feature only encodes hand *shape*, not its position in the frame.
    - Scale invariance: divide by the wrist-to-middle-finger-MCP (landmark
      9) distance, so the feature doesn't depend on how close the hand is
      to the camera.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    wrist = coords[0].copy()
    coords -= wrist

    scale = np.linalg.norm(coords[9])
    if scale < 1e-6:
        scale = 1e-6
    coords /= scale

    return coords.flatten()  # shape (63,)


def draw_landmarks(frame_bgr, hand_landmarks, color=(0, 220, 0), radius=4):
    """Draw the hand skeleton onto a BGR frame in place."""
    h, w = frame_bgr.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[start_idx], points[end_idx], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame_bgr, (x, y), radius, color, -1)

    return frame_bgr
