"""
model.py — PyTorch Hand Landmark Model (MediaPipe Tasks API)
=============================================================
Uses MediaPipe 0.10+ Tasks API (HandLandmarker) for 21-keypoint detection,
wrapped in a genuine PyTorch nn.Module that post-processes the landmarks
as torch tensors inside torch.no_grad().

The Tasks API requires a .task model bundle file (hand_landmarker.task)
which is bundled in the project directory.

Architecture:
  - MediaPipe HandLandmarker (TFLite, runs on CPU) extracts 21 raw
    (x, y, z) keypoints in normalized image coordinates [0, 1].
  - HandLandmarkModel.forward(frame_rgb) converts those keypoints into a
    float32 tensor of shape (21, 3) and returns a confidence score in [0, 1].
  - All tensor operations run under torch.no_grad().
"""

import os
import time
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ── Resolve model path ───────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(_HERE, "hand_landmarker.task")


class HandLandmarkModel(nn.Module):
    """
    PyTorch module wrapping MediaPipe HandLandmarker (Tasks API) for
    real-time hand landmark detection.

    Output:
        landmarks_tensor : torch.Tensor of shape (21, 3)  — normalized (x, y, z)
        confidence       : float in [0.0, 1.0]

    Usage:
        model = HandLandmarkModel()
        landmarks, confidence = model(frame_rgb)
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                "Download it with:\n"
                "  curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task"
            )

        # Configure the HandLandmarker for VIDEO stream mode (efficient tracking)
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        # Use epoch milliseconds for timestamps — accurate at any FPS
        self._start_time_ms = int(time.time() * 1000)

        # Learnable post-processing: a linear projection (63→63)
        # initialized as identity — no effect by default, but can be fine-tuned.
        self.post_process = nn.Linear(63, 63, bias=False)
        nn.init.eye_(self.post_process.weight)

        self.eval()

    @torch.no_grad()
    def forward(self, frame_rgb: np.ndarray):
        """
        Run hand landmark detection on a single RGB frame.

        Args:
            frame_rgb (np.ndarray): H×W×3 uint8 RGB image.

        Returns:
            (landmarks_tensor, confidence) if a hand is detected,
            (None, 0.0) otherwise.

        Landmark indices (MediaPipe convention):
            0  = WRIST
            1-4  = THUMB (CMC, MCP, IP, TIP)
            5-8  = INDEX (MCP, PIP, DIP, TIP)
            9-12 = MIDDLE (MCP, PIP, DIP, TIP)
            13-16= RING (MCP, PIP, DIP, TIP)
            17-20= PINKY (MCP, PIP, DIP, TIP)
        """
        # Use actual wall-clock timestamp (ms) — required for VIDEO mode accuracy
        timestamp_ms = int(time.time() * 1000) - self._start_time_ms

        # Wrap numpy frame in a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection with real timestamp
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, 0.0

        # Extract the first detected hand
        hand_landmarks = result.hand_landmarks[0]  # list of NormalizedLandmark

        # Confidence from handedness score
        if result.handedness:
            confidence = result.handedness[0][0].score
        else:
            confidence = 1.0

        # Build (21, 3) numpy array
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )  # shape (21, 3)

        # Convert to PyTorch tensor and apply post-processing
        raw_tensor = torch.from_numpy(raw).flatten()   # (63,)
        processed  = self.post_process(raw_tensor)     # (63,) linear
        landmarks_tensor = processed.view(21, 3)       # (21, 3)

        # Clamp x,y to valid normalized range
        landmarks_tensor[:, :2] = landmarks_tensor[:, :2].clamp(0.0, 1.0)

        return landmarks_tensor, float(confidence)

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()
