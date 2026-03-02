"""
utils.py — Smoothing and Helper Utilities
==========================================
Provides:
  - ExponentialSmoother: reduces jitter in fingertip coordinates via EMA.
  - interpolate_points: fills gaps between consecutive draw points to avoid
    broken/dotted strokes when the finger moves quickly.
  - draw_status_bar: renders the HUD overlay at the top of the display frame.
  - get_pixel_coords: converts normalized landmark coords to pixel coords.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Exponential Moving Average Smoother
# ---------------------------------------------------------------------------

class ExponentialSmoother:
    """
    Applies exponential moving average (EMA) smoothing to 2D point coordinates.

    How it works:
        smoothed = alpha * new_value + (1 - alpha) * previous_smoothed

    A lower alpha makes the output more stable (more lag).
    A higher alpha makes it more responsive (less lag, more jitter).

    Args:
        alpha (float): Smoothing factor in (0, 1]. Default 0.4 works well.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def smooth(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Smooth a new (x, y) integer point and return the smoothed point.

        On the first call there's no history, so the raw point is returned
        and stored as the baseline.
        """
        pt = np.array(point, dtype=np.float32)

        if self._prev is None:
            # First data point — initialize state, return as-is
            self._prev = pt
            return point

        # EMA: blend previous and new measurement
        smoothed = self.alpha * pt + (1.0 - self.alpha) * self._prev
        self._prev = smoothed

        return (int(smoothed[0]), int(smoothed[1]))

    def reset(self):
        """Reset smoothing history (call when drawing stops or hand is lost)."""
        self._prev = None


# ---------------------------------------------------------------------------
# Stroke Interpolation
# ---------------------------------------------------------------------------

def interpolate_points(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    density: int = 3,
) -> List[Tuple[int, int]]:
    """
    Generate intermediate points between p1 and p2 to ensure continuous strokes.

    Why this is needed:
        When the finger moves quickly between frames, consecutive fingertip
        positions can be many pixels apart. Drawing only at those positions
        creates dotted/broken strokes. By linearly interpolating intermediate
        points we fill the gaps smoothly.

    Args:
        p1: Start point (x1, y1).
        p2: End point (x2, y2).
        density: Number of intermediate points to insert per pixel of distance.
                 A value of 3 means one point per 3 pixels of travel distance.

    Returns:
        List of (x, y) tuples from p1 to p2, inclusive.
    """
    x1, y1 = p1
    x2, y2 = p2
    dist = int(np.hypot(x2 - x1, y2 - y1))

    if dist == 0:
        return [p1]

    num_points = max(dist // density, 1)
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        points.append((x, y))

    return points


# ---------------------------------------------------------------------------
# Pixel Coordinate Helper
# ---------------------------------------------------------------------------

def get_pixel_coords(
    landmark_tensor,
    idx: int,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int]:
    """
    Convert a normalized landmark coordinate to absolute pixel coordinates.

    MediaPipe landmarks are in range [0, 1]. Multiply by frame dimensions.

    Args:
        landmark_tensor: torch.Tensor of shape (21, 3), normalized coords.
        idx: Landmark index (0–20).
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        (x, y) pixel coordinates as integers.
    """
    lm = landmark_tensor[idx]
    x = int(float(lm[0]) * frame_w)
    y = int(float(lm[1]) * frame_h)
    return x, y


# ---------------------------------------------------------------------------
# HUD / Status Bar Renderer
# ---------------------------------------------------------------------------

# Color name → BGR tuple (for status bar display)
COLOR_NAME_TO_BGR = {
    "Blue":   (255, 100, 0),
    "Green":  (0, 200, 50),
    "Red":    (0, 0, 220),
    "Yellow": (0, 200, 200),
}

def draw_status_bar(
    frame: np.ndarray,
    mode: str,
    color_name: str,
    brush_size: int,
    eraser_size: int,
    fps: float,
) -> np.ndarray:
    """
    Render a semi-transparent HUD bar at the top of the frame.

    Displays: Mode | Color swatch | Brush size | Eraser size | FPS

    Args:
        frame: BGR display frame (will be modified in place and returned).
        mode: Current state string, e.g. 'DRAW', 'ERASE', 'IDLE'.
        color_name: Active color name, e.g. 'Blue'.
        brush_size: Current brush radius in pixels.
        eraser_size: Current eraser radius in pixels.
        fps: Current frames-per-second.

    Returns:
        Frame with HUD bar overlaid.
    """
    bar_h = 60
    # Semi-transparent black background for the bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Mode indicator ────────────────────────────────────────────────────
    mode_colors = {
        "DRAW":  (0, 230, 100),
        "ERASE": (0, 80, 255),
        "IDLE":  (160, 160, 160),
    }
    mode_bgr = mode_colors.get(mode, (200, 200, 200))
    cv2.putText(
        frame, f"Mode: {mode}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_bgr, 2, cv2.LINE_AA,
    )

    # ── Color swatch ──────────────────────────────────────────────────────
    swatch_x = 215
    swatch_bgr = COLOR_NAME_TO_BGR.get(color_name, (255, 255, 255))
    cv2.rectangle(frame, (swatch_x, 12), (swatch_x + 30, 42), swatch_bgr, -1)
    cv2.rectangle(frame, (swatch_x, 12), (swatch_x + 30, 42), (255, 255, 255), 1)
    cv2.putText(
        frame, color_name, (swatch_x + 38, 33),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 1, cv2.LINE_AA,
    )

    # ── Brush size ────────────────────────────────────────────────────────
    cv2.putText(
        frame, f"Brush: {brush_size}px", (360, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # ── Eraser size ──────────────────────────────────────────────────────
    cv2.putText(
        frame, f"Eraser: {eraser_size}px", (490, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # ── FPS counter ──────────────────────────────────────────────────────
    fps_color = (0, 220, 0) if fps >= 20 else (0, 100, 255)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (frame.shape[1] - 130, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA,
    )

    return frame
