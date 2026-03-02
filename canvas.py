"""
canvas.py — Drawing Canvas System
===================================
Maintains a separate off-screen NumPy array (the "canvas") that holds all
drawn strokes. The canvas is blended with the live webcam frame each tick
using cv2.addWeighted, creating the illusion of drawing on a transparent
surface.

Key design decisions:
  - The canvas is completely independent of the display frame. This means
    resolution changes or frame flips don't corrupt drawing history.
  - Erasing works by painting black circles on the canvas (not by
    clearing entire regions), which is fast and precise.
  - Drawing uses filled circles (not lines) at each point for consistent
    brush feel. Stroke continuity is achieved by the caller passing
    interpolated intermediate points (see utils.interpolate_points).
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Tuple


class Canvas:
    """
    Off-screen drawing canvas backed by a NumPy uint8 BGR array.

    All drawing operations modify the internal `self.data` array directly
    (in-place). The array is initialized to pure black (0, 0, 0).

    When blended with the live frame via cv2.addWeighted, pixels that remain
    (0, 0, 0) are treated as "transparent" because:
        result = alpha * canvas + (1-alpha) * frame
    At canvas=(0,0,0): result ≈ (1-alpha)*frame → background shows through.

    Args:
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        # All zeros = black background (transparent in blended view)
        self.data = np.zeros((height, width, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        point: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Paint a filled circle on the canvas at `point`.

        Called for every interpolated point in the stroke path. Using filled
        circles (not polylines) avoids thin gaps between frames.

        Args:
            point: (x, y) pixel position to draw at.
            color_bgr: BGR color tuple, e.g. (255, 100, 0) for blue.
            thickness: Brush radius in pixels.
        """
        cv2.circle(self.data, point, thickness, color_bgr, -1)

    # ------------------------------------------------------------------
    # Erasing
    # ------------------------------------------------------------------

    def erase(
        self,
        point: Tuple[int, int],
        eraser_size: int,
    ) -> None:
        """
        Erase a circular region by painting it black (0, 0, 0).

        Because the canvas uses black as the "transparent" base, painting
        black effectively removes any previously drawn strokes in that area.

        Args:
            point: (x, y) center of the eraser circle.
            eraser_size: Radius of the eraser in pixels.
        """
        cv2.circle(self.data, point, eraser_size, (0, 0, 0), -1)

    # ------------------------------------------------------------------
    # Blending with live frame
    # ------------------------------------------------------------------

    def blend(
        self,
        frame: np.ndarray,
        alpha: float = 0.85,
    ) -> np.ndarray:
        """
        Blend the canvas onto the live webcam frame.

        Uses cv2.addWeighted:
            output = alpha * canvas + (1 - alpha) * frame + 0

        Where canvas pixels are black (0,0,0), the frame dominates.
        Where canvas has color, the stroke shows prominently.

        Args:
            frame: Live BGR frame from the webcam (H×W×3).
            alpha: Canvas opacity. 0.85 makes strokes vivid but slightly
                   see-through.

        Returns:
            Blended BGR frame (same shape as input).
        """
        return cv2.addWeighted(self.data, alpha, frame, 1.0 - alpha, 0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Clear all strokes by zeroing the canvas array.

        Uses numpy in-place fill for maximum speed.
        """
        self.data[:] = 0

    def save(self, directory: str = ".") -> str:
        """
        Save the current canvas as a timestamped PNG file.

        Images are saved to `directory` with filename:
            drawing_YYYYMMDD_HHMMSS.png

        Args:
            directory: Folder path to save into (default: current dir).

        Returns:
            Absolute path to the saved file.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, self.data)
        return os.path.abspath(filepath)

    def draw_eraser_cursor(
        self,
        frame: np.ndarray,
        point: Tuple[int, int],
        eraser_size: int,
    ) -> None:
        """
        Draw a visual eraser cursor circle on the display frame (not canvas).

        This gives visual feedback about eraser position and size without
        permanently affecting the canvas.

        Args:
            frame: Display frame to draw the cursor onto (modified in-place).
            point: Eraser center position.
            eraser_size: Eraser radius.
        """
        cv2.circle(frame, point, eraser_size, (180, 180, 180), 2)
        cv2.circle(frame, point, 3, (255, 255, 255), -1)
