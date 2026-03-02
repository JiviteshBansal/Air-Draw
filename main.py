"""
main.py — Air Drawing System — Application Entry Point
=======================================================
This is the main loop that ties together all modules:

  model.py    → PyTorch hand landmark detection (MediaPipe Tasks API)
  gestures.py → Finger state analysis and gesture-to-mode mapping
  canvas.py   → Off-screen drawing surface
  state.py    → Mode FSM with debounce
  utils.py    → Smoothing, interpolation, HUD rendering

Loop flow (per frame):
  1. Capture frame from webcam, flip horizontally (mirror view).
  2. Convert to RGB, run HandLandmarkModel → 21 landmarks + confidence.
  3. If confidence < threshold → treat as "no hand" → reset state.
  4. Compute finger states → detect gesture → resolve (mode, color).
  5. Update StateManager with detected mode.
  6. Smooth index fingertip coordinate via ExponentialSmoother.
  7. Act on current mode:
       DRAW  → draw filled circle on canvas at smoothed fingertip
               interpolate between prev_point and current point
       ERASE → erase circle on canvas at smoothed fingertip
               show eraser cursor on display frame
       IDLE  → no drawing action
  8. Blend canvas onto display frame.
  9. Draw HUD status bar.
  10. Draw hand landmark skeleton overlay (optional, toggleable).
  11. Display frame in window.
  12. Handle keyboard input.

Keyboard Controls:
  ESC           → exit application
  C             → clear canvas
  S             → save canvas as PNG in project directory
  Up   Arrow    → increase brush thickness
  Down Arrow    → decrease brush thickness
  Left  Arrow   → decrease eraser size
  Right Arrow   → increase eraser size
  L             → toggle landmark skeleton overlay
"""

import os
import sys
import time
import cv2
import numpy as np

from model    import HandLandmarkModel
from gestures import (
    get_finger_states, detect_gesture,
    MODE_IDLE, MODE_DRAW, MODE_ERASE,
    COLOR_BLUE, COLOR_MAP,
)
from canvas   import Canvas
from state    import StateManager
from utils    import (
    ExponentialSmoother, interpolate_points,
    get_pixel_coords, draw_status_bar,
)


# ── Configuration ────────────────────────────────────────────────────────────

WINDOW_NAME          = "Air Drawing System"
CONFIDENCE_THRESHOLD = 0.50    # Slightly lower threshold = fewer missed frames
INITIAL_COLOR        = COLOR_BLUE
INITIAL_BRUSH_SIZE   = 8       # Brush radius in pixels
INITIAL_ERASER_SIZE  = 35      # Eraser radius in pixels
BRUSH_SIZE_MIN       = 2
BRUSH_SIZE_MAX       = 40
ERASER_SIZE_MIN      = 10
ERASER_SIZE_MAX      = 100
BRUSH_SIZE_STEP      = 2
ERASER_SIZE_STEP     = 5
SMOOTHER_ALPHA       = 0.6     # Higher = more responsive, slightly more jitter
DEBOUNCE_FRAMES      = 4       # Fewer frames = faster mode switching
SAVE_DIR             = os.path.dirname(os.path.abspath(__file__))

# Inference resolution: model runs on a downscaled copy for speed.
# Landmarks are returned as normalized [0,1] coords so drawing accuracy
# on the full-res display frame is unaffected.
INFER_W, INFER_H     = 640, 360

# Webcam target resolution (lower than native 1920×1080 for better FPS)
CAP_W,   CAP_H       = 1280, 720

# MediaPipe landmark connections (index pairs) for skeleton drawing
# Based on the 21-point hand topology
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # Thumb
    (0,5),(5,6),(6,7),(7,8),           # Index
    (0,9),(9,10),(10,11),(11,12),      # Middle
    (0,13),(13,14),(14,15),(15,16),    # Ring
    (0,17),(17,18),(18,19),(19,20),    # Pinky
    (5,9),(9,13),(13,17),              # Palm cross-connections
]


# ── Skeleton drawing (Tasks API) ──────────────────────────────────────────────

def draw_hand_skeleton(frame, landmarks_tensor, frame_w, frame_h):
    """
    Draw hand skeleton overlay using landmarks from the PyTorch tensor.

    Uses the landmark tensor (21×3 normalized coords) and draws the
    pre-defined HAND_CONNECTIONS as lines, plus dots at each keypoint.

    Args:
        frame: BGR display frame (modified in-place).
        landmarks_tensor: torch.Tensor (21, 3) of normalized coords.
        frame_w, frame_h: Frame dimensions in pixels.
    """
    # Convert all landmarks to pixel coords
    pts = []
    for i in range(21):
        px = int(float(landmarks_tensor[i, 0]) * frame_w)
        py = int(float(landmarks_tensor[i, 1]) * frame_h)
        pts.append((px, py))

    # Draw connections
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 80), 1, cv2.LINE_AA)

    # Draw landmark dots
    for i, pt in enumerate(pts):
        # Fingertips get larger dots
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, pt, r, (0, 220, 120), -1)
        cv2.circle(frame, pt, r, (255, 255, 255), 1)


# ── Main function ─────────────────────────────────────────────────────────────

def main():
    # ── Init webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Exiting.")
        sys.exit(1)

    # Cap resolution for higher FPS (was 1920×1080 by default — very slow)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Buffer size 1 = always grab the freshest frame, no queued stale frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Webcam opened at {frame_w}×{frame_h}")

    # ── Init modules ─────────────────────────────────────────────────────────
    print("[INFO] Loading hand landmark model...")
    model     = HandLandmarkModel(
                    min_detection_confidence=CONFIDENCE_THRESHOLD,
                    min_tracking_confidence=0.5,
                )
    canvas    = Canvas(frame_w, frame_h)
    state_mgr = StateManager(debounce_frames=DEBOUNCE_FRAMES)
    smoother  = ExponentialSmoother(alpha=SMOOTHER_ALPHA)
    print("[INFO] Model loaded.")

    # ── Application state ────────────────────────────────────────────────────
    current_color   = INITIAL_COLOR
    brush_size      = INITIAL_BRUSH_SIZE
    eraser_size     = INITIAL_ERASER_SIZE
    show_landmarks  = True

    fps             = 0.0
    frame_times     = []

    print("[INFO] Starting Air Drawing System.")
    print("       Controls: ESC=quit | C=clear | S=save | ↑↓=brush | ←→=eraser | L=landmarks")

    # ── Create window ────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        t_start = time.perf_counter()

        # 1. Capture frame ────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed. Retrying…")
            continue

        # Mirror the frame so it behaves like looking in a mirror
        frame = cv2.flip(frame, 1)

        # Create a small copy for inference (much faster than full resolution)
        # Normalized landmark coords map back to the full display frame.
        infer_frame = cv2.resize(frame, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
        frame_rgb   = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)

        # 2. Run hand landmark model (PyTorch inference, torch.no_grad() inside)
        landmarks_tensor, confidence = model(frame_rgb)

        # 3. Evaluate confidence ───────────────────────────────────────────────
        detected = (landmarks_tensor is not None and confidence >= CONFIDENCE_THRESHOLD)

        if not detected:
            state_mgr.reset()
            smoother.reset()

        # 4–7. Gesture handling ───────────────────────────────────────────────
        if detected:
            # 4a. Finger states from landmark geometry (pure math, no screen zones)
            finger_states = get_finger_states(landmarks_tensor)

            # 4b. Resolve gesture → (mode, color)
            requested_mode, current_color = detect_gesture(finger_states, current_color)

            # 5. Update FSM with debounce
            confirmed_mode = state_mgr.update(requested_mode)

            # 6. Smooth index fingertip (landmark 8) coordinates with EMA
            raw_tip    = get_pixel_coords(landmarks_tensor, 8, frame_w, frame_h)
            smooth_tip = smoother.smooth(raw_tip)

            # 7. Act on mode ──────────────────────────────────────────────────
            if confirmed_mode == MODE_DRAW:
                color_bgr = COLOR_MAP[current_color]

                if state_mgr.prev_point is not None:
                    # Interpolate to fill gaps for fast-moving hand
                    for pt in interpolate_points(state_mgr.prev_point, smooth_tip):
                        canvas.draw(pt, color_bgr, brush_size)
                else:
                    canvas.draw(smooth_tip, color_bgr, brush_size)

                state_mgr.prev_point = smooth_tip

            elif confirmed_mode == MODE_ERASE:
                if state_mgr.prev_point is not None:
                    for pt in interpolate_points(state_mgr.prev_point, smooth_tip):
                        canvas.erase(pt, eraser_size)
                else:
                    canvas.erase(smooth_tip, eraser_size)

                state_mgr.prev_point = smooth_tip
                # Show eraser cursor ring on display frame
                canvas.draw_eraser_cursor(frame, smooth_tip, eraser_size)

            else:
                # IDLE — reset stroke continuity so next DRAW starts fresh
                state_mgr.prev_point = None
                # Note: don't reset smoother here — keeping recent position
                # makes the first DRAW frame's tip position more accurate.

            # Draw skeleton overlay (using our own tensor-based drawer)
            if show_landmarks:
                draw_hand_skeleton(frame, landmarks_tensor, frame_w, frame_h)

            # Fingertip indicator dot
            if confirmed_mode in (MODE_DRAW, MODE_ERASE):
                indicator_color = (
                    COLOR_MAP[current_color] if confirmed_mode == MODE_DRAW
                    else (200, 200, 200)
                )
                cv2.circle(frame, smooth_tip, 8, indicator_color, 2)
                cv2.circle(frame, smooth_tip, 3, (255, 255, 255), -1)

        else:
            # No hand detected — show notice
            confirmed_mode = MODE_IDLE
            cv2.putText(
                frame,
                "Searching for hand...",
                (frame_w // 2 - 150, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 255), 2, cv2.LINE_AA,
            )

        # 8. Blend canvas onto frame ──────────────────────────────────────────
        display = canvas.blend(frame, alpha=0.85)

        # 9. HUD status bar ───────────────────────────────────────────────────
        draw_status_bar(
            display,
            mode       = confirmed_mode if detected else MODE_IDLE,
            color_name = current_color,
            brush_size = brush_size,
            eraser_size= eraser_size,
            fps        = fps,
        )

        # 10. FPS ─────────────────────────────────────────────────────────────
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

        # 11. Show ────────────────────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, display)

        # 12. Keyboard controls ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:                              # ESC → quit
            print("[INFO] ESC pressed. Exiting.")
            break

        elif key in (ord('c'), ord('C')):          # C → clear canvas
            canvas.clear()
            smoother.reset()
            state_mgr.reset()
            print("[INFO] Canvas cleared.")

        elif key in (ord('s'), ord('S')):          # S → save PNG
            saved_path = canvas.save(SAVE_DIR)
            print(f"[INFO] Drawing saved: {saved_path}")
            cv2.putText(
                display, "Saved!",
                (frame_w // 2 - 55, frame_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3, cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, display)
            cv2.waitKey(500)

        elif key in (ord('l'), ord('L')):          # L → landmark toggle
            show_landmarks = not show_landmarks
            print(f"[INFO] Landmarks: {'ON' if show_landmarks else 'OFF'}")

        elif key in (82, 0):                       # ↑ → bigger brush
            brush_size = min(brush_size + BRUSH_SIZE_STEP, BRUSH_SIZE_MAX)

        elif key in (84, 1):                       # ↓ → smaller brush
            brush_size = max(brush_size - BRUSH_SIZE_STEP, BRUSH_SIZE_MIN)

        elif key in (83, 3):                       # → → bigger eraser
            eraser_size = min(eraser_size + ERASER_SIZE_STEP, ERASER_SIZE_MAX)

        elif key in (81, 2):                       # ← → smaller eraser
            eraser_size = max(eraser_size - ERASER_SIZE_STEP, ERASER_SIZE_MIN)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[INFO] Releasing resources…")
    model.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Goodbye.")


if __name__ == "__main__":
    main()
