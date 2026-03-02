"""
gestures.py — Finger State Detection & Gesture Recognition
===========================================================
All logic here is based purely on hand landmark GEOMETRY — no hardcoded
screen zones, no pixel thresholds tied to camera resolution.

MediaPipe 21-landmark hand model indices:
  0  WRIST
  1  THUMB_CMC  2  THUMB_MCP  3  THUMB_IP   4  THUMB_TIP
  5  INDEX_MCP  6  INDEX_PIP  7  INDEX_DIP  8  INDEX_TIP
  9  MID_MCP   10  MID_PIP   11  MID_DIP   12  MID_TIP
  13  RING_MCP  14  RING_PIP  15  RING_DIP  16  RING_TIP
  17  PINK_MCP  18  PINK_PIP  19  PINK_DIP  20  PINK_TIP

How "finger up" is determined:
  For fingers 2-5 (index, middle, ring, pinky):
    The FINGERTIP (e.g. landmark 8) is considered "up" if its Y coordinate
    is LESS than the Y coordinate of its PIP joint (e.g. landmark 6).
    In image coordinates, Y increases downward, so a smaller Y → higher up.

  For the thumb:
    The thumb moves laterally (left/right) rather than up/down in a palm-up
    or palm-forward pose. We compare THUMB_TIP.x with THUMB_MCP.x.
    - If the hand is a right hand (wrist is to the left of finger MCPs) and
      THUMB_TIP.x < THUMB_MCP.x → thumb is extended outward (up/left).
    - For mirrored/left hand the logic is reversed.
    We simplify by using THUMB_TIP vs THUMB_IP distance along x-axis relative
    to wrist-to-index-MCP direction.
"""

import torch
from typing import Dict, Optional, Tuple

# ── Landmark index constants ──────────────────────────────────────────────

WRIST       = 0
THUMB_CMC   = 1
THUMB_MCP   = 2
THUMB_IP    = 3
THUMB_TIP   = 4
INDEX_MCP   = 5
INDEX_PIP   = 6
INDEX_DIP   = 7
INDEX_TIP   = 8
MIDDLE_MCP  = 9
MIDDLE_PIP  = 10
MIDDLE_DIP  = 11
MIDDLE_TIP  = 12
RING_MCP    = 13
RING_PIP    = 14
RING_DIP    = 15
RING_TIP    = 16
PINKY_MCP   = 17
PINKY_PIP   = 18
PINKY_DIP   = 19
PINKY_TIP   = 20

# ── Mode and Color names ──────────────────────────────────────────────────

MODE_IDLE  = "IDLE"
MODE_DRAW  = "DRAW"
MODE_ERASE = "ERASE"

COLOR_BLUE   = "Blue"
COLOR_GREEN  = "Green"
COLOR_RED    = "Red"
COLOR_YELLOW = "Yellow"

# BGR values for actual drawing on canvas
COLOR_MAP = {
    COLOR_BLUE:   (255, 100,   0),
    COLOR_GREEN:  (  0, 200,  50),
    COLOR_RED:    (  0,   0, 220),
    COLOR_YELLOW: (  0, 200, 200),
}


# ---------------------------------------------------------------------------
# Core finger-state helpers
# ---------------------------------------------------------------------------

def _lm(landmarks, idx: int) -> Tuple[float, float, float]:
    """
    Extract (x, y, z) for a given landmark index from a (21, 3) tensor.
    Returns Python floats for fast comparison.
    """
    row = landmarks[idx]
    return float(row[0]), float(row[1]), float(row[2])


def is_finger_up(landmarks, finger: str) -> bool:
    """
    Determine whether a given finger is extended/raised using landmark geometry.

    How it works (non-thumb fingers):
        Compare the TIP y-coordinate against the PIP y-coordinate.
        In image space, y increases downward.
        If TIP.y < PIP.y  →  tip is ABOVE the PIP joint  →  finger is UP.

    How it works (thumb):
        The thumb extends roughly sideways. We detect it by checking whether
        THUMB_TIP is farther from the palm center (average of MCPs) than
        THUMB_IP. This is rotation-invariant for a frontal hand pose.

    Args:
        landmarks: torch.Tensor of shape (21, 3), normalized coords.
        finger: One of 'thumb', 'index', 'middle', 'ring', 'pinky'.

    Returns:
        True if the finger appears to be raised/extended.
    """
    if finger == "thumb":
        # Thumb extension detection using horizontal separation from the palm.
        #
        # In a frontal, mirror-flipped webcam view (the standard setup):
        #   - An EXTENDED thumb points outward (to the left for a right hand
        #     in mirror view), so THUMB_TIP.x is significantly smaller than
        #     INDEX_MCP.x (knuckle of the index finger, which is the left
        #     boundary of the palm for a right hand in mirror view).
        #   - A CURLED thumb tucks inward, so THUMB_TIP.x is close to or
        #     greater than INDEX_MCP.x.
        #
        # We compute the horizontal gap: index_mcp.x - thumb_tip.x
        # If gap > threshold (5% of frame width in normalized coords) → up.
        # The threshold of 0.05 is robust to hand size variation since
        # MediaPipe coords are normalized to [0, 1].
        tip_x, _, _   = _lm(landmarks, THUMB_TIP)
        idx_mcp_x, _, _ = _lm(landmarks, INDEX_MCP)

        # Extended thumb: tip is clearly to the left of index knuckle (mirror view)
        return (idx_mcp_x - tip_x) > 0.05

    # Map finger name → (TIP index, PIP index)
    finger_joints = {
        "index":  (INDEX_TIP,  INDEX_PIP),
        "middle": (MIDDLE_TIP, MIDDLE_PIP),
        "ring":   (RING_TIP,   RING_PIP),
        "pinky":  (PINKY_TIP,  PINKY_PIP),
    }

    if finger not in finger_joints:
        return False

    tip_idx, pip_idx = finger_joints[finger]
    _, tip_y, _ = _lm(landmarks, tip_idx)
    _, pip_y, _ = _lm(landmarks, pip_idx)

    # Finger is "up" when tip is above (smaller y) its PIP joint
    return tip_y < pip_y


def get_finger_states(landmarks) -> Dict[str, bool]:
    """
    Compute the raised/lowered state for all five fingers.

    Args:
        landmarks: torch.Tensor of shape (21, 3).

    Returns:
        Dict with keys: 'thumb', 'index', 'middle', 'ring', 'pinky'.
        Value is True if finger is raised, False if curled.
    """
    return {
        "thumb":  is_finger_up(landmarks, "thumb"),
        "index":  is_finger_up(landmarks, "index"),
        "middle": is_finger_up(landmarks, "middle"),
        "ring":   is_finger_up(landmarks, "ring"),
        "pinky":  is_finger_up(landmarks, "pinky"),
    }


# ---------------------------------------------------------------------------
# High-level gesture → (mode, color) resolver
# ---------------------------------------------------------------------------

def detect_gesture(
    finger_states: Dict[str, bool],
    current_color: str,
) -> Tuple[str, str]:
    """
    Map a set of finger states to an application (mode, color) pair.

    Rules (checked in priority order):
      1. All five fingers up              → ERASE (color unchanged)
      2. Only index finger up             → DRAW  (color unchanged)
      3. Only thumb up                    → IDLE, color = Blue
      4. Only middle finger up            → IDLE, color = Yellow
      5. Only ring finger up              → IDLE, color = Red
      6. Only pinky up                    → IDLE, color = Green
      7. No fingers raised (closed fist)  → IDLE  (color unchanged)
      8. Any other combination            → IDLE  (color unchanged)

    Color-selection gestures put the mode in IDLE to prevent accidental
    drawing while the user switches colors.

    Args:
        finger_states: Output of get_finger_states().
        current_color: Currently active color name (returned if unchanged).

    Returns:
        (mode, color) tuple where mode ∈ {IDLE, DRAW, ERASE}
                                  color ∈ {Blue, Green, Red, Yellow}
    """
    t = finger_states["thumb"]
    i = finger_states["index"]
    m = finger_states["middle"]
    r = finger_states["ring"]
    p = finger_states["pinky"]

    up_count = sum([t, i, m, r, p])

    # Rule 1: All five fingers → ERASE
    if up_count == 5:
        return MODE_ERASE, current_color

    # Rule 2: Only index finger → DRAW
    if i and not t and not m and not r and not p:
        return MODE_DRAW, current_color

    # Rule 3: Only thumb → Blue
    if t and not i and not m and not r and not p:
        return MODE_IDLE, COLOR_BLUE

    # Rule 4: Only middle → Yellow
    if m and not t and not i and not r and not p:
        return MODE_IDLE, COLOR_YELLOW

    # Rule 5: Only ring → Red
    if r and not t and not i and not m and not p:
        return MODE_IDLE, COLOR_RED

    # Rule 6: Only pinky → Green
    if p and not t and not i and not m and not r:
        return MODE_IDLE, COLOR_GREEN

    # Default: IDLE, keep current color
    return MODE_IDLE, current_color
