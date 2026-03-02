"""
state.py — Mode State Machine with Debounce Logic
===================================================
Implements an explicit finite-state machine (FSM) that governs which mode
the application is in: IDLE, DRAW, or ERASE.

Why a state machine?
  Raw finger-state detections are noisy. Without a controlled transition
  mechanism, the mode flickers rapidly between DRAW, IDLE, ERASE as small
  hand tremors misclassify individual frames. The StateManager addresses
  this with two techniques:

  1. Debounce: A new mode is only accepted after it has been observed
     consistently for N consecutive frames. Single-frame blips are ignored.

  2. Explicit transitions: Not all state transitions are allowed. The FSM
     only permits:
        IDLE  ↔ DRAW
        IDLE  ↔ ERASE
        DRAW  → ERASE  (via IDLE implicitly in practice)
        ERASE → DRAW   (via IDLE implicitly in practice)
     Direct DRAW → ERASE transitions are allowed but require stable debounce.

Drawing state (prev_point) is tracked here because the state machine
controls when a stroke starts and ends — only the StateManager knows whether
the last frame was also a DRAW frame.
"""

from typing import Optional, Tuple
from gestures import MODE_IDLE, MODE_DRAW, MODE_ERASE


class StateManager:
    """
    Finite-state machine for application mode management.

    Valid states:
        IDLE  — hand detected but not in draw/erase gesture
        DRAW  — index finger raised, drawing on canvas
        ERASE — all fingers raised, erasing on canvas

    Debounce mechanism:
        The `_candidate` attribute holds the mode that has been requested
        but not yet confirmed. `_candidate_count` tracks how many consecutive
        frames have requested that same mode. Only when `_candidate_count`
        reaches `debounce_frames` does `_current_mode` transition.

    Args:
        debounce_frames (int): Number of consecutive frames a gesture must
            be held before the state transitions. Default 5 frames ≈ 250ms
            at 20 FPS — enough to prevent flickering, short enough to feel
            responsive.
    """

    def __init__(self, debounce_frames: int = 5):
        self.debounce_frames = debounce_frames

        # Confirmed current mode
        self._current_mode: str = MODE_IDLE

        # Candidate (pending) mode — must be stable for debounce_frames
        self._candidate: str = MODE_IDLE
        self._candidate_count: int = 0

        # Previous drawing point — used to interpolate strokes between frames.
        # Reset to None whenever drawing stops (mode leaves DRAW).
        self.prev_point: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, requested_mode: str) -> str:
        """
        Feed a new gesture-detected mode and return the confirmed current mode.

        State transition logic:
          - If requested_mode == _candidate: increment counter.
          - If requested_mode != _candidate: reset counter, update candidate.
          - If counter >= debounce_frames: confirm the mode transition.

        Args:
            requested_mode: Mode detected this frame by gestures.detect_gesture().

        Returns:
            The confirmed current mode (may be different from requested_mode
            if the gesture hasn't been held long enough).
        """
        if requested_mode == self._candidate:
            # Same request as ongoing candidate → increment stability counter
            self._candidate_count += 1
        else:
            # Different request → reset candidate
            self._candidate       = requested_mode
            self._candidate_count = 1

        # Confirm transition once gesture has been stable for debounce_frames
        if self._candidate_count >= self.debounce_frames:
            if self._candidate != self._current_mode:
                # Mode is actually changing — handle side effects
                self._on_mode_change(self._current_mode, self._candidate)
            self._current_mode = self._candidate

        return self._current_mode

    # ------------------------------------------------------------------
    # Side effects on mode change
    # ------------------------------------------------------------------

    def _on_mode_change(self, old_mode: str, new_mode: str) -> None:
        """
        Called when the confirmed mode is about to change.

        Currently:
          - Resets prev_point whenever we leave DRAW mode, so the next
            time DRAW is entered there's no stale "last position" causing
            a line from a distant point.
        """
        if old_mode == MODE_DRAW:
            # Stroke ended — clear the last known draw position
            self.prev_point = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current confirmed mode."""
        return self._current_mode

    @property
    def candidate_mode(self) -> str:
        """Pending mode (may not be confirmed yet)."""
        return self._candidate

    @property
    def candidate_progress(self) -> float:
        """
        Fraction of debounce window filled (0.0 → 1.0).
        Useful for rendering a visual progress indicator if desired.
        """
        return min(self._candidate_count / self.debounce_frames, 1.0)

    def reset(self) -> None:
        """
        Full reset — called when the hand is lost (no detection this frame).

        Resets mode to IDLE, clears debounce state, and clears prev_point
        so that when the hand reappears there's no stale stroke history.
        """
        self._current_mode    = MODE_IDLE
        self._candidate       = MODE_IDLE
        self._candidate_count = 0
        self.prev_point       = None
