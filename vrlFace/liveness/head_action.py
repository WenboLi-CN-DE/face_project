"""Head action event detection.

This module upgrades head action recognition from a simple peak-to-peak threshold
inside a sliding window to an event-based state machine.

Goals:
- Detect *slow* nod up/down by tracking a stable baseline and accumulated motion.
- Reduce false positives: head turn (yaw) shouldn't trigger nod (pitch).
- Keep detector public API stable: callers still call detect_head_action(pitch, yaw)
  and get one of: head_turn_left | head_turn_right | head_nod_down | head_nod_up | none.

The algorithm:
- Maintain exponentially-smoothed baselines for pitch/yaw when the head is stable.
- Compute deltas: dp=pitch-baseline_pitch, dy=yaw-baseline_yaw.
- Turn detection: trigger when |dy| crosses a threshold, require some persistence.
- Nod detection: trigger when |dp| crosses threshold AND yaw activity is low.
- After trigger: cooldown frames to avoid repeated fire and to let baseline settle.

This is designed to be lightweight and not depend on MediaPipe/OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeadActionConfig:
    yaw_threshold: float = 15.0
    pitch_threshold: float = 15.0

    # How small yaw must be (relative to yaw_threshold) to allow nod classification.
    nod_yaw_gate_ratio: float = 0.35  # stricter than the old 0.6

    # Baseline update.
    baseline_alpha: float = 0.08  # EMA speed when stable
    stable_dp: float = 2.0        # deg
    stable_dy: float = 2.5        # deg

    # Guard: once pitch/yaw delta grows beyond this, stop adapting baseline so
    # very slow intentional motion isn't "absorbed" by baseline drift.
    baseline_freeze_dp: float = 6.0
    baseline_freeze_dy: float = 8.0

    # Require a few consecutive frames over threshold to confirm.
    confirm_frames: int = 2

    # Cooldown after firing, in frames.
    cooldown_frames: int = 10


class HeadActionDetector:
    """Event-based head action detector."""

    def __init__(self, config: HeadActionConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._baseline_pitch: float | None = None
        self._baseline_yaw: float | None = None

        self._cooldown: int = 0
        self._pending_action: str = "none"
        self._pending_count: int = 0

        # Track last deltas for direction estimation.
        self._last_dp: float = 0.0
        self._last_dy: float = 0.0

    def _init_baseline_if_needed(self, pitch: float, yaw: float) -> None:
        if self._baseline_pitch is None:
            self._baseline_pitch = float(pitch)
        if self._baseline_yaw is None:
            self._baseline_yaw = float(yaw)

    def _update_baseline_if_stable(self, dp: float, dy: float, pitch: float, yaw: float) -> None:
        # Stop baseline adaptation when deltas indicate an intentional movement.
        if abs(dp) >= self.cfg.baseline_freeze_dp or abs(dy) >= self.cfg.baseline_freeze_dy:
            return

        # Only update baseline when motion is small.
        if abs(dp) <= self.cfg.stable_dp and abs(dy) <= self.cfg.stable_dy:
            a = self.cfg.baseline_alpha
            self._baseline_pitch = (1 - a) * float(self._baseline_pitch) + a * float(pitch)
            self._baseline_yaw = (1 - a) * float(self._baseline_yaw) + a * float(yaw)

    def _confirm(self, action: str) -> str:
        if action == "none":
            self._pending_action = "none"
            self._pending_count = 0
            return "none"

        if self._pending_action != action:
            self._pending_action = action
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending_count >= max(1, self.cfg.confirm_frames):
            self._pending_action = "none"
            self._pending_count = 0
            self._cooldown = max(1, self.cfg.cooldown_frames)
            return action
        return "none"

    def detect(self, pitch: float, yaw: float) -> str:
        """Return a head action label or 'none'."""
        self._init_baseline_if_needed(pitch, yaw)

        if self._cooldown > 0:
            self._cooldown -= 1
            # During cooldown: still gently adapt baseline to current pose.
            dp_cd = float(pitch) - float(self._baseline_pitch)
            dy_cd = float(yaw) - float(self._baseline_yaw)
            self._update_baseline_if_stable(dp_cd, dy_cd, pitch, yaw)
            self._last_dp = dp_cd
            self._last_dy = dy_cd
            return "none"

        dp = float(pitch) - float(self._baseline_pitch)
        dy = float(yaw) - float(self._baseline_yaw)

        # Update baseline if stable.
        self._update_baseline_if_stable(dp, dy, pitch, yaw)

        # Recompute after baseline update.
        dp = float(pitch) - float(self._baseline_pitch)
        dy = float(yaw) - float(self._baseline_yaw)

        # 1) Turn detection: yaw dominates.
        if abs(dy) >= self.cfg.yaw_threshold:
            direction = "head_turn_left" if dy > 0 else "head_turn_right"
            self._last_dp = dp
            self._last_dy = dy
            return self._confirm(direction)

        # 2) Nod detection: pitch dominates AND yaw must be quiet.
        yaw_gate = self.cfg.yaw_threshold * self.cfg.nod_yaw_gate_ratio
        if abs(dp) >= self.cfg.pitch_threshold and abs(dy) <= yaw_gate:
            direction = "head_nod_down" if dp > 0 else "head_nod_up"
            self._last_dp = dp
            self._last_dy = dy
            return self._confirm(direction)

        self._last_dp = dp
        self._last_dy = dy
        return self._confirm("none")

