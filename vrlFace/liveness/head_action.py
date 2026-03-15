"""Head action event detection - Peak-to-Peak method.

Simple and reliable detection based on angle range in sliding window.
No baseline tracking, no drift issues.

Goals:
- Detect head movements by measuring angle range in sliding window
- Simple logic similar to blink detection (absolute thresholds)
- Keep detector public API stable: detect(pitch, yaw) -> action_name
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque


@dataclass
class HeadActionConfig:
    yaw_threshold: float = 8.0
    pitch_threshold: float = 8.0

    nod_yaw_gate_ratio: float = 0.6

    window_size: int = 30

    confirm_frames: int = 2

    cooldown_frames: int = 5


class HeadActionDetector:
    """Peak-to-peak head action detector."""

    def __init__(self, config: HeadActionConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self.pitch_history: deque = deque(maxlen=self.cfg.window_size)
        self.yaw_history: deque = deque(maxlen=self.cfg.window_size)

        self._cooldown: int = 0
        self._pending_action: str = "none"
        self._pending_count: int = 0

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
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)

        if self._cooldown > 0:
            self._cooldown -= 1
            return "none"

        if len(self.pitch_history) < 5:
            return "none"

        pitch_range = max(self.pitch_history) - min(self.pitch_history)
        yaw_range = max(self.yaw_history) - min(self.yaw_history)

        if yaw_range >= self.cfg.yaw_threshold:
            recent_yaws = list(self.yaw_history)[-10:]
            if len(recent_yaws) >= 2:
                if recent_yaws[-1] > recent_yaws[0]:
                    direction = "head_turn_left"
                else:
                    direction = "head_turn_right"
                return self._confirm(direction)

        yaw_gate = self.cfg.yaw_threshold * self.cfg.nod_yaw_gate_ratio
        if pitch_range >= self.cfg.pitch_threshold and yaw_range <= yaw_gate:
            recent_pitches = list(self.pitch_history)[-10:]
            if len(recent_pitches) >= 2:
                if recent_pitches[-1] > recent_pitches[0]:
                    direction = "head_nod_down"
                else:
                    direction = "head_nod_up"
                return self._confirm(direction)

        return self._confirm("none")
