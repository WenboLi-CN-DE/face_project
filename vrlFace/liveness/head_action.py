"""Head action event detection - Hybrid method (Peak-to-Peak + Absolute Angle).

Hybrid approach:
- Peak-to-peak detection for rapid movements (shake_head, nod)
- Absolute angle detection for sustained poses (turn_left, turn_right, nod_up, nod_down)
- Baseline tracking with automatic reset

Goals:
- Detect both rapid movements and sustained poses
- Handle single-direction actions that exceed window_size
- Keep detector public API stable: detect(pitch, yaw) -> action_name
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque


@dataclass
class HeadActionConfig:
    # Peak-to-peak thresholds (for rapid movements)
    yaw_threshold: float = 5.0  # 降低从 8.0 → 5.0
    pitch_threshold: float = 5.0  # 降低从 8.0 → 5.0

    # Absolute angle thresholds (for sustained poses)
    yaw_absolute_threshold: float = (
        12.0  # |yaw| > 12° → turn_left/turn_right (降低从 15.0°)
    )
    pitch_absolute_threshold: float = (
        12.0  # |pitch| > 12° → nod_up/nod_down (降低从 15.0°)
    )

    # 增加 nod_yaw_gate_ratio 从 0.6 到 0.75，允许更大的 yaw 变化范围
    # 原因：失败视频 yaw_range=5.0° 被 4.8°(8*0.6) 过滤掉
    nod_yaw_gate_ratio: float = 0.75

    window_size: int = 60  # 增加从 30 → 60 帧，覆盖更长的动作持续时间

    confirm_frames: int = 2

    cooldown_frames: int = 5

    # 基线重置阈值：角度变化 < 2°/帧 持续 10 帧 → 重置基线
    baseline_reset_threshold: float = 2.0
    baseline_reset_frames: int = 10


class HeadActionDetector:
    """Hybrid head action detector (peak-to-peak + absolute angle)."""

    def __init__(self, config: HeadActionConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self.pitch_history: deque = deque(maxlen=self.cfg.window_size)
        self.yaw_history: deque = deque(maxlen=self.cfg.window_size)

        self._cooldown: int = 0
        self._pending_action: str = "none"
        self._pending_count: int = 0

        # 基线追踪（用于绝对角度检测）
        self._baseline_yaw: float = 0.0
        self._baseline_pitch: float = 0.0
        self._baseline_set: bool = False
        self._stable_frames: int = 0

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

        # 更新基线（当角度稳定时）
        self._update_baseline(pitch, yaw)

        if self._cooldown > 0:
            self._cooldown -= 1
            return "none"

        if len(self.pitch_history) < 5:
            return "none"

        # 方法 1: 峰峰值检测（快速往返运动）
        action = self._detect_peak_to_peak()
        if action != "none":
            return self._confirm(action)

        # 方法 2: 绝对角度检测（持续姿势）
        action = self._detect_absolute_angle()
        if action != "none":
            return self._confirm(action)

        return self._confirm("none")

    def _update_baseline(self, pitch: float, yaw: float) -> None:
        """Update baseline when head is stable."""
        if len(self.yaw_history) < 5:
            self._baseline_yaw = yaw
            self._baseline_pitch = pitch
            self._baseline_set = True
            return

        # 检查是否稳定（角度变化 < threshold）
        yaw_range = max(self.yaw_history) - min(self.yaw_history)
        pitch_range = max(self.pitch_history) - min(self.pitch_history)

        if (
            yaw_range < self.cfg.baseline_reset_threshold
            and pitch_range < self.cfg.baseline_reset_threshold
        ):
            self._stable_frames += 1
            if self._stable_frames >= self.cfg.baseline_reset_frames:
                # 稳定足够时间，更新基线
                self._baseline_yaw = yaw
                self._baseline_pitch = pitch
                self._baseline_set = True
                self._stable_frames = 0
        else:
            self._stable_frames = 0

    def _detect_peak_to_peak(self) -> str:
        """Detect rapid movements using peak-to-peak method."""
        pitch_range = max(self.pitch_history) - min(self.pitch_history)
        yaw_range = max(self.yaw_history) - min(self.yaw_history)

        # 检测摇头（左右往返）
        if yaw_range >= self.cfg.yaw_threshold:
            recent_yaws = list(self.yaw_history)[-10:]
            if len(recent_yaws) >= 2:
                # 检测往返运动：当前方向与起始方向相反
                if recent_yaws[-1] > recent_yaws[0]:
                    return "head_turn_left"
                else:
                    return "head_turn_right"

        # 检测点头（上下往返）
        yaw_gate = self.cfg.yaw_threshold * self.cfg.nod_yaw_gate_ratio
        if pitch_range >= self.cfg.pitch_threshold and yaw_range <= yaw_gate:
            recent_pitches = list(self.pitch_history)[-10:]
            if len(recent_pitches) >= 2:
                if recent_pitches[-1] > recent_pitches[0]:
                    return "head_nod_down"
                else:
                    return "head_nod_up"

        return "none"

    def _detect_absolute_angle(self) -> str:
        """Detect sustained poses using absolute angle thresholds."""
        if not self._baseline_set:
            return "none"

        # 计算与基线的角度差
        yaw_delta = self.yaw_history[-1] - self._baseline_yaw
        pitch_delta = self.pitch_history[-1] - self._baseline_pitch

        # 检测单方向转头（绝对角度）
        if abs(yaw_delta) >= self.cfg.yaw_absolute_threshold:
            if yaw_delta > 0:
                return "head_turn_left"
            else:
                return "head_turn_right"

        # 检测单方向点头（绝对角度）
        if abs(pitch_delta) >= self.cfg.pitch_absolute_threshold:
            if pitch_delta > 0:
                return "head_nod_down"
            else:
                return "head_nod_up"

        return "none"
