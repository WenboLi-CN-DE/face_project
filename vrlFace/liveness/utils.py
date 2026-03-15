"""
活体检测共享工具函数

提供在 cli.py / recorder.py 等多处使用的公共辅助函数，
消除原 liveness_main.py 与 save_csv.py 之间的代码重复。
"""

from .config import LivenessConfig


def build_fast_detector_config(config: LivenessConfig) -> dict:
    """
    将 LivenessConfig 映射为 FastLivenessDetector 构造参数字典。

    统一阈值传递，避免各调用方各自硬编码参数导致的阈值不一致。
    """
    return {
        "ear_threshold": config.ear_threshold,
        # eye_open/close_threshold 与 ear_threshold 保持一致，
        # 避免双阈值不对称导致漏检
        "eye_open_threshold": config.ear_threshold,
        "eye_close_threshold": config.ear_threshold,
        "mar_threshold": config.mar_threshold,
        "yaw_threshold": config.yaw_threshold,
        "pitch_threshold": config.pitch_threshold,
        "window_size": config.window_size,
        "action_confirm_frames": config.action_confirm_frames,
    }


def resolve_current_action(fd_result: dict) -> str:
    """
    从 FastLivenessDetector 结果字典中提取当前动作标签。

    优先级：瞬时眨眼事件 > 张嘴事件 > 头部动作。
    使用 hold 持久化字段（blink_active / mouth_active）确保 UI 稳定显示。
    """
    if fd_result.get("blink_active", fd_result.get("blink_detected", False)):
        return "blinking"
    if fd_result.get("mouth_active", fd_result.get("mouth_open", False)):
        return "mouth_open"
    return fd_result.get("head_action", "none")

