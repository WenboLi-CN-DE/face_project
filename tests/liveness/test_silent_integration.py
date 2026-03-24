import pytest
from vrlFace.liveness import VideoLivenessAnalyzer, LivenessConfig


def test_silent_detection_disabled_by_default():
    """测试静默检测默认禁用"""
    config = LivenessConfig()
    analyzer = VideoLivenessAnalyzer(config)
    assert analyzer._silent_detector is None


def test_silent_detection_enabled():
    """测试静默检测启用"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    analyzer = VideoLivenessAnalyzer(config)
    assert analyzer._silent_detector is not None


def test_silent_detection_strict_mode_rejects():
    """测试严格模式拒绝伪造视频"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    config.silent_detection_mode = "strict"
    analyzer = VideoLivenessAnalyzer(config)

    # 需要准备测试视频
    # result = analyzer.analyze("fake_video.mp4", required_actions=[])
    # assert result.is_liveness == 0
    # assert result.reject_reason in ["traditional_spoof", "ai_spoof"]


def test_silent_detection_loose_mode():
    """测试宽松模式降低置信度"""
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    config.silent_detection_mode = "loose"
    analyzer = VideoLivenessAnalyzer(config)

    # 需要准备测试视频
    pass
