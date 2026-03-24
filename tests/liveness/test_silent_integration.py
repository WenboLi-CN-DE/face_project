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
