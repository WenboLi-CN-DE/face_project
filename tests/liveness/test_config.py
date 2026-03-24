import os
import pytest
from vrlFace.liveness.config import LivenessConfig


def test_silent_detection_config_defaults():
    config = LivenessConfig()
    assert config.enable_silent_detection is False
    assert config.silent_detection_mode == "strict"
    assert config.silent_sample_frames == 5


def test_video_anti_spoofing_with_silent_config():
    config = LivenessConfig.video_anti_spoofing_with_silent_config()
    assert config.enable_silent_detection is True
    assert config.silent_detection_mode == "strict"
    assert config.silent_sample_frames == 5


def test_from_env_silent_detection(monkeypatch):
    monkeypatch.setenv("LIVENESS_ENABLE_SILENT", "true")
    monkeypatch.setenv("LIVENESS_SILENT_MODE", "loose")
    monkeypatch.setenv("LIVENESS_SILENT_SAMPLE_FRAMES", "3")

    config = LivenessConfig.from_env()
    assert config.enable_silent_detection is True
    assert config.silent_detection_mode == "loose"
    assert config.silent_sample_frames == 3
