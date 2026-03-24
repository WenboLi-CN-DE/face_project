import pytest
import numpy as np
from vrlFace.liveness.frame_sampler import FrameSampler


def test_frame_sampler_initialization():
    """测试采样器初始化"""
    sampler = FrameSampler()
    assert sampler is not None


def test_sample_keyframes_returns_frames(tmp_path):
    """测试采样返回帧列表"""
    # 需要真实视频文件进行测试
    # 使用项目中的测试视频
    video_path = "data/test_video.mp4"  # 假设存在
    sampler = FrameSampler()

    frames = sampler.sample_keyframes(video_path, num_frames=3)
    assert isinstance(frames, list)
    assert len(frames) <= 3
    if frames:
        assert isinstance(frames[0], np.ndarray)


def test_sample_keyframes_quality_filtering():
    """测试质量过滤"""
    sampler = FrameSampler()
    # 质量评分应该在 0-1 之间
    score = sampler._calculate_quality_score(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        face_size=0.3,
        angle=5.0,
    )
    assert 0.0 <= score <= 1.0
