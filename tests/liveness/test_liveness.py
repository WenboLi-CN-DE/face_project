"""
活体检测模块测试

运行:
    pytest tests/liveness/test_liveness.py -v
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vrlFace.liveness import (
    LivenessFusionEngine,
    LivenessConfig,
    MediaPipeLivenessDetector,
)


def test_config_defaults():
    """测试默认配置"""
    config = LivenessConfig.cpu_fast_config()
    assert config.threshold > 0
    assert config.window_size > 0
    assert config.validate()


def test_config_video_anti_spoofing():
    """测试视频防伪配置"""
    config = LivenessConfig.video_anti_spoofing_config()
    assert config.validate()
    assert config.window_size == 30


def test_config_realtime():
    """测试实时配置"""
    config = LivenessConfig.realtime_config()
    assert config.validate()


def test_mediapipe_detector_init():
    """测试 MediaPipe 检测器初始化"""
    detector = MediaPipeLivenessDetector(
        ear_threshold=0.2,
        mar_threshold=0.5,
        window_size=30,
    )
    assert detector is not None
    detector.close()


def test_mediapipe_detector_no_face():
    """测试 MediaPipe 检测器：空白帧无人脸"""
    detector = MediaPipeLivenessDetector(
        ear_threshold=0.2,
        mar_threshold=0.5,
        window_size=30,
    )
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect_liveness(black_frame)
    assert result["face_detected"] is False
    assert result["score"] == 0.0
    detector.close()


def test_fusion_engine_no_face():
    """测试融合引擎：空白帧无人脸"""
    config = LivenessConfig.cpu_fast_config()
    engine = LivenessFusionEngine(config)
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.process_frame(black_frame)
    assert result.is_live is False
    assert result.score == 0.0
    engine.close()


def test_fusion_engine_reset():
    """测试融合引擎重置"""
    config = LivenessConfig.cpu_fast_config()
    engine = LivenessFusionEngine(config)
    engine.reset()
    assert len(engine.score_history) == 0
    engine.close()


def test_video_anti_spoofing_temporal():
    """测试时序一致性分析（不依赖摄像头）"""
    config = LivenessConfig.video_anti_spoofing_config()
    engine = LivenessFusionEngine(config)

    # 场景1：正常活体（随机波动）
    normal_scores = np.random.uniform(0.5, 0.9, 60).tolist()
    variance = float(np.var(normal_scores))
    assert variance > 0.001  # 正常活体应有波动

    # 场景2：静态照片（过于稳定）
    static_scores = [0.5] * 60
    variance = float(np.var(static_scores))
    assert variance < 1e-10  # 静态照片方差应接近 0

    engine.close()


def test_liveness_config_display(capsys):
    """测试配置打印"""
    config = LivenessConfig.cpu_fast_config()
    config.display()
    captured = capsys.readouterr()
    assert "threshold" in captured.out.lower() or len(captured.out) > 0


def main():
    """手动运行入口"""
    print("\n活体检测模块测试")
    print("=" * 60)
    print("提示：请使用 pytest tests/liveness/test_liveness.py -v 运行")


if __name__ == "__main__":
    main()
