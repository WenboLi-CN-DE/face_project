"""
活体检测子包

纯 MediaPipe 方案，提供:
    LivenessFusionEngine      — 活体检测融合引擎（主入口）
    LivenessConfig            — 配置类
    MediaPipeLivenessDetector — MediaPipe 完整推理检测器
    FastLivenessDetector      — 轻量快速检测器
    InsightFaceQualityDetector— InsightFace 质量检测器
    VideoLivenessAnalyzer     — 视频逐段分析器（供 API 使用）
    BenchmarkCalibrator       — 基准帧校准器（防替换攻击）
    BenchmarkConfig           — 基准帧配置类
"""

from .mediapipe_detector import MediaPipeLivenessDetector
from .fusion_engine import LivenessFusionEngine
from .config import LivenessConfig
from .fast_detector import FastLivenessDetector
from .insightface_quality import InsightFaceQualityDetector
from .video_analyzer import VideoLivenessAnalyzer
from .benchmark_calibrator import BenchmarkCalibrator, BenchmarkConfig

__all__ = [
    "MediaPipeLivenessDetector",
    "LivenessFusionEngine",
    "LivenessConfig",
    "FastLivenessDetector",
    "InsightFaceQualityDetector",
    "VideoLivenessAnalyzer",
    "BenchmarkCalibrator",
    "BenchmarkConfig",
]
