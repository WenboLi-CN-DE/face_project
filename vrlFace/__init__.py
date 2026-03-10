"""
vrlFace — 人脸识别 & 活体检测综合包

两大子模块:
    vrlFace.face      — 人脸识别（检测 / 1:1比对 / 1:N搜索）
    vrlFace.liveness  — 活体检测（MediaPipe 动作检测）

顶层快速导入（向后兼容原有调用方式）:
    from vrlFace import face_detection, gen_verify_res, face_search
    from vrlFace import LivenessFusionEngine, LivenessConfig

启动 API 服务:
    python -m vrlFace.main_fastapi

人脸识别命令行:
    python -m vrlFace.face.cli --demo

活体检测命令行:
    python -m vrlFace.liveness.cli --camera 0
    python -m vrlFace.liveness.cli --video path/to/video.mp4

活体 CSV 录制:
    python -m vrlFace.liveness.recorder --video path/to/video.mp4
"""

# ── 人脸识别 API（向后兼容）──────────────────────────────────────────
from .face import face_detection, gen_verify_res, face_search
from .face.config import config, FaceConfig

# ── 活体检测 API ─────────────────────────────────────────────────────
from .liveness import LivenessFusionEngine, LivenessConfig

__version__ = "3.0.0"
__all__ = [
    # 人脸识别
    "face_detection",
    "gen_verify_res",
    "face_search",
    "config",
    "FaceConfig",
    # 活体检测
    "LivenessFusionEngine",
    "LivenessConfig",
]
