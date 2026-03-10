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

注意:
    顶层 __init__.py 不再主动导入子模块，避免 liveness 服务启动时
    强制加载人脸模块（反之亦然）。各服务入口直接从子包导入所需内容。
    向后兼容的符号通过 __getattr__ 懒加载提供。
"""

__version__ = "3.1.0"

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


def __getattr__(name: str):
    """
    懒加载：仅在实际使用时才导入对应子模块。
    这样 liveness 服务启动时不会触发 face 模块加载，反之亦然。
    """
    _face_symbols = {
        "face_detection", "gen_verify_res", "face_search", "verify_face",
        "detection_face_exits", "get_recognizer", "config", "FaceConfig",
    }
    _liveness_symbols = {"LivenessFusionEngine", "LivenessConfig"}

    if name in _face_symbols:
        import sys
        from vrlFace.face.config import config as _cfg, FaceConfig as _FC
        from vrlFace.face.recognizer import (
            face_detection, gen_verify_res, face_search,
            verify_face, detection_face_exits, get_recognizer,
        )
        _m = sys.modules[__name__]
        _m.config = _cfg
        _m.FaceConfig = _FC
        _m.face_detection = face_detection
        _m.gen_verify_res = gen_verify_res
        _m.face_search = face_search
        _m.verify_face = verify_face
        _m.detection_face_exits = detection_face_exits
        _m.get_recognizer = get_recognizer
        return getattr(_m, name)

    if name in _liveness_symbols:
        import sys
        from vrlFace.liveness import LivenessFusionEngine as _LFE, LivenessConfig as _LC
        _m = sys.modules[__name__]
        _m.LivenessFusionEngine = _LFE
        _m.LivenessConfig = _LC
        return getattr(_m, name)

    raise AttributeError(f"module 'vrlFace' has no attribute {name!r}")

