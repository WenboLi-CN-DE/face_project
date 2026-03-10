"""
向后兼容层 — 请使用 vrlFace.face.recognizer

此文件保留是为了不破坏直接导入 `vrlFace.models` 的旧代码。
新代码请使用:
    from vrlFace.face.recognizer import face_detection, gen_verify_res, face_search
"""

from .face.recognizer import (  # noqa: F401
    face_detection,
    gen_verify_res,
    face_search,
    verify_face,
    detection_face_exits,
    get_recognizer,
)

__all__ = [
    "face_detection",
    "gen_verify_res",
    "face_search",
    "verify_face",
    "detection_face_exits",
    "get_recognizer",
]
