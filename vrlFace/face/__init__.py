"""
vrlFace.face — 人脸识别子包

公开导出常用函数，方便直接从包导入：
    from vrlFace.face import face_detection, gen_verify_res, face_search
"""

from .recognizer import (  # noqa: F401
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
