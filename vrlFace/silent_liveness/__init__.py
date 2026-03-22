"""
vrlFace.silent_liveness — 静默活体检测模块

基于 deepface-antispoofing 的被动式活体检测。
无需用户配合动作，通过图像纹理/频域分析判定真伪。

主要类:
    SilentLivenessDetector — 静默活体检测器（单例）

使用示例:
    from vrlFace.silent_liveness import SilentLivenessDetector

    detector = SilentLivenessDetector.get_instance()
    result = detector.detect("/path/to/image.jpg")
    print(result["is_liveness"], result["confidence"])
"""

from .detector import SilentLivenessDetector

__all__ = ["SilentLivenessDetector"]
