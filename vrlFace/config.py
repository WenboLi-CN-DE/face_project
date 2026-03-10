"""
向后兼容层 — 请使用 vrlFace.face.config

此文件保留是为了不破坏直接导入 `vrlFace.config` 的旧代码。
新代码请使用:
    from vrlFace.face.config import config, FaceConfig
"""

from .face.config import (  # noqa: F401
    FaceConfig,
    config,
    DEFAULT_CONFIG,
    CPU_FAST_CONFIG,
    GPU_HIGH_ACCURACY_CONFIG,
    STRICT_CONFIG,
    LOOSE_CONFIG,
)

__all__ = [
    "FaceConfig",
    "config",
    "DEFAULT_CONFIG",
    "CPU_FAST_CONFIG",
    "GPU_HIGH_ACCURACY_CONFIG",
    "STRICT_CONFIG",
    "LOOSE_CONFIG",
]
