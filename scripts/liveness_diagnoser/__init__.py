"""活体诊断工具 - 分析活体检测日志并生成诊断报告。"""

__version__ = "1.0.0"

from .models import (
    ActionDetail,
    DiagnosisResult,
    FetchConfig,
    FrameAnalysis,
    VideoInfo,
)

__all__ = [
    "VideoInfo",
    "FrameAnalysis",
    "ActionDetail",
    "DiagnosisResult",
    "FetchConfig",
    "__version__",
]
