"""
静默活体检测 API 的请求与响应 Pydantic 模型
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 请求体
# ---------------------------------------------------------------------------


class SilentLivenessRequest(BaseModel):
    """静默活体检测请求 — JSON body 传文件路径"""

    picture_path: str = Field(..., description="图片文件绝对路径")


# ---------------------------------------------------------------------------
# 响应体
# ---------------------------------------------------------------------------


class LivenessResult(BaseModel):
    """活体检测结果"""

    success: str = Field(description="检测是否成功：True/False")
    is_real: str = Field(description="是否真人：True/False")
    confidence: float = Field(description="置信度 (0-1)")
    spoof_type: str = Field(description="类型：Real Face/Spoof/Error")
    processing_time: float = Field(description="处理耗时（秒）")


class SilentLivenessResponse(BaseModel):
    """静默活体检测响应"""

    code: int = 0
    msg: str = "silent liveness checking successful"
    liveness_results: Optional[LivenessResult] = None
    filename: str = Field(description="输入文件路径")
