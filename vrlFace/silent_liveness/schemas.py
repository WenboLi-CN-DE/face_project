"""
静默活体检测 API 的请求与响应 Pydantic 模型
"""

from typing import Optional, Dict
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

    is_liveness: int = Field(description="活体判定：1=真人，0=伪造")
    confidence: float = Field(description="活体置信度 (0-1)")
    is_face_exist: int = Field(description="是否检测到人脸：1=是，0=否")
    face_exist_confidence: float = Field(description="人脸检测置信度 (0-1)")
    reject_reason: Optional[str] = Field(
        default=None,
        description="拒绝原因：null=通过，'no_face'=无人脸，'traditional_spoof'=传统攻击，'ai_spoof'=AI 生成",
    )
    details: Optional[Dict[str, bool]] = Field(
        default=None,
        description="检测详情：{'uniface_passed': bool, 'frequency_check_passed': bool, 'heuristic_check_passed': bool}",
    )


class SilentLivenessResponse(BaseModel):
    """静默活体检测响应"""

    code: int = 0
    msg: str = "silent liveness checking successful"
    liveness_results: Optional[LivenessResult] = None
    filename: str = Field(description="输入文件路径")
