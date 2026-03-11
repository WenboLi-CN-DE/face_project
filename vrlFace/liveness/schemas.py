"""
活体检测 API 的请求与响应 Pydantic 模型
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 请求体
# ---------------------------------------------------------------------------


class ThresholdConfig(BaseModel):
    liveness_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="活体判定阈值"
    )
    action_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="单动作通过阈值"
    )


class ActionConfig(BaseModel):
    max_video_duration: Optional[float] = Field(
        None, ge=1.0, description="最大分析时长（秒），超过则截断"
    )
    per_action_timeout: Optional[float] = Field(
        None, ge=0.5, description="每个动作的时间窗口（秒）"
    )


class MoveLivenessRequest(BaseModel):
    request_id: str = Field(..., description="请求唯一 ID")
    task_id: str = Field(..., description="任务 ID")
    video_path: str = Field(..., description="视频文件绝对路径")
    actions: List[str] = Field(
        default=["blink"],
        description="要求的动作列表，支持：blink / mouth_open / shake_head / nod / turn_left / turn_right",
    )
    threshold_config: ThresholdConfig = Field(default_factory=ThresholdConfig)
    action_config: Optional[ActionConfig] = Field(None, description="时间维度配置")


class MoveLivenessAsyncRequest(BaseModel):
    """异步活体检测请求"""

    request_id: str = Field(..., description="请求唯一 ID")
    task_id: str = Field(..., description="任务 ID")
    video_path: str = Field(..., description="视频文件绝对路径")
    actions: List[str] = Field(
        default=["blink"],
        description="要求的动作列表，支持：blink / mouth_open / shake_head / nod / turn_left / turn_right",
    )
    callback_url: str = Field(..., description="回调地址")
    callback_secret: Optional[str] = Field(
        None, description="签名密钥（可选，默认使用系统配置）"
    )
    threshold_config: ThresholdConfig = Field(default_factory=ThresholdConfig)
    action_config: Optional[ActionConfig] = Field(None, description="时间维度配置")


# ---------------------------------------------------------------------------
# 响应体
# ---------------------------------------------------------------------------


class FaceInfoResponse(BaseModel):
    confidence: float
    quality_score: float


class ActionDetailResponse(BaseModel):
    action: str
    passed: bool
    confidence: float
    msg: str


class ActionVerifyResponse(BaseModel):
    passed: bool
    required_actions: List[str]
    action_details: List[ActionDetailResponse]


class MoveLivenessData(BaseModel):
    is_liveness: int = Field(description="全局活体判定：1=通过，0=不通过")
    liveness_confidence: float
    is_face_exist: int = Field(description="是否检测到人脸：1=是，0=否")
    face_info: Optional[FaceInfoResponse] = None
    action_verify: ActionVerifyResponse


class MoveLivenessResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: Optional[MoveLivenessData] = None
    request_id: str = ""
    task_id: str = ""


class MoveLivenessAsyncResponse(BaseModel):
    """异步活体检测响应（立即返回）"""

    code: int = 0
    msg: str = "任务已接收"
    task_id: str
    estimated_time: Optional[int] = Field(None, description="预计处理时间（秒）")


class LivenessCallbackData(BaseModel):
    """回调数据"""

    is_liveness: int = Field(description="全局活体判定：1=通过，0=不通过")
    liveness_confidence: float
    is_face_exist: int = Field(description="是否检测到人脸：1=是，0=否")
    face_info: Optional[FaceInfoResponse] = None
    action_verify: ActionVerifyResponse


class LivenessCallbackRequest(BaseModel):
    """回调请求体"""

    code: int
    msg: str
    task_id: str
    data: Optional[LivenessCallbackData]
