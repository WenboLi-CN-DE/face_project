"""
静默活体检测 FastAPI 路由

接口:
    POST /vrlSilentLiveness — 静默活体检测（JSON body 传文件路径）
"""

import logging
import os
import traceback

from fastapi import APIRouter, HTTPException

from .schemas import (
    SilentLivenessRequest,
    SilentLivenessResponse,
    LivenessResult,
)
from .detector import SilentLivenessDetector
from .config import get_config

logger = logging.getLogger(__name__)

router = APIRouter()

_config = get_config()


@router.post("/vrlSilentLiveness", response_model=SilentLivenessResponse)
async def vrl_silent_liveness(req: SilentLivenessRequest) -> SilentLivenessResponse:
    """
    静默活体检测

    通过 JSON body 传入图片文件路径，返回活体检测结果。
    无需用户做任何动作，基于图像纹理/频域分析判定真伪。

    - **picture_path**: 图片文件绝对路径
    """
    logger.info("vrlSilentLiveness request picture_path=%s", req.picture_path)

    # 路径前缀校验（安全限制）
    if not _config.is_path_allowed(req.picture_path):
        logger.warning(
            "拒绝访问：路径不在允许列表中 picture_path=%s allowed_prefixes=%s",
            req.picture_path,
            _config.allowed_prefixes,
        )
        raise HTTPException(
            status_code=400,
            detail=f"不允许的路径前缀：{req.picture_path}，允许的前缀：{', '.join(_config.allowed_prefixes)}",
        )

    # 文件存在性校验
    if not os.path.exists(req.picture_path):
        raise HTTPException(
            status_code=400,
            detail=f"图片文件不存在：{req.picture_path}",
        )

    # 文件类型校验
    ext = os.path.splitext(req.picture_path)[1].lower()
    if ext not in _config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{ext}，支持：{', '.join(sorted(_config.ALLOWED_EXTENSIONS))}",
        )

    try:
        detector = SilentLivenessDetector.get_instance()
        result = detector.detect(req.picture_path)

        return SilentLivenessResponse(
            code=0,
            msg="silent liveness checking successful",
            liveness_results=LivenessResult(**result),
            filename=req.picture_path,  # 返回原始传入路径（保持 API 一致性）
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("vrlSilentLiveness error: %s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"静默活体检测出错：{str(e)}",
        )
