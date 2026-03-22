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

    - **picture_path**: 图片文件绝对路径（支持多路径映射，如 /opt/test -> /data/videos）
    """
    logger.info("vrlSilentLiveness request picture_path=%s", req.picture_path)

    # Step 1: 路径前缀校验（安全限制）
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

    # Step 2: 路径解析（外部路径 -> 内部路径映射）
    resolved_path = _config.resolve_path(req.picture_path)
    if resolved_path != req.picture_path:
        logger.info("路径映射：%s -> %s", req.picture_path, resolved_path)

    # Step 3: 文件存在性校验
    if not os.path.exists(resolved_path):
        raise HTTPException(
            status_code=400,
            detail=f"图片文件不存在：{resolved_path}",
        )

    # Step 4: 文件类型校验
    ext = os.path.splitext(resolved_path)[1].lower()
    if ext not in _config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{ext}，支持：{', '.join(sorted(_config.ALLOWED_EXTENSIONS))}",
        )

    try:
        detector = SilentLivenessDetector.get_instance()
        result = detector.detect(resolved_path)

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
