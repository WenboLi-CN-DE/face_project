"""
活体检测 FastAPI 路由

接口:
    POST /vrlMoveLiveness — 视频活体检测 + 动作验证（异步回调）
"""

import logging
import os

from fastapi import APIRouter, HTTPException, BackgroundTasks

from .schemas import (
    MoveLivenessAsyncRequest,
    MoveLivenessAsyncResponse,
)
from .async_processor import process_liveness_task

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/vrlMoveLiveness", response_model=MoveLivenessAsyncResponse)
async def vrl_move_liveness(
    req: MoveLivenessAsyncRequest, background_tasks: BackgroundTasks
) -> MoveLivenessAsyncResponse:
    """
    视频活体检测 + 动作验证接口（异步回调）

    - 接收视频路径和动作列表
    - 立即返回 task_id
    - 后台异步处理，完成后回调通知
    """
    logger.info(
        "vrlMoveLiveness request_id=%s task_id=%s video=%s actions=%s callback=%s",
        req.request_id,
        req.task_id,
        req.video_path,
        req.actions,
        req.callback_url,
    )

    # 支持的动作白名单（先验证参数）
    SUPPORTED_ACTIONS = {
        "blink",
        "mouth_open",
        "shake_head",
        "nod",
        "nod_down",
        "nod_up",
        "turn_left",
        "turn_right",
    }
    invalid_actions = [a for a in req.actions if a not in SUPPORTED_ACTIONS]
    if invalid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的动作：{invalid_actions}，支持列表：{sorted(SUPPORTED_ACTIONS)}",
        )

    # 路径安全校验
    if not os.path.exists(req.video_path):
        raise HTTPException(
            status_code=400,
            detail=f"视频文件不存在：{req.video_path}",
        )

    # 提取配置参数
    max_duration = None
    per_action_timeout = None
    if req.action_config:
        max_duration = req.action_config.max_video_duration
        per_action_timeout = req.action_config.per_action_timeout

    # 添加后台任务
    background_tasks.add_task(
        process_liveness_task,
        task_id=req.task_id,
        request_id=req.request_id,
        video_path=req.video_path,
        actions=req.actions,
        callback_url=req.callback_url,
        liveness_threshold=req.threshold_config.liveness_threshold,
        action_threshold=req.threshold_config.action_threshold,
        max_video_duration=max_duration,
        per_action_timeout=per_action_timeout,
        callback_secret=req.callback_secret,
    )

    logger.info("vrlMoveLiveness 任务已接收 task_id=%s", req.task_id)

    return MoveLivenessAsyncResponse(
        code=0,
        msg="任务已接收，处理完成后将回调通知",
        task_id=req.task_id,
        estimated_time=10,  # 预计 10 秒
    )
