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
from .config import CallbackConfig
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

    # 获取回调配置
    callback_config = CallbackConfig.from_env()
    callback_url = req.callback_url or callback_config.default_url

    # 提取配置参数
    max_duration = None
    per_action_timeout = None
    if req.action_config:
        max_duration = req.action_config.max_video_duration
        per_action_timeout = req.action_config.per_action_timeout

    # 阈值安全调整（自动裁剪到安全范围）
    requested_liveness_threshold = req.threshold_config.liveness_threshold
    requested_action_threshold = req.threshold_config.action_threshold

    # 安全范围定义
    LIVENESS_MIN, LIVENESS_MAX = 0.30, 0.75
    ACTION_MIN, ACTION_MAX = 0.50, 0.95

    # 推荐范围（用于日志警告）
    LIVENESS_RECOMMENDED_MIN, LIVENESS_RECOMMENDED_MAX = 0.45, 0.60
    ACTION_RECOMMENDED_MIN, ACTION_RECOMMENDED_MAX = 0.70, 0.85

    # 自动调整 liveness_threshold 到安全范围
    adjusted_liveness = requested_liveness_threshold
    liveness_adjusted = False
    if requested_liveness_threshold < LIVENESS_MIN:
        adjusted_liveness = LIVENESS_MIN
        liveness_adjusted = True
        logger.warning(
            "vrlMoveLiveness liveness_threshold=%.2f 低于安全下限，自动调整到 %.2f task_id=%s",
            requested_liveness_threshold,
            LIVENESS_MIN,
            req.task_id,
        )
    elif requested_liveness_threshold > LIVENESS_MAX:
        adjusted_liveness = LIVENESS_MAX
        liveness_adjusted = True
        logger.warning(
            "vrlMoveLiveness liveness_threshold=%.2f 高于安全上限，自动调整到 %.2f task_id=%s",
            requested_liveness_threshold,
            LIVENESS_MAX,
            req.task_id,
        )

    # 自动调整 action_threshold 到安全范围
    adjusted_action = requested_action_threshold
    action_adjusted = False
    if requested_action_threshold < ACTION_MIN:
        adjusted_action = ACTION_MIN
        action_adjusted = True
        logger.warning(
            "vrlMoveLiveness action_threshold=%.2f 低于安全下限，自动调整到 %.2f task_id=%s",
            requested_action_threshold,
            ACTION_MIN,
            req.task_id,
        )
    elif requested_action_threshold > ACTION_MAX:
        adjusted_action = ACTION_MAX
        action_adjusted = True
        logger.warning(
            "vrlMoveLiveness action_threshold=%.2f 高于安全上限，自动调整到 %.2f task_id=%s",
            requested_action_threshold,
            ACTION_MAX,
            req.task_id,
        )

    # 记录最终使用的阈值
    logger.info(
        "vrlMoveLiveness 阈值配置 task_id=%s 请求值 [liveness=%.2f, action=%.2f] 使用值 [liveness=%.2f, action=%.2f]",
        req.task_id,
        requested_liveness_threshold,
        requested_action_threshold,
        adjusted_liveness,
        adjusted_action,
    )

    # 如果超出推荐范围但未超出安全范围，记录提示
    if not liveness_adjusted and (
        requested_liveness_threshold < LIVENESS_RECOMMENDED_MIN
        or requested_liveness_threshold > LIVENESS_RECOMMENDED_MAX
    ):
        logger.info(
            "vrlMoveLiveness liveness_threshold=%.2f 超出推荐范围 [%.2f-%.2f] task_id=%s",
            requested_liveness_threshold,
            LIVENESS_RECOMMENDED_MIN,
            LIVENESS_RECOMMENDED_MAX,
            req.task_id,
        )

    if not action_adjusted and (
        requested_action_threshold < ACTION_RECOMMENDED_MIN
        or requested_action_threshold > ACTION_RECOMMENDED_MAX
    ):
        logger.info(
            "vrlMoveLiveness action_threshold=%.2f 超出推荐范围 [%.2f-%.2f] task_id=%s",
            requested_action_threshold,
            ACTION_RECOMMENDED_MIN,
            ACTION_RECOMMENDED_MAX,
            req.task_id,
        )

    # 添加后台任务（使用调整后的阈值）
    background_tasks.add_task(
        process_liveness_task,
        task_id=req.task_id,
        request_id=req.request_id,
        video_path=req.video_path,
        actions=req.actions,
        callback_url=callback_url,
        liveness_threshold=adjusted_liveness,
        action_threshold=adjusted_action,
        max_video_duration=max_duration,
        per_action_timeout=per_action_timeout,
        callback_secret=req.callback_secret,
    )

    logger.info(
        "vrlMoveLiveness 任务已接收 task_id=%s callback_url=%s",
        req.task_id,
        callback_url,
    )

    return MoveLivenessAsyncResponse(
        code=0,
        msg="任务已接收，处理完成后将回调通知",
        task_id=req.task_id,
        estimated_time=10,  # 预计 10 秒
    )
