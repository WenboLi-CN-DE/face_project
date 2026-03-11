"""
活体检测 FastAPI 路由

接口:
    POST /vrlMoveLiveness       — 视频活体检测 + 动作验证（同步）
    POST /vrlMoveLiveness/async — 视频活体检测 + 动作验证（异步回调）
"""

import logging
import os
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks

from .schemas import (
    MoveLivenessRequest,
    MoveLivenessResponse,
    MoveLivenessData,
    MoveLivenessAsyncRequest,
    MoveLivenessAsyncResponse,
    FaceInfoResponse,
    ActionDetailResponse,
    ActionVerifyResponse,
)
from .video_analyzer import VideoLivenessAnalyzer
from .config import LivenessConfig
from .async_processor import process_liveness_task

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_analyzer(req: MoveLivenessRequest) -> VideoLivenessAnalyzer:
    """根据请求参数构建分析器实例"""
    config = LivenessConfig.video_fast_config()
    return VideoLivenessAnalyzer(
        liveness_config=config,
        liveness_threshold=req.threshold_config.liveness_threshold,
        action_threshold=req.threshold_config.action_threshold,
    )


@router.post("/vrlMoveLiveness", response_model=MoveLivenessResponse)
async def vrl_move_liveness(req: MoveLivenessRequest) -> MoveLivenessResponse:
    """
    视频活体检测 + 动作验证接口

    - 接收视频路径和动作列表
    - 按动作时间段逐段分析
    - 返回全局活体判定和每动作检测结果
    """
    logger.info(
        "vrlMoveLiveness request_id=%s task_id=%s video=%s actions=%s",
        req.request_id,
        req.task_id,
        req.video_path,
        req.actions,
    )

    # 路径安全校验
    if not os.path.exists(req.video_path):
        raise HTTPException(
            status_code=400,
            detail=f"视频文件不存在：{req.video_path}",
        )

    # 支持的动作白名单
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

    try:
        analyzer = _build_analyzer(req)

        max_duration = None
        per_action_timeout = None
        if req.action_config:
            max_duration = req.action_config.max_video_duration
            per_action_timeout = req.action_config.per_action_timeout

        result = analyzer.analyze(
            video_path=req.video_path,
            actions=req.actions,
            max_video_duration=max_duration,
            per_action_timeout=per_action_timeout,
        )

        # 组装响应
        face_info_resp = None
        if result.face_info:
            face_info_resp = FaceInfoResponse(
                confidence=result.face_info.confidence,
                quality_score=result.face_info.quality_score,
            )

        action_details_resp = [
            ActionDetailResponse(
                action=d.action,
                passed=d.passed,
                confidence=d.confidence,
                msg=d.msg,
            )
            for d in result.action_verify.action_details
        ]

        action_verify_resp = ActionVerifyResponse(
            passed=result.action_verify.passed,
            required_actions=result.action_verify.required_actions,
            action_details=action_details_resp,
        )

        data = MoveLivenessData(
            is_liveness=result.is_liveness,
            liveness_confidence=result.liveness_confidence,
            is_face_exist=result.is_face_exist,
            face_info=face_info_resp,
            action_verify=action_verify_resp,
        )

        logger.info(
            "vrlMoveLiveness done request_id=%s is_liveness=%d liveness_conf=%.3f action_passed=%s",
            req.request_id,
            result.is_liveness,
            result.liveness_confidence,
            result.action_verify.passed,
        )

        return MoveLivenessResponse(
            code=0,
            msg="success",
            data=data,
            request_id=req.request_id,
            task_id=req.task_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("vrlMoveLiveness error: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"视频分析出错：{str(e)}")


@router.post("/vrlMoveLiveness/async", response_model=MoveLivenessAsyncResponse)
async def vrl_move_liveness_async(
    req: MoveLivenessAsyncRequest, background_tasks: BackgroundTasks
) -> MoveLivenessAsyncResponse:
    """
    视频活体检测 + 动作验证接口（异步回调）

    - 接收视频路径和动作列表
    - 立即返回 task_id
    - 后台异步处理，完成后回调通知
    """
    logger.info(
        "vrlMoveLiveness/async request_id=%s task_id=%s video=%s actions=%s callback=%s",
        req.request_id,
        req.task_id,
        req.video_path,
        req.actions,
        req.callback_url,
    )

    # 路径安全校验
    if not os.path.exists(req.video_path):
        raise HTTPException(
            status_code=400,
            detail=f"视频文件不存在：{req.video_path}",
        )

    # 支持的动作白名单
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

    logger.info("vrlMoveLiveness/async 任务已接收 task_id=%s", req.task_id)

    return MoveLivenessAsyncResponse(
        code=0,
        msg="任务已接收，处理完成后将回调通知",
        task_id=req.task_id,
        estimated_time=10,  # 预计 10 秒
    )
