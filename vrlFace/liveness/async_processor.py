"""
活体检测异步任务处理器

负责：
- 接收任务参数
- 调用 VideoLivenessAnalyzer 处理视频
- 处理完成后调用回调客户端
"""

import logging
import traceback
from typing import List, Optional, Dict, Any

from .video_analyzer import VideoLivenessAnalyzer
from .config import LivenessConfig, CallbackConfig
from .callback import send_callback

logger = logging.getLogger(__name__)


async def process_liveness_task(
    task_id: str,
    request_id: str,
    video_path: str,
    actions: List[str],
    callback_url: str,
    liveness_threshold: float = 0.5,
    action_threshold: float = 0.85,
    max_video_duration: Optional[float] = None,
    per_action_timeout: Optional[float] = None,
    callback_secret: Optional[str] = None,
) -> None:
    """
    异步处理活体检测任务

    参数:
        task_id: 任务 ID
        request_id: 请求 ID
        video_path: 视频文件路径
        actions: 动作列表
        callback_url: 回调地址
        liveness_threshold: 活体判定阈值
        action_threshold: 动作通过阈值
        max_video_duration: 最大视频时长
        per_action_timeout: 每个动作超时时间
        callback_secret: 回调签名密钥（可选）
    """
    logger.info(
        "开始处理活体检测任务 task_id=%s request_id=%s video=%s actions=%s",
        task_id,
        request_id,
        video_path,
        actions,
    )

    try:
        # 构建分析器
        config = LivenessConfig.video_fast_config()
        analyzer = VideoLivenessAnalyzer(
            liveness_config=config,
            liveness_threshold=liveness_threshold,
            action_threshold=action_threshold,
        )

        # 执行分析
        result = analyzer.analyze(
            video_path=video_path,
            actions=actions,
            max_video_duration=max_video_duration,
            per_action_timeout=per_action_timeout,
        )

        # 构建回调数据
        callback_data = _build_callback_data(
            task_id=task_id,
            request_id=request_id,
            result=result,
            code=0,
            msg="success",
        )

        logger.info(
            "活体检测完成 task_id=%s is_liveness=%d confidence=%.3f",
            task_id,
            result.is_liveness,
            result.liveness_confidence,
        )

    except Exception as e:
        logger.error(
            "活体检测异常 task_id=%s error=%s\n%s",
            task_id,
            str(e),
            traceback.format_exc(),
        )

        # 构建错误回调数据
        callback_data = {
            "code": 500,
            "msg": f"处理失败: {str(e)}",
            "task_id": task_id,
            "data": None,
        }

    # 发送回调
    try:
        callback_config = CallbackConfig.from_env()

        logger.info(
            "准备发送回调 task_id=%s callback_url=%s data_keys=%s",
            task_id,
            callback_url,
            list(callback_data.keys()) if callback_data else "None",
        )

        success = await send_callback(
            url=callback_url,
            data=callback_data,
            secret=callback_secret,
            config=callback_config,
        )

        if success:
            logger.info("回调发送成功 task_id=%s", task_id)
        else:
            logger.error("回调发送失败 task_id=%s", task_id)

    except Exception as e:
        logger.error(
            "回调发送异常 task_id=%s error=%s\n%s",
            task_id,
            str(e),
            traceback.format_exc(),
        )


def _build_callback_data(
    task_id: str,
    request_id: str,
    result: Any,
    code: int,
    msg: str,
) -> Dict[str, Any]:
    """
    构建回调数据

    参数:
        task_id: 任务 ID
        request_id: 请求 ID
        result: VideoLivenessResult 对象
        code: 状态码
        msg: 状态消息

    返回:
        回调数据字典
    """
    # 构建 face_info
    face_info = None
    if result.face_info:
        face_info = {
            "confidence": result.face_info.confidence,
            "quality_score": result.face_info.quality_score,
        }

    # 构建 action_details
    action_details = [
        {
            "action": detail.action,
            "passed": detail.passed,
            "confidence": detail.confidence,
            "msg": detail.msg,
        }
        for detail in result.action_verify.action_details
    ]

    # 构建完整数据
    data = {
        "is_liveness": result.is_liveness,
        "liveness_confidence": result.liveness_confidence,
        "is_face_exist": result.is_face_exist,
        "face_info": face_info,
        "action_verify": {
            "passed": result.action_verify.passed,
            "required_actions": result.action_verify.required_actions,
            "action_details": action_details,
        },
    }

    return {
        "code": code,
        "msg": msg,
        "task_id": task_id,
        "data": data,
    }
