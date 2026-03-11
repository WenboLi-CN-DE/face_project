"""
活体检测回调客户端

负责：
- 生成 HMAC-SHA256 签名
- 发送 HTTP POST 请求到前端回调地址
- 处理重试逻辑
"""

import hmac
import hashlib
import json
import logging
from typing import Dict, Any, Optional

import httpx

from .config import CallbackConfig

logger = logging.getLogger(__name__)


def generate_signature(body: bytes, secret: str) -> str:
    """
    生成 HMAC-SHA256 签名

    参数:
        body: 请求体字节流
        secret: 签名密钥

    返回:
        十六进制签名字符串
    """
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


async def send_callback(
    url: str,
    data: Dict[str, Any],
    secret: Optional[str] = None,
    config: Optional[CallbackConfig] = None,
) -> bool:
    """
    发送回调请求到前端

    参数:
        url: 回调地址
        data: 回调数据（将被序列化为 JSON）
        secret: 签名密钥（优先使用此参数，否则使用 config）
        config: 回调配置（包含密钥、超时、重试等）

    返回:
        是否成功（True=成功，False=失败）
    """
    if config is None:
        config = CallbackConfig()

    if secret is None:
        secret = config.secret_key

    # 序列化数据
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    signature = generate_signature(body, secret)

    headers = {
        "Content-Type": "application/json",
        "X-ThirdParty-Signature": signature,
    }

    # 重试逻辑
    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info(
                "发送回调 attempt=%d/%d url=%s task_id=%s",
                attempt,
                config.max_retries,
                url,
                data.get("task_id", "unknown"),
            )

            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(url, content=body, headers=headers)

            if response.status_code == 200:
                logger.info(
                    "回调成功 url=%s task_id=%s status=%d",
                    url,
                    data.get("task_id", "unknown"),
                    response.status_code,
                )
                return True
            else:
                logger.warning(
                    "回调失败 url=%s task_id=%s status=%d response=%s",
                    url,
                    data.get("task_id", "unknown"),
                    response.status_code,
                    response.text[:200],
                )

        except httpx.TimeoutException:
            logger.warning(
                "回调超时 attempt=%d/%d url=%s task_id=%s",
                attempt,
                config.max_retries,
                url,
                data.get("task_id", "unknown"),
            )
        except Exception as e:
            logger.error(
                "回调异常 attempt=%d/%d url=%s task_id=%s error=%s",
                attempt,
                config.max_retries,
                url,
                data.get("task_id", "unknown"),
                str(e),
            )

        # 如果不是最后一次尝试，等待后重试
        if attempt < config.max_retries:
            import asyncio

            await asyncio.sleep(config.retry_delay)

    logger.error(
        "回调最终失败 url=%s task_id=%s max_retries=%d",
        url,
        data.get("task_id", "unknown"),
        config.max_retries,
    )
    return False
