"""
回调客户端测试
"""

import pytest
import hmac
import hashlib
import json
from unittest.mock import AsyncMock, patch, MagicMock

from vrlFace.liveness.callback import generate_signature, send_callback
from vrlFace.liveness.config import CallbackConfig


class TestGenerateSignature:
    """测试签名生成"""

    def test_signature_basic(self):
        """测试基本签名生成"""
        body = b'{"test": "data"}'
        secret = "test-secret"

        signature = generate_signature(body, secret)

        # 验证签名格式
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest 长度

        # 验证签名正确性
        expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        assert signature == expected

    def test_signature_empty_body(self):
        """测试空请求体签名"""
        body = b""
        secret = "test-secret"

        signature = generate_signature(body, secret)

        assert isinstance(signature, str)
        assert len(signature) == 64

    def test_signature_chinese_content(self):
        """测试中文内容签名"""
        data = {"msg": "测试中文"}
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        secret = "test-secret"

        signature = generate_signature(body, secret)

        assert isinstance(signature, str)
        assert len(signature) == 64

    def test_signature_consistency(self):
        """测试签名一致性"""
        body = b'{"task_id": "123"}'
        secret = "test-secret"

        sig1 = generate_signature(body, secret)
        sig2 = generate_signature(body, secret)

        assert sig1 == sig2

    def test_signature_different_secrets(self):
        """测试不同密钥产生不同签名"""
        body = b'{"task_id": "123"}'

        sig1 = generate_signature(body, "secret1")
        sig2 = generate_signature(body, "secret2")

        assert sig1 != sig2


class TestSendCallback:
    """测试回调发送"""

    @pytest.mark.asyncio
    async def test_send_callback_success(self):
        """测试成功发送回调"""
        url = "http://localhost:8092/api/v1/callbacks/liveness/action"
        data = {
            "code": 0,
            "msg": "success",
            "task_id": "test-task-123",
            "data": {"is_liveness": 1},
        }

        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"code": 0}'

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await send_callback(url, data)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_callback_failure(self):
        """测试回调失败（非 200 状态码）"""
        url = "http://localhost:8092/api/v1/callbacks/liveness/action"
        data = {"code": 0, "task_id": "test-task-123"}

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        config = CallbackConfig(max_retries=1, retry_delay=0)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await send_callback(url, data, config=config)

            assert result is False

    @pytest.mark.asyncio
    async def test_send_callback_timeout(self):
        """测试回调超时"""
        import httpx

        url = "http://localhost:8092/api/v1/callbacks/liveness/action"
        data = {"code": 0, "task_id": "test-task-123"}

        config = CallbackConfig(max_retries=1, retry_delay=0, timeout=1)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await send_callback(url, data, config=config)

            assert result is False

    @pytest.mark.asyncio
    async def test_send_callback_retry(self):
        """测试回调重试机制"""
        url = "http://localhost:8092/api/v1/callbacks/liveness/action"
        data = {"code": 0, "task_id": "test-task-123"}

        # 第一次失败，第二次成功
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Error"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.text = '{"code": 0}'

        config = CallbackConfig(max_retries=2, retry_delay=0)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=[mock_response_fail, mock_response_success]
            )

            result = await send_callback(url, data, config=config)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_callback_custom_secret(self):
        """测试自定义签名密钥"""
        url = "http://localhost:8092/api/v1/callbacks/liveness/action"
        data = {"code": 0, "task_id": "test-task-123"}
        custom_secret = "custom-secret-key"

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await send_callback(url, data, secret=custom_secret)

            assert result is True

            # 验证签名使用了自定义密钥
            call_args = mock_post.call_args
            headers = call_args.kwargs["headers"]
            body = call_args.kwargs["content"]

            expected_sig = generate_signature(body, custom_secret)
            assert headers["X-ThirdParty-Signature"] == expected_sig

    @pytest.mark.asyncio
    async def test_send_callback_signature_header(self):
        """测试签名 Header 正确设置"""
        url = "http://localhost:8092/api/v1/callbacks/liveness/action"
        data = {"code": 0, "task_id": "test-task-123"}

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await send_callback(url, data)

            # 验证 Header
            call_args = mock_post.call_args
            headers = call_args.kwargs["headers"]

            assert "X-ThirdParty-Signature" in headers
            assert "Content-Type" in headers
            assert headers["Content-Type"] == "application/json"
