"""
异步处理器测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from vrlFace.liveness.async_processor import process_liveness_task, _build_callback_data
from vrlFace.liveness.video_analyzer import (
    VideoLivenessResult,
    FaceInfo,
    ActionVerifyResult,
    ActionResult,
)


class TestBuildCallbackData:
    """测试回调数据构建"""

    def test_build_callback_data_success(self):
        """测试成功结果的回调数据构建"""
        # 构建模拟结果
        face_info = FaceInfo(confidence=0.95, quality_score=0.88)
        action_details = [
            ActionResult(
                action="blink", passed=True, confidence=0.92, msg="检测到眨眼"
            ),
        ]
        action_verify = ActionVerifyResult(
            passed=True,
            required_actions=["blink"],
            action_details=action_details,
        )
        result = VideoLivenessResult(
            is_liveness=1,
            liveness_confidence=0.89,
            is_face_exist=1,
            face_info=face_info,
            action_verify=action_verify,
        )

        # 构建回调数据
        callback_data = _build_callback_data(
            task_id="test-task-123",
            request_id="test-req-456",
            result=result,
            code=0,
            msg="success",
        )

        # 验证结构
        assert callback_data["code"] == 0
        assert callback_data["msg"] == "success"
        assert callback_data["task_id"] == "test-task-123"
        assert callback_data["data"] is not None

        # 验证数据内容
        data = callback_data["data"]
        assert data["is_liveness"] == 1
        assert data["liveness_confidence"] == 0.89
        assert data["is_face_exist"] == 1
        assert data["face_info"]["confidence"] == 0.95
        assert data["face_info"]["quality_score"] == 0.88
        assert data["action_verify"]["passed"] is True
        assert len(data["action_verify"]["action_details"]) == 1

    def test_build_callback_data_no_face(self):
        """测试无人脸情况的回调数据"""
        action_verify = ActionVerifyResult(
            passed=False,
            required_actions=["blink"],
            action_details=[],
        )
        result = VideoLivenessResult(
            is_liveness=0,
            liveness_confidence=0.0,
            is_face_exist=0,
            face_info=None,
            action_verify=action_verify,
        )

        callback_data = _build_callback_data(
            task_id="test-task-123",
            request_id="test-req-456",
            result=result,
            code=0,
            msg="success",
        )

        assert callback_data["data"]["face_info"] is None
        assert callback_data["data"]["is_face_exist"] == 0


class TestProcessLivenessTask:
    """测试异步任务处理"""

    @pytest.mark.asyncio
    async def test_process_task_success(self, tmp_path):
        """测试成功处理任务"""
        # 创建临时视频文件
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")

        # Mock VideoLivenessAnalyzer
        mock_result = VideoLivenessResult(
            is_liveness=1,
            liveness_confidence=0.89,
            is_face_exist=1,
            face_info=FaceInfo(confidence=0.95, quality_score=0.88),
            action_verify=ActionVerifyResult(
                passed=True,
                required_actions=["blink"],
                action_details=[
                    ActionResult(
                        action="blink", passed=True, confidence=0.92, msg="检测到眨眼"
                    )
                ],
            ),
        )

        with patch(
            "vrlFace.liveness.async_processor.VideoLivenessAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer

            with patch("vrlFace.liveness.async_processor.send_callback") as mock_send:
                mock_send.return_value = True

                await process_liveness_task(
                    task_id="test-task-123",
                    request_id="test-req-456",
                    video_path=str(video_file),
                    actions=["blink"],
                    callback_url="http://localhost:8092/callback",
                )

                # 验证回调被调用
                assert mock_send.called
                call_args = mock_send.call_args
                callback_data = call_args[1]["data"]
                assert callback_data["code"] == 0
                assert callback_data["task_id"] == "test-task-123"

    @pytest.mark.asyncio
    async def test_process_task_analyzer_error(self, tmp_path):
        """测试分析器异常处理"""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")

        with patch(
            "vrlFace.liveness.async_processor.VideoLivenessAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.side_effect = Exception("分析失败")
            mock_analyzer_class.return_value = mock_analyzer

            with patch("vrlFace.liveness.async_processor.send_callback") as mock_send:
                mock_send.return_value = True

                await process_liveness_task(
                    task_id="test-task-123",
                    request_id="test-req-456",
                    video_path=str(video_file),
                    actions=["blink"],
                    callback_url="http://localhost:8092/callback",
                )

                # 验证错误回调被发送
                assert mock_send.called
                call_args = mock_send.call_args
                callback_data = call_args[1]["data"]
                assert callback_data["code"] == 500
                assert "分析失败" in callback_data["msg"]
                assert callback_data["data"] is None

    @pytest.mark.asyncio
    async def test_process_task_callback_failure(self, tmp_path):
        """测试回调发送失败"""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")

        mock_result = VideoLivenessResult(
            is_liveness=1,
            liveness_confidence=0.89,
            is_face_exist=1,
            face_info=None,
            action_verify=ActionVerifyResult(
                passed=True,
                required_actions=["blink"],
                action_details=[],
            ),
        )

        with patch(
            "vrlFace.liveness.async_processor.VideoLivenessAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer

            with patch("vrlFace.liveness.async_processor.send_callback") as mock_send:
                mock_send.return_value = False  # 回调失败

                # 不应该抛出异常
                await process_liveness_task(
                    task_id="test-task-123",
                    request_id="test-req-456",
                    video_path=str(video_file),
                    actions=["blink"],
                    callback_url="http://localhost:8092/callback",
                )

                assert mock_send.called

    @pytest.mark.asyncio
    async def test_process_task_with_custom_thresholds(self, tmp_path):
        """测试自定义阈值"""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")

        mock_result = VideoLivenessResult(
            is_liveness=1,
            liveness_confidence=0.89,
            is_face_exist=1,
            face_info=None,
            action_verify=ActionVerifyResult(
                passed=True,
                required_actions=["blink"],
                action_details=[],
            ),
        )

        with patch(
            "vrlFace.liveness.async_processor.VideoLivenessAnalyzer"
        ) as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer

            with patch("vrlFace.liveness.async_processor.send_callback") as mock_send:
                mock_send.return_value = True

                await process_liveness_task(
                    task_id="test-task-123",
                    request_id="test-req-456",
                    video_path=str(video_file),
                    actions=["blink"],
                    callback_url="http://localhost:8092/callback",
                    liveness_threshold=0.6,
                    action_threshold=0.9,
                )

                # 验证分析器使用了自定义阈值
                mock_analyzer_class.assert_called_once()
                call_kwargs = mock_analyzer_class.call_args[1]
                assert call_kwargs["liveness_threshold"] == 0.6
                assert call_kwargs["action_threshold"] == 0.9
