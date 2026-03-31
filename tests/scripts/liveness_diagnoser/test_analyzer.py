#!/usr/bin/env python3
"""测试视频分析模块 - analyzer.py"""

import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.liveness_diagnoser.analyzer import VideoAnalyzer
from scripts.liveness_diagnoser.models import (
    FrameAnalysis,
    ActionDetail,
    DiagnosisResult,
    VideoInfo,
)
from vrlFace.liveness.config import LivenessConfig


class TestVideoAnalyzerInit:
    """测试 VideoAnalyzer 初始化"""

    @patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine")
    @patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector")
    @patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector")
    def test_init_default_config(self, mock_head, mock_fast, mock_engine):
        """测试使用默认配置初始化"""
        analyzer = VideoAnalyzer()

        assert analyzer.config is not None
        assert analyzer.config == LivenessConfig.video_fast_config()
        assert hasattr(analyzer, "engine")
        assert hasattr(analyzer, "fast_detector")
        assert hasattr(analyzer, "head_detector")

    @patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine")
    @patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector")
    @patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector")
    def test_init_custom_config(
        self, mock_head, mock_fast, mock_engine, custom_config=None
    ):
        """测试使用自定义配置初始化"""
        if custom_config is None:
            custom_config = LivenessConfig(
                ear_threshold=0.20,
                mar_threshold=0.55,
                yaw_threshold=10.0,
                pitch_threshold=10.0,
            )
        analyzer = VideoAnalyzer(config=custom_config)

        assert analyzer.config == custom_config
        assert analyzer.config.ear_threshold == 0.20
        assert analyzer.config.mar_threshold == 0.55


class TestCalculateActionDetails:
    """测试 _calculate_action_details 方法"""

    def setup_method(self):
        """每个测试前设置"""
        with patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine"):
            with patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector"):
                with patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector"):
                    self.analyzer = VideoAnalyzer()
        self.analyzer.config.ear_threshold = 0.20
        self.analyzer.config.mar_threshold = 0.55
        self.analyzer.config.pitch_threshold = 8.0
        self.analyzer.config.yaw_threshold = 8.0

    def test_calculate_blink_detail(self):
        """测试眨眼动作详情计算"""
        frame_analyses = [
            FrameAnalysis(
                frame_idx=i,
                has_face=True,
                ear=0.15 if i % 2 == 0 else 0.25,  # 交替眨眼
                mar=0.50,
                pitch=0.0,
                yaw=0.0,
                is_blink=i % 2 == 0,
                is_mouth_open=False,
            )
            for i in range(10)
        ]

        details = self.analyzer._calculate_action_details(frame_analyses, ["blink"])

        assert len(details) == 1
        blink_detail = details[0]
        assert blink_detail.name == "blink"
        assert blink_detail.name_cn == "眨眼"
        assert len(blink_detail.frames) == 5  # 5 帧眨眼
        assert blink_detail.events == 5
        assert blink_detail.avg_score == 0.5  # 50% 眨眼率
        assert blink_detail.passed is True  # >= 30%
        # 50% >= 50% -> high
        assert blink_detail.confidence == "high"

    def test_calculate_mouth_open_detail(self):
        """测试张嘴动作详情计算"""
        frame_analyses = [
            FrameAnalysis(
                frame_idx=i,
                has_face=True,
                ear=0.20,
                mar=0.60 if i % 3 == 0 else 0.40,  # 每 3 帧张嘴
                pitch=0.0,
                yaw=0.0,
                is_blink=False,
                is_mouth_open=i % 3 == 0,
            )
            for i in range(9)
        ]

        details = self.analyzer._calculate_action_details(
            frame_analyses, ["mouth_open"]
        )

        assert len(details) == 1
        mouth_detail = details[0]
        assert mouth_detail.name == "mouth_open"
        assert mouth_detail.name_cn == "张嘴"
        assert len(mouth_detail.frames) == 3  # 3 帧张嘴
        assert mouth_detail.events == 3
        assert mouth_detail.avg_score == pytest.approx(0.333, rel=0.01)
        assert mouth_detail.passed is True  # >= 30%

    def test_calculate_nod_detail(self):
        """测试点头动作详情计算"""
        frame_analyses = [
            FrameAnalysis(
                frame_idx=i,
                has_face=True,
                ear=0.20,
                mar=0.50,
                pitch=10.0 if i < 5 else -5.0,  # Pitch 峰峰值=15°
                yaw=0.0,
                is_blink=False,
                is_mouth_open=False,
                head_action="nod" if i < 5 else None,
            )
            for i in range(10)
        ]

        details = self.analyzer._calculate_action_details(frame_analyses, ["nod"])

        assert len(details) == 1
        nod_detail = details[0]
        assert nod_detail.name == "nod"
        assert nod_detail.name_cn == "点头"
        # Pitch 峰峰值 = 10.0 - (-5.0) = 15.0°
        assert nod_detail.avg_score == pytest.approx(
            15.0 / 8.0, rel=0.01
        )  # 15.0 / threshold(8.0)
        assert nod_detail.passed is True  # 15.0° >= 8.0°
        assert nod_detail.confidence == "high"  # >= 1.5 * threshold

    def test_calculate_shake_head_detail(self):
        """测试摇头动作详情计算"""
        frame_analyses = [
            FrameAnalysis(
                frame_idx=i,
                has_face=True,
                ear=0.20,
                mar=0.50,
                pitch=0.0,
                yaw=12.0 if i < 5 else -8.0,  # Yaw 峰峰值=20°
                is_blink=False,
                is_mouth_open=False,
                head_action="head_turn_left" if i < 5 else "head_turn_right",
            )
            for i in range(10)
        ]

        details = self.analyzer._calculate_action_details(
            frame_analyses, ["shake_head"]
        )

        assert len(details) == 1
        shake_detail = details[0]
        assert shake_detail.name == "shake_head"
        assert shake_detail.name_cn == "摇头"
        # Yaw 峰峰值 = 12.0 - (-8.0) = 20.0°
        assert shake_detail.avg_score == pytest.approx(
            20.0 / 8.0, rel=0.01
        )  # 20.0 / threshold(8.0)
        assert shake_detail.passed is True  # 20.0° >= 8.0°
        assert shake_detail.confidence == "high"

    def test_calculate_multiple_actions(self):
        """测试多个动作同时计算"""
        frame_analyses = [
            FrameAnalysis(
                frame_idx=i,
                has_face=True,
                ear=0.15 if i % 2 == 0 else 0.25,
                mar=0.60 if i % 3 == 0 else 0.40,
                pitch=0.0,
                yaw=0.0,
                is_blink=i % 2 == 0,
                is_mouth_open=i % 3 == 0,
            )
            for i in range(12)
        ]

        details = self.analyzer._calculate_action_details(
            frame_analyses, ["blink", "mouth_open"]
        )

        assert len(details) == 2
        assert details[0].name == "blink"
        assert details[1].name == "mouth_open"


class TestGenerateSuggestions:
    """测试 _generate_suggestions 方法"""

    def setup_method(self):
        """每个测试前设置"""
        with patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine"):
            with patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector"):
                with patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector"):
                    self.analyzer = VideoAnalyzer()

    def test_generate_suggestions_low_face_detection_rate(self):
        """测试人脸检出率低的建议"""
        result = DiagnosisResult(
            video_info=Mock(spec=VideoInfo),
            total_frames=100,
            frames_with_face=50,
            face_detection_rate=0.5,
            action_details=[],
            frame_analyses=[],
            avg_quality_score=0.7,
            overall_passed=True,
            overall_message="测试通过",
        )

        suggestions = self.analyzer._generate_suggestions(result)

        assert any("人脸检出率偏低" in s for s in suggestions)

    def test_generate_suggestions_low_quality(self):
        """测试质量评分低的建议"""
        result = DiagnosisResult(
            video_info=Mock(spec=VideoInfo),
            total_frames=100,
            frames_with_face=90,
            face_detection_rate=0.9,
            action_details=[],
            frame_analyses=[],
            avg_quality_score=0.3,
            overall_passed=True,
            overall_message="测试通过",
        )

        suggestions = self.analyzer._generate_suggestions(result)

        assert any("视频质量偏低" in s for s in suggestions)

    def test_generate_suggestions_blink_failed(self):
        """测试眨眼检测未通过的建议"""
        frame_analyses = [
            FrameAnalysis(frame_idx=i, has_face=True, ear=0.25, mar=0.50)
            for i in range(10)
        ]
        action_detail = ActionDetail(
            name="blink",
            name_cn="眨眼",
            frames=[],
            events=0,
            avg_score=0.0,
            confidence="low",
            passed=False,
            message="眨眼帧数：0/10 (0.0%)",
        )

        result = DiagnosisResult(
            video_info=Mock(spec=VideoInfo),
            total_frames=10,
            frames_with_face=10,
            face_detection_rate=1.0,
            action_details=[action_detail],
            frame_analyses=frame_analyses,
            avg_quality_score=0.8,
            overall_passed=False,
            overall_message="眨眼未通过",
        )

        suggestions = self.analyzer._generate_suggestions(result)

        assert any("眨眼检测未通过" in s for s in suggestions)
        assert any("ear_threshold" in s for s in suggestions)

    def test_generate_suggestions_all_passed(self):
        """测试全部通过的建议"""
        result = DiagnosisResult(
            video_info=Mock(spec=VideoInfo),
            total_frames=100,
            frames_with_face=95,
            face_detection_rate=0.95,
            action_details=[],
            frame_analyses=[],
            avg_quality_score=0.85,
            overall_passed=True,
            overall_message="所有动作通过",
        )

        suggestions = self.analyzer._generate_suggestions(result)

        assert any("总体判定：通过" in s for s in suggestions)


class TestAnalyze:
    """测试 analyze 主分析方法"""

    @patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine")
    @patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector")
    @patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector")
    def test_analyze_basic_flow(self, mock_head, mock_fast, mock_engine):
        """测试 analyze 基本流程（使用 mock）"""
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True

        # 使用具体值而不是 cv2 常量
        def mock_get(prop_id):
            prop_map = {
                7: 100,  # CAP_PROP_FRAME_COUNT
                5: 30.0,  # CAP_PROP_FPS
                3: 1280,  # CAP_PROP_FRAME_WIDTH
                4: 720,  # CAP_PROP_FRAME_HEIGHT
            }
            return prop_map.get(prop_id, 0)

        mock_cap.get.side_effect = mock_get

        # Mock 帧读取 - 提供足够的帧数（analyzer 限制最多 500 帧）
        frames_to_read = [
            (True, np.zeros((720, 1280, 3), dtype=np.uint8)) for _ in range(100)
        ] + [(False, None)]  # 最后一帧结束
        mock_cap.read.side_effect = frames_to_read

        analyzer = VideoAnalyzer()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            # Mock _analyze_frame 返回简单结果
            with patch.object(
                analyzer,
                "_analyze_frame",
                return_value=FrameAnalysis(
                    frame_idx=0,
                    has_face=True,
                    ear=0.20,
                    mar=0.50,
                    pitch=0.0,
                    yaw=0.0,
                    motion_score=0.5,
                    smoothed_score=0.5,
                    quality_score=0.7,
                    is_blink=False,
                    is_mouth_open=False,
                ),
            ):
                result = analyzer.analyze("test_video.mp4", ["blink", "mouth_open"])

                # 验证结果结构
                assert isinstance(result, DiagnosisResult)
                assert result.video_info is not None
                assert result.video_info.filename == "test_video.mp4"
                assert result.total_frames == 100
                assert result.frames_with_face == 100
                assert result.face_detection_rate == 1.0
                assert len(result.action_details) == 2
                assert isinstance(result.suggestions, list)

    @patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine")
    @patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector")
    @patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector")
    def test_analyze_video_not_found(self, mock_head, mock_fast, mock_engine):
        """测试视频文件不存在的情况"""
        analyzer = VideoAnalyzer()

        with patch("cv2.VideoCapture") as mock_capture:
            mock_capture.return_value.isOpened.return_value = False

            with pytest.raises(ValueError, match="无法打开视频"):
                analyzer.analyze("nonexistent_video.mp4")

    @patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine")
    @patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector")
    @patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector")
    def test_analyze_resource_cleanup(self, mock_head, mock_fast, mock_engine):
        """测试资源释放（cap.release）"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(x, 0)
        mock_cap.read.return_value = (False, None)  # 立即结束读取

        analyzer = VideoAnalyzer()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            try:
                analyzer.analyze("test_video.mp4")
            except Exception:
                pass  # 忽略可能的异常

            # 验证 cap.release() 被调用
            mock_cap.release.assert_called_once()


class TestAnalyzeFrame:
    """测试 _analyze_frame 方法"""

    def setup_method(self):
        """每个测试前设置"""
        with patch("scripts.liveness_diagnoser.analyzer.LivenessFusionEngine"):
            with patch("scripts.liveness_diagnoser.analyzer.FastLivenessDetector"):
                with patch("scripts.liveness_diagnoser.analyzer.HeadActionDetector"):
                    self.analyzer = VideoAnalyzer()

    def test_analyze_frame_no_face(self):
        """测试无脸检测的情况"""
        # Mock extract_landmarks 返回 None
        with patch.object(
            self.analyzer.engine.mp_detector, "extract_landmarks", return_value=None
        ):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = self.analyzer._analyze_frame(frame, 0)

            assert result is not None
            assert result.has_face is False
            assert result.ear is None
            assert result.mar is None

    def test_analyze_frame_with_face(self):
        """测试有脸检测的情况"""
        # Mock extract_landmarks 返回有效数据
        mock_landmarks = np.random.rand(478, 3)
        with patch.object(
            self.analyzer.engine.mp_detector,
            "extract_landmarks",
            return_value={
                "landmarks": mock_landmarks,
                "transform_matrix": None,
                "aspect_ratio": 1.0,
            },
        ):
            # Mock 计算方法
            with patch.object(
                self.analyzer.engine.mp_detector, "calculate_ear", return_value=0.18
            ):
                with patch.object(
                    self.analyzer.engine.mp_detector, "calculate_mar", return_value=0.55
                ):
                    with patch.object(
                        self.analyzer.engine.mp_detector,
                        "calculate_head_pose",
                        return_value=(0.0, 0.0, 0.0),
                    ):
                        with patch.object(
                            self.analyzer.head_detector, "detect", return_value="none"
                        ):
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            result = self.analyzer._analyze_frame(frame, 0)

                            assert result is not None
                            assert result.has_face is True
                            assert result.ear == 0.18
                            assert result.mar == 0.55
                            assert result.pitch == 0.0
                            assert result.yaw == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
