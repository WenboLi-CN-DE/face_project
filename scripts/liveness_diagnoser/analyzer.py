#!/usr/bin/env python3
"""
视频分析模块 - 深度分析活体检测失效原因

功能:
- 逐帧分析视频，提取 landmarks、EAR/MAR、头部姿态
- 计算运动分数和平滑分数
- 统计人脸检出率
- 计算各动作的检测详情
- 生成诊断建议
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.fusion_engine import LivenessFusionEngine
from vrlFace.liveness.head_action import HeadActionDetector, HeadActionConfig
from vrlFace.liveness.utils import build_fast_detector_config
from vrlFace.liveness.fast_detector import FastLivenessDetector

from .models import (
    VideoInfo,
    FrameAnalysis,
    ActionDetail,
    DiagnosisResult,
)


class VideoAnalyzer:
    """视频分析器 - 深度分析活体检测失效原因"""

    def __init__(self, config: Optional[LivenessConfig] = None):
        """
        初始化视频分析器

        Args:
            config: 活体检测配置，默认使用 video_fast_config
        """
        self.config = config or LivenessConfig.video_fast_config()
        self._init_detectors()

    def _init_detectors(self) -> None:
        """初始化检测器"""
        # 主融合引擎
        self.engine = LivenessFusionEngine(self.config)

        # 快速检测器（用于对比）
        fast_config = build_fast_detector_config(self.config)
        self.fast_detector = FastLivenessDetector(**fast_config)

        # 头部动作检测器
        self.head_detector = HeadActionDetector(
            HeadActionConfig(
                yaw_threshold=self.config.yaw_threshold,
                pitch_threshold=self.config.pitch_threshold,
                window_size=self.config.window_size,
                confirm_frames=self.config.action_confirm_frames,
            )
        )

    def _analyze_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Optional[FrameAnalysis]:
        """
        分析单帧

        Args:
            frame: 视频帧
            frame_idx: 帧索引

        Returns:
            FrameAnalysis 或 None（如果处理失败）
        """
        # 缩小帧以加速处理
        max_w = self.config.max_width
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (max_w, int(frame.shape[0] * scale)))

        # 使用融合引擎处理
        result = self.engine.process_frame(frame)

        # 提取 landmarks 数据
        lm_data = self.engine.mp_detector.extract_landmarks(frame)
        if lm_data is None:
            # 未检测到人脸
            return FrameAnalysis(
                frame_idx=frame_idx,
                has_face=False,
            )

        landmarks = lm_data["landmarks"]
        transform_matrix = lm_data.get("transform_matrix")
        aspect_ratio = lm_data.get("aspect_ratio", 1.0)

        # 计算 EAR, MAR
        ear = self.engine.mp_detector.calculate_ear(landmarks, aspect_ratio)
        mar = self.engine.mp_detector.calculate_mar(landmarks, aspect_ratio)

        # 计算头部姿态
        pitch, yaw, roll = self.engine.mp_detector.calculate_head_pose(
            landmarks, frame.shape, transform_matrix
        )

        # 检测头部动作
        head_action = self.head_detector.detect(pitch, yaw)

        # 判断眨眼/张嘴
        is_blink = ear < self.config.ear_threshold
        is_mouth_open = mar > self.config.mar_threshold

        # 计算运动分数（简化版本）
        motion_score = result.motion_score

        # 平滑分数
        self.engine.score_history.append(result.score)
        smoothed_score = (
            np.mean(list(self.engine.score_history))
            if self.engine.score_history
            else result.score
        )

        return FrameAnalysis(
            frame_idx=frame_idx,
            has_face=True,
            ear=ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            motion_score=motion_score,
            smoothed_score=smoothed_score,
            quality_score=result.quality_score,
            is_blink=is_blink,
            is_mouth_open=is_mouth_open,
            head_action=head_action if head_action != "none" else None,
        )

    def _calculate_action_details(
        self, frame_analyses: List[FrameAnalysis], expected_actions: List[str]
    ) -> List[ActionDetail]:
        """
        计算动作检测详情

        Args:
            frame_analyses: 逐帧分析结果
            expected_actions: 期望的动作列表

        Returns:
            动作详情列表
        """
        action_details = []

        # 统计数据
        total_frames = len(frame_analyses)
        frames_with_face = [f for f in frame_analyses if f.has_face]

        for action in expected_actions:
            detail = ActionDetail(name=action, name_cn=self._get_action_cn(action))

            if action == "blink":
                # 眨眼检测
                blink_frames = [f.frame_idx for f in frames_with_face if f.is_blink]
                detail.frames = blink_frames
                detail.events = len(blink_frames)

                if frames_with_face:
                    blink_rate = len(blink_frames) / len(frames_with_face)
                    detail.avg_score = blink_rate
                    detail.passed = blink_rate >= 0.3  # 至少 30% 帧眨眼
                    detail.confidence = (
                        "high"
                        if blink_rate >= 0.5
                        else "medium"
                        if blink_rate >= 0.3
                        else "low"
                    )
                    detail.message = f"眨眼帧数：{len(blink_frames)}/{len(frames_with_face)} ({blink_rate * 100:.1f}%)"

            elif action == "mouth_open":
                # 张嘴检测
                mouth_frames = [
                    f.frame_idx for f in frames_with_face if f.is_mouth_open
                ]
                detail.frames = mouth_frames
                detail.events = len(mouth_frames)

                if frames_with_face:
                    mouth_rate = len(mouth_frames) / len(frames_with_face)
                    detail.avg_score = mouth_rate
                    detail.passed = mouth_rate >= 0.3
                    detail.confidence = (
                        "high"
                        if mouth_rate >= 0.5
                        else "medium"
                        if mouth_rate >= 0.3
                        else "low"
                    )
                    detail.message = f"张嘴帧数：{len(mouth_frames)}/{len(frames_with_face)} ({mouth_rate * 100:.1f}%)"

            elif action in ("nod", "nod_up", "nod_down"):
                # 点头检测
                nod_frames = [
                    f.frame_idx
                    for f in frames_with_face
                    if f.head_action and "nod" in f.head_action
                ]
                detail.frames = nod_frames
                detail.events = len(nod_frames)

                if frames_with_face:
                    # 计算 pitch 峰峰值
                    pitch_values = [
                        f.pitch for f in frames_with_face if f.pitch is not None
                    ]
                    if pitch_values:
                        pitch_range = max(pitch_values) - min(pitch_values)
                        detail.avg_score = pitch_range / self.config.pitch_threshold
                        detail.passed = pitch_range >= self.config.pitch_threshold
                        detail.confidence = (
                            "high"
                            if pitch_range >= self.config.pitch_threshold * 1.5
                            else "medium"
                            if pitch_range >= self.config.pitch_threshold
                            else "low"
                        )
                        detail.message = f"Pitch 峰峰值：{pitch_range:.1f}° (阈值：{self.config.pitch_threshold}°)"

            elif action in ("shake_head", "head_turn"):
                # 摇头/转头检测
                shake_frames = [
                    f.frame_idx
                    for f in frames_with_face
                    if f.head_action
                    and ("turn" in f.head_action or "shake" in f.head_action)
                ]
                detail.frames = shake_frames
                detail.events = len(shake_frames)

                if frames_with_face:
                    # 计算 yaw 峰峰值
                    yaw_values = [f.yaw for f in frames_with_face if f.yaw is not None]
                    if yaw_values:
                        yaw_range = max(yaw_values) - min(yaw_values)
                        detail.avg_score = yaw_range / self.config.yaw_threshold
                        detail.passed = yaw_range >= self.config.yaw_threshold
                        detail.confidence = (
                            "high"
                            if yaw_range >= self.config.yaw_threshold * 1.5
                            else "medium"
                            if yaw_range >= self.config.yaw_threshold
                            else "low"
                        )
                        detail.message = f"Yaw 峰峰值：{yaw_range:.1f}° (阈值：{self.config.yaw_threshold}°)"

            action_details.append(detail)

        return action_details

    def _get_action_cn(self, action: str) -> str:
        """获取动作中文名"""
        mapping = {
            "blink": "眨眼",
            "mouth_open": "张嘴",
            "nod": "点头",
            "nod_up": "抬头",
            "nod_down": "低头",
            "shake_head": "摇头",
            "head_turn": "转头",
            "turn_left": "左转",
            "turn_right": "右转",
        }
        return mapping.get(action, action)

    def _generate_suggestions(self, result: DiagnosisResult) -> List[str]:
        """
        生成诊断建议

        Args:
            result: 诊断结果

        Returns:
            建议列表
        """
        suggestions = []

        # 人脸检出率建议
        if result.face_detection_rate < 0.8:
            suggestions.append(
                f"⚠️  人脸检出率偏低 ({result.face_detection_rate * 100:.1f}%)，建议检查视频质量或光线条件"
            )

        # 质量评分建议
        if result.avg_quality_score < 0.5:
            suggestions.append(
                f"⚠️  视频质量偏低 (平均分：{result.avg_quality_score:.2f})，建议使用更高清的视频"
            )

        # 动作检测建议
        for action_detail in result.action_details:
            if not action_detail.passed:
                if action_detail.name == "blink":
                    suggestions.append(
                        f"⚠️  眨眼检测未通过：{action_detail.message}，建议降低 ear_threshold 到 {min([f.ear for f in result.frame_analyses if f.ear] or [0.2]) * 1.1:.3f}"
                    )
                elif action_detail.name == "mouth_open":
                    suggestions.append(
                        f"⚠️  张嘴检测未通过：{action_detail.message}，建议降低 mar_threshold 到 {max([f.mar for f in result.frame_analyses if f.mar] or [0.5]) * 0.9:.3f}"
                    )
                elif action_detail.name in ("nod", "nod_up", "nod_down"):
                    # 提取 pitch 范围
                    pitch_values = [
                        f.pitch for f in result.frame_analyses if f.pitch is not None
                    ]
                    if pitch_values:
                        pitch_range = max(pitch_values) - min(pitch_values)
                        suggestions.append(
                            f"⚠️  点头检测未通过：{action_detail.message}，建议降低 pitch_threshold 到 {pitch_range * 0.8:.1f}°"
                        )
                elif action_detail.name in ("shake_head", "head_turn"):
                    yaw_values = [
                        f.yaw for f in result.frame_analyses if f.yaw is not None
                    ]
                    if yaw_values:
                        yaw_range = max(yaw_values) - min(yaw_values)
                        suggestions.append(
                            f"⚠️  转头检测未通过：{action_detail.message}，建议降低 yaw_threshold 到 {yaw_range * 0.8:.1f}°"
                        )

        # 总体判定建议
        if not result.overall_passed:
            suggestions.append(f"\n📋  总体判定：未通过。{result.overall_message}")
        else:
            suggestions.append(f"\n✅  总体判定：通过。{result.overall_message}")

        return suggestions

    def analyze(
        self,
        video_path: str,
        actions: Optional[List[str]] = None,
        task_id: Optional[str] = None,
    ) -> DiagnosisResult:
        """
        主分析方法

        Args:
            video_path: 视频文件路径
            actions: 期望的动作列表，默认 ["blink", "mouth_open", "nod", "shake_head"]
            task_id: 可选的任务 ID

        Returns:
            DiagnosisResult 诊断结果
        """
        if actions is None:
            actions = ["blink", "mouth_open", "nod", "shake_head"]

        # 记录开始时间
        start_time = time.time()

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")

        try:
            # 获取视频基本信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 处理帧数为负数的情况
            if total_frames <= 0:
                temp_cap = cv2.VideoCapture(video_path)
                actual_frames = 0
                while temp_cap.read()[0]:
                    actual_frames += 1
                temp_cap.release()
                total_frames = actual_frames

            duration = total_frames / fps if fps > 0 else 0.0

            video_info = VideoInfo(
                path=video_path,
                filename=Path(video_path).name,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
                duration=duration,
            )

            # 逐帧分析
            frame_analyses: List[FrameAnalysis] = []
            frame_count = 0
            max_frames = min(total_frames, 500)  # 限制最大分析帧数

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                analysis = self._analyze_frame(frame, frame_count)
                if analysis:
                    frame_analyses.append(analysis)

                frame_count += 1

            # 重置检测器状态
            self.head_detector.reset()
            self.fast_detector.reset()

            # 统计信息
            frames_with_face = [f for f in frame_analyses if f.has_face]
            face_detection_rate = (
                len(frames_with_face) / len(frame_analyses) if frame_analyses else 0.0
            )

            # 动作统计
            blink_count = sum(1 for f in frame_analyses if f.is_blink)
            mouth_open_count = sum(1 for f in frame_analyses if f.is_mouth_open)
            nod_count = sum(
                1 for f in frame_analyses if f.head_action and "nod" in f.head_action
            )
            shake_count = sum(
                1
                for f in frame_analyses
                if f.head_action
                and ("turn" in f.head_action or "shake" in f.head_action)
            )

            # 质量评估
            quality_scores = [
                f.quality_score for f in frame_analyses if f.quality_score is not None
            ]
            avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0

            # 质量评级
            if avg_quality_score >= 0.8:
                quality_rating = "excellent"
            elif avg_quality_score >= 0.6:
                quality_rating = "good"
            elif avg_quality_score >= 0.4:
                quality_rating = "fair"
            else:
                quality_rating = "poor"

            # 计算动作详情
            action_details = self._calculate_action_details(frame_analyses, actions)

            # 计算总体判定（best_score >= threshold）
            motion_scores = [
                f.motion_score for f in frame_analyses if f.motion_score is not None
            ]
            best_score = max(motion_scores) if motion_scores else 0.0
            overall_passed = best_score >= self.config.threshold

            # 构建结果
            result = DiagnosisResult(
                video_info=video_info,
                total_frames=len(frame_analyses),
                frames_with_face=len(frames_with_face),
                face_detection_rate=face_detection_rate,
                blink_count=blink_count,
                mouth_open_count=mouth_open_count,
                nod_count=nod_count,
                shake_count=shake_count,
                action_details=action_details,
                frame_analyses=frame_analyses,
                avg_quality_score=avg_quality_score,
                quality_rating=quality_rating,
                overall_passed=overall_passed,
                overall_message=f"最佳运动分数：{best_score:.3f} (阈值：{self.config.threshold})",
            )

            # 计算分析时间
            analysis_time = time.time() - start_time

            # 生成建议
            result.suggestions = self._generate_suggestions(result)

            # 填充新增字段
            result.task_id = task_id
            result.analysis_time = analysis_time
            result.threshold = self.config.threshold
            result.ear_threshold = self.config.ear_threshold
            result.mar_threshold = self.config.mar_threshold
            result.yaw_threshold = self.config.yaw_threshold
            result.pitch_threshold = self.config.pitch_threshold

            return result

        finally:
            cap.release()

    def close(self) -> None:
        """释放资源"""
        if hasattr(self.engine, "close"):
            self.engine.close()
