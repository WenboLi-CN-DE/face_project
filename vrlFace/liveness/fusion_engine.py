"""
纯 MediaPipe 活体检测引擎

仅使用 MediaPipe 进行动作检测：
- 眨眼检测（EAR 阈值）
- 张嘴检测（MAR 阈值）
- 头部动作检测（Yaw/Pitch 峰峰值）
- 质量评分（基于人脸尺寸和亮度）
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, NamedTuple
import time

from .config import LivenessConfig
from .mediapipe_detector import MediaPipeLivenessDetector
from .benchmark_calibrator import BenchmarkCalibrator, BenchmarkConfig


class LivenessResult(NamedTuple):
    is_live: bool
    score: float
    confidence: float
    quality_score: float
    motion_score: float
    temporal_score: float
    details: Dict[str, Any]
    reason: str = ""


class LivenessFusionEngine:
    def __init__(self, config: Optional[LivenessConfig] = None):
        self.config = config or LivenessConfig.cpu_fast_config()

        self.mp_detector = MediaPipeLivenessDetector(
            ear_threshold=self.config.ear_threshold,
            mar_threshold=self.config.mar_threshold,
            head_movement_threshold=0.0,
            yaw_threshold=self.config.yaw_threshold,
            pitch_threshold=self.config.pitch_threshold,
            window_size=self.config.window_size,
            max_faces=self.config.max_faces,
            action_confirm_frames=self.config.action_confirm_frames,
        )

        self.score_history: deque = deque(maxlen=self.config.smooth_window)
        self.quality_history: deque = deque(maxlen=self.config.window_size)
        self.motion_history: deque = deque(maxlen=self.config.window_size)

        self.is_running = False
        self.start_time = 0.0
        self.frame_count = 0

        self.challenge_active = False
        self.challenge_start_time = 0.0
        self.challenge_actions_completed = set()

        self.enable_benchmark: bool = getattr(self.config, "enable_benchmark", False)
        self.calibrator: Optional[BenchmarkCalibrator] = None
        if self.enable_benchmark:
            benchmark_config = BenchmarkConfig(
                benchmark_duration=getattr(self.config, "benchmark_duration", 2.0),
                min_benchmark_frames=getattr(self.config, "benchmark_min_frames", 3),
                max_benchmark_frames=getattr(self.config, "benchmark_max_frames", 10),
                min_quality_score=getattr(self.config, "benchmark_min_quality", 0.6),
                max_face_angle=getattr(self.config, "benchmark_max_angle", 15.0),
                embedding_threshold=getattr(self.config, "embedding_threshold", 0.55),
                enable_threshold_calibration=getattr(
                    self.config, "enable_threshold_calibration", False
                ),
            )
            self.calibrator = BenchmarkCalibrator(benchmark_config)

    def _resize_for_inference(self, frame: np.ndarray) -> np.ndarray:
        max_w = self.config.max_width
        if max_w <= 0:
            return frame
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        scale = max_w / w
        new_w = max_w
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def process_frame(self, frame: np.ndarray) -> LivenessResult:
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()

        self.frame_count += 1
        small_frame = self._resize_for_inference(frame)

        mp_skip = (self.config.skip_frames > 0) and (
            (self.frame_count % (self.config.skip_frames + 1)) != 0
        )

        # 同步推理：直接在主线程调用，消除后台线程导致的 1~2 帧延迟。
        # 对于眨眼（3~8 帧）等短暂动作，延迟会直接导致漏检。
        # MediaPipe FaceLandmarker 在 CPU 上单帧约 8~15ms，实时性足够。
        motion_result = self.mp_detector.detect_liveness(small_frame, skip=mp_skip)

        face_detected = motion_result.get("face_detected", False)
        quality_score = motion_result.get("quality_score", 0.0)
        motion_score = motion_result.get("score", 0.0)

        if not face_detected:
            self._add_to_history(0.0, 0.0, 0.0)
            return LivenessResult(
                is_live=False,
                score=0.0,
                confidence=0.0,
                quality_score=0.0,
                motion_score=0.0,
                temporal_score=0.0,
                details={"motion": motion_result},
                reason="NO_FACE_DETECTED",
            )

        # 基准帧校准逻辑
        benchmark_status = None
        if self.enable_benchmark and self.calibrator is not None:
            embedding = motion_result.get("embedding")
            landmarks = motion_result.get("landmarks")
            pitch = motion_result.get("pitch", 0.0)
            yaw = motion_result.get("yaw", 0.0)
            face_bbox = motion_result.get("face_bbox")

            if (
                embedding is not None
                and landmarks is not None
                and face_bbox is not None
            ):
                if self.calibrator.is_collecting_benchmark():
                    # 采集基准帧
                    added = self.calibrator.add_candidate_frame(
                        embedding=embedding,
                        landmarks=landmarks,
                        quality_score=quality_score,
                        face_bbox=face_bbox,
                        pitch=pitch,
                        yaw=yaw,
                        frame_index=self.frame_count,
                    )
                    benchmark_status = self.calibrator.get_status()
                else:
                    # 验证当前帧与基准的匹配度
                    verification = self.calibrator.verify_frame(
                        embedding=embedding,
                        landmarks=landmarks,
                        pitch=pitch,
                        yaw=yaw,
                    )
                    benchmark_status = verification

                    # 如果不匹配（可能换人），降低活体分数
                    if not verification.get("verified", True):
                        motion_score *= 0.5  # 惩罚分数

        self._add_to_history(motion_score, quality_score, motion_score)

        smoothed_score = self._smooth_score(motion_score)

        is_live = smoothed_score > self.config.threshold
        confidence = self._calculate_confidence(smoothed_score)

        details = {
            "motion": motion_result,
            "current_action": motion_result.get("current_action", "none"),
            "is_blinking": motion_result.get("is_blinking", False),
            "is_mouth_open": motion_result.get("is_mouth_open", False),
            "yaw": motion_result.get("yaw", 0.0),
            "benchmark": benchmark_status,
        }

        reason = self._determine_reason(is_live, details, face_detected)

        return LivenessResult(
            is_live=is_live,
            score=smoothed_score,
            confidence=confidence,
            quality_score=quality_score,
            motion_score=motion_score,
            temporal_score=0.0,
            details=details,
            reason=reason,
        )

    def _add_to_history(
        self, fused_score: float, quality_score: float, motion_score: float
    ):
        self.score_history.append(fused_score)
        self.quality_history.append(quality_score)
        self.motion_history.append(motion_score)

    def _smooth_score(self, current_score: float) -> float:
        if len(self.score_history) < self.config.smooth_window:
            return current_score
        recent_scores = list(self.score_history)[-self.config.smooth_window :]
        return float(np.mean(recent_scores))

    def _calculate_confidence(self, score: float) -> float:
        if len(self.score_history) < 5:
            return 0.5
        std = (
            float(np.std(list(self.score_history)[-10:]))
            if len(self.score_history) >= 10
            else 0.0
        )
        confidence = 1.0 - min(std * 5, 0.5)
        if score > 0.8:
            confidence = min(confidence + 0.2, 1.0)
        elif score < 0.3:
            confidence = min(confidence + 0.1, 1.0)
        return confidence

    def _determine_reason(
        self, is_live: bool, details: Dict, face_detected: bool
    ) -> str:
        if not face_detected:
            return "未检测到人脸"

        if is_live:
            reasons = []
            motion = details.get("motion", {})
            if motion.get("blink_detected", False):
                reasons.append("眨眼")
            if motion.get("mouth_moved", False):
                reasons.append("嘴部动作")
            if motion.get("head_moved", False):
                reasons.append("头部移动")
            if not reasons:
                reasons.append("动作正常")
            return "活体：" + ", ".join(reasons)
        else:
            motion_score = details.get("motion", {}).get("score", 0.0)
            if motion_score < 0.3:
                return "动作不足"
            return f"分数过低 ({motion_score:.2f})"

    def reset(self):
        self.mp_detector.reset()
        self.score_history.clear()
        self.quality_history.clear()
        self.motion_history.clear()
        self.is_running = False
        self.frame_count = 0
        self.challenge_active = False
        self.challenge_actions_completed.clear()
        if self.calibrator is not None:
            self.calibrator.reset()

    def close(self):
        self.mp_detector.close()

    def start_challenge(self, required_actions: Optional[list] = None):
        self.challenge_active = True
        self.challenge_start_time = time.time()
        self.challenge_actions_completed = set()
        if required_actions:
            self.config.required_actions = tuple(required_actions)

    def check_challenge_progress(self, result: LivenessResult) -> Dict[str, Any]:
        if not self.challenge_active:
            return {"active": False}
        progress = {}
        elapsed = time.time() - self.challenge_start_time
        if "blink" in self.config.required_actions:
            progress["blink"] = result.details.get("motion", {}).get(
                "blink_detected", False
            )
        if "mouth_open" in self.config.required_actions:
            progress["mouth_open"] = result.details.get("motion", {}).get(
                "mouth_moved", False
            )
        timeout = elapsed > self.config.challenge_timeout
        all_completed = all(progress.values()) if progress else False
        return {
            "active": True,
            "progress": progress,
            "elapsed": elapsed,
            "timeout": timeout,
            "completed": all_completed,
        }

    def end_challenge(self) -> bool:
        if not self.challenge_active:
            return False
        self.challenge_active = False
        return True
