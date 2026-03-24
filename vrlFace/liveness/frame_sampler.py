"""关键帧采样器 — 为静默检测提供高质量人脸帧"""

import cv2
import numpy as np
import logging
from typing import List
from .mediapipe_detector import MediaPipeLivenessDetector

logger = logging.getLogger(__name__)


class FrameSampler:
    """关键帧采样器"""

    def __init__(self):
        self.detector = MediaPipeLivenessDetector()

    def sample_keyframes(
        self,
        video_path: str,
        num_frames: int = 5,
        min_quality: float = 0.6,
        max_angle: float = 15.0,
        max_scan_frames: int = 100,
    ) -> List[np.ndarray]:
        """采样高质量正面人脸帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频：{video_path}")
            return []

        candidates = []
        frame_idx = 0

        while frame_idx < max_scan_frames:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.detector.detect_liveness(frame)
            if not result.get("face_detected"):
                frame_idx += 1
                continue

            yaw = result.get("yaw", 0.0)
            pitch = result.get("pitch", 0.0)

            if abs(yaw) > max_angle or abs(pitch) > max_angle:
                frame_idx += 1
                continue

            quality = self._calculate_quality_score(
                frame, face_size=0.3, angle=max(abs(yaw), abs(pitch))
            )

            if quality >= min_quality:
                candidates.append((quality, frame.copy()))

            frame_idx += 1

        cap.release()

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [frame for _, frame in candidates[:num_frames]]

    def _calculate_quality_score(
        self, frame: np.ndarray, face_size: float, angle: float
    ) -> float:
        """计算帧质量分数"""
        size_score = min(face_size / 0.5, 1.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 500.0, 1.0)

        frontal_score = 1.0 - (angle / 15.0)

        quality = 0.4 * size_score + 0.4 * sharpness_score + 0.2 * frontal_score
        return max(0.0, min(1.0, quality))
