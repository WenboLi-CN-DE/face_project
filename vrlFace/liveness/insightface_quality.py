"""
InsightFace 人脸质量检测器

基于 InsightFace 的人脸检测和质量评估：
- 人脸检测与定位
- 人脸质量评分
- 模糊度检测
- 光照评估
- 人脸角度估计
"""

import cv2
import numpy as np
import insightface
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


class InsightFaceQualityDetector:
    """InsightFace 人脸质量检测器"""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_path: Optional[str] = None,
        det_size: Tuple[int, int] = (320, 320),
        ctx_id: int = -1,
    ):
        """
        初始化 InsightFace 检测器

        Args:
            model_name: 模型名称
            model_path: 模型路径 (None 使用默认路径)
            det_size: 检测尺寸 (越小越快)
            ctx_id: GPU ID (-1 表示 CPU)
        """
        # 初始化 InsightFace App
        self.app = insightface.app.FaceAnalysis(
            name=model_name,
            root=model_path or str(Path.home() / ".insightface" / "models"),
            providers=["CPUExecutionProvider"]
            if ctx_id < 0
            else ["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # 配置参数
        self.det_size = det_size
        self.ctx_id = ctx_id

        # 性能统计
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = 0.0

    def detect_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测人脸质量

        Args:
            frame: BGR 图像

        Returns:
            检测结果字典
        """
        self.frame_count += 1
        self._update_fps()

        # 默认结果
        default_result = {
            "face_detected": False,
            "face_count": 0,
            "quality_score": 0.0,
            "face_bbox": None,
            "face_landmarks": None,
            "face_angle": 0.0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "fps": self.fps,
        }

        # 人脸检测
        faces = self.app.get(frame)

        if not faces:
            return default_result

        # 选择最大人脸 (最清晰)
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

        # 计算质量分数
        quality_score = self._calculate_quality_score(face, frame)

        # 计算模糊度
        blur_score = self._calculate_blur(frame, face.bbox)

        # 计算亮度
        brightness = self._calculate_brightness(frame, face.bbox)

        # 计算人脸角度
        face_angle = self._calculate_face_angle(face)

        return {
            "face_detected": True,
            "face_count": len(faces),
            "quality_score": quality_score,
            "face_bbox": tuple(face.bbox.astype(int)),
            "face_landmarks": face.landmark_2d_106
            if hasattr(face, "landmark_2d_106")
            else None,
            "face_angle": face_angle,
            "blur_score": blur_score,
            "brightness": brightness,
            "det_score": face.det_score if hasattr(face, "det_score") else 1.0,
            "embedding": face.embedding if hasattr(face, "embedding") else None,
            "fps": self.fps,
        }

    def _calculate_quality_score(self, face, frame: np.ndarray) -> float:
        """
        计算综合质量分数

        Args:
            face: InsightFace 人脸对象
            frame: BGR 图像

        Returns:
            质量分数 (0-1)
        """
        scores = []

        # 1. 检测置信度
        det_score = getattr(face, "det_score", 1.0)
        scores.append(det_score)

        # 2. 人脸尺寸分数 (越大越清晰)
        bbox = face.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / frame_area

        # 理想人脸尺寸占比 5%-30%
        if size_ratio < 0.01:
            size_score = 0.2
        elif size_ratio < 0.05:
            size_score = 0.5
        elif size_ratio < 0.3:
            size_score = 1.0
        else:
            size_score = 0.7

        scores.append(size_score)

        # 3. 模糊度分数
        blur_score = self._calculate_blur(frame, bbox)
        scores.append(1.0 - blur_score)  # 模糊度越低越好

        # 4. 亮度分数
        brightness = self._calculate_brightness(frame, bbox)
        # 理想亮度 0.3-0.7
        if 0.3 <= brightness <= 0.7:
            brightness_score = 1.0
        elif brightness < 0.3:
            brightness_score = brightness / 0.3
        else:
            brightness_score = (1.0 - brightness) / 0.3

        scores.append(brightness_score)

        # 5. 角度分数
        angle = self._calculate_face_angle(face)
        angle_score = max(0.0, 1.0 - angle / 45.0)  # 超过 45 度分数降低
        scores.append(angle_score)

        # 加权平均
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        quality_score = sum(s * w for s, w in zip(scores, weights))

        return np.clip(quality_score, 0.0, 1.0)

    def _calculate_blur(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        计算模糊度 (Laplacian 方差)

        Args:
            frame: BGR 图像
            bbox: 人脸边界框

        Returns:
            模糊度 (0-1, 越大越模糊)
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

        # 裁剪人脸区域
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            return 1.0

        # 转灰度
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Laplacian 方差
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 归一化到 0-1 (经验阈值)
        # variance < 50: 很模糊
        # variance > 200: 很清晰
        blur_score = 1.0 - min(variance / 200.0, 1.0)

        return blur_score

    def _calculate_brightness(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        计算亮度

        Args:
            frame: BGR 图像
            bbox: 人脸边界框

        Returns:
            亮度 (0-1)
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

        # 裁剪人脸区域
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            return 0.0

        # 转灰度并计算平均亮度
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0

        return brightness

    def _calculate_face_angle(self, face) -> float:
        """
        计算人脸偏转角度

        Args:
            face: InsightFace 人脸对象

        Returns:
            角度值 (度)
        """
        # 如果有 3D 关键点，计算姿态
        if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
            landmarks_3d = face.landmark_3d_68

            # 使用眼睛和鼻子计算角度
            left_eye = landmarks_3d[36]
            right_eye = landmarks_3d[45]
            nose_tip = landmarks_3d[30]

            # 计算水平角度 (yaw)
            eye_vector = right_eye - left_eye
            angle = np.arctan2(eye_vector[1], eye_vector[0])

            return abs(np.degrees(angle))

        # 如果没有 3D 信息，使用 2D 关键点估算
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            landmarks_2d = face.landmark_2d_106

            # 使用眼睛连线计算 roll 角度
            left_eye = landmarks_2d[42]  # 左眼中心
            right_eye = landmarks_2d[43]  # 右眼中心

            eye_vector = right_eye - left_eye
            angle = np.arctan2(eye_vector[1], eye_vector[0])

            return abs(np.degrees(angle))

        return 0.0

    def _update_fps(self):
        """更新 FPS 统计"""
        import time

        current_time = time.time()

        if self.last_fps_time > 0:
            elapsed = current_time - self.last_fps_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = current_time
        else:
            self.last_fps_time = current_time

    def reset(self):
        """重置状态"""
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = 0.0
