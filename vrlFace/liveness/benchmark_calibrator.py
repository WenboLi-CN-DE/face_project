"""
基准帧校准模块

功能：
- 动态检测高质量正面人脸帧作为基准
- 从基准帧提取人脸 embedding 和 landmarks 特征
- 后续帧与基准比对，确保是同一人（防替换攻击）
- 根据基准状态个性化校准动作检测阈值

使用场景：
- 视频活体检测
- 实时摄像头活体检测
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque
import time

from .config import LivenessConfig


class BenchmarkFrame(NamedTuple):
    """基准帧数据"""

    embedding: np.ndarray
    landmarks: np.ndarray
    quality_score: float
    face_bbox: Tuple[int, int, int, int]
    pitch: float
    yaw: float
    timestamp: float
    frame_index: int


@dataclass
class BenchmarkConfig:
    """基准帧配置"""

    # 基准采集阶段
    benchmark_duration: float = 2.0  # 基准采集时长（秒）
    min_benchmark_frames: int = 3  # 最少基准帧数
    max_benchmark_frames: int = 10  # 最多基准帧数

    # 质量阈值
    min_quality_score: float = 0.6  # 最低质量分数
    max_face_angle: float = 15.0  # 最大人脸角度（度）

    # 比对阈值
    embedding_threshold: float = 0.55  # embedding 相似度阈值
    landmark_threshold: float = 0.70  # landmarks 相似度阈值

    # 校准参数
    enable_threshold_calibration: bool = False  # 是否启用阈值校准
    calibration_factor: float = 0.1  # 校准因子

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.benchmark_duration <= 0:
            return False
        if (
            self.min_benchmark_frames <= 0
            or self.max_benchmark_frames < self.min_benchmark_frames
        ):
            return False
        if not (0.0 <= self.min_quality_score <= 1.0):
            return False
        if not (0.0 <= self.embedding_threshold <= 1.0):
            return False
        return True


class BenchmarkCalibrator:
    """
    基准帧校准器

    工作流程：
    1. 采集阶段：动态收集高质量正面帧作为基准
    2. 校准阶段：后续帧与基准比对，确保同一人
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.reset()

    def reset(self):
        """重置校准器状态"""
        # 基准帧集合
        self.benchmark_frames: List[BenchmarkFrame] = []

        # 采集状态
        self.is_collecting = True  # 是否正在采集基准
        self.collection_start_time: Optional[float] = None
        self.collection_start_frame: int = 0

        # 基准特征（采集完成后计算）
        self.benchmark_embedding: Optional[np.ndarray] = None
        self.benchmark_landmarks: Optional[np.ndarray] = None
        self.benchmark_quality: float = 0.0
        self.benchmark_pitch: float = 0.0
        self.benchmark_yaw: float = 0.0

        # 统计信息
        self.total_frames_processed: int = 0
        self.benchmark_frames_collected: int = 0

        # 比对历史
        self.verification_history: deque = deque(maxlen=30)

    def start_collection(self, frame_index: int = 0):
        """开始基准采集"""
        self.reset()
        self.is_collecting = True
        self.collection_start_time = time.time()
        self.collection_start_frame = frame_index

    def add_candidate_frame(
        self,
        embedding: np.ndarray,
        landmarks: np.ndarray,
        quality_score: float,
        face_bbox: Tuple[int, int, int, int],
        pitch: float,
        yaw: float,
        frame_index: int,
    ) -> bool:
        """
        添加候选基准帧

        只收集高质量、正面的人脸帧

        Returns:
            是否成功添加为基准帧
        """
        if not self.is_collecting:
            return False

        # 检查采集时间窗口
        elapsed = time.time() - self.collection_start_time
        if elapsed > self.config.benchmark_duration:
            self._finalize_benchmark()
            return False

        # 质量检查
        if quality_score < self.config.min_quality_score:
            return False

        # 角度检查（确保是正面）
        if (
            abs(pitch) > self.config.max_face_angle
            or abs(yaw) > self.config.max_face_angle
        ):
            return False

        # 添加到基准帧集合
        benchmark_frame = BenchmarkFrame(
            embedding=embedding.copy(),
            landmarks=landmarks.copy(),
            quality_score=quality_score,
            face_bbox=face_bbox,
            pitch=pitch,
            yaw=yaw,
            timestamp=time.time(),
            frame_index=frame_index,
        )

        self.benchmark_frames.append(benchmark_frame)
        self.benchmark_frames_collected += 1

        # 检查是否已达到最少帧数
        if len(self.benchmark_frames) >= self.config.min_benchmark_frames:
            # 继续收集直到达到最大帧数或时间窗口结束
            if len(self.benchmark_frames) >= self.config.max_benchmark_frames:
                self._finalize_benchmark()

        return True

    def _finalize_benchmark(self):
        """完成基准采集，计算基准特征"""
        if len(self.benchmark_frames) == 0:
            self.is_collecting = False
            return

        # 计算平均 embedding
        embeddings = np.array([bf.embedding for bf in self.benchmark_frames])
        self.benchmark_embedding = np.mean(embeddings, axis=0)

        # 计算平均 landmarks
        landmarks_list = np.array([bf.landmarks for bf in self.benchmark_frames])
        self.benchmark_landmarks = np.mean(landmarks_list, axis=0)

        # 计算平均质量分数
        self.benchmark_quality = np.mean(
            [bf.quality_score for bf in self.benchmark_frames]
        )

        # 计算平均角度
        self.benchmark_pitch = np.mean([bf.pitch for bf in self.benchmark_frames])
        self.benchmark_yaw = np.mean([bf.yaw for bf in self.benchmark_frames])

        # 结束采集
        self.is_collecting = False

        print(
            f"✅ 基准采集完成：收集 {len(self.benchmark_frames)} 帧，平均质量={self.benchmark_quality:.3f}"
        )

    def verify_frame(
        self,
        embedding: np.ndarray,
        landmarks: np.ndarray,
        pitch: float,
        yaw: float,
    ) -> Dict[str, Any]:
        """
        验证当前帧是否与基准帧匹配（防替换攻击）

        Returns:
            验证结果字典
        """
        if self.is_collecting:
            return {
                "verified": False,
                "reason": "STILL_COLLECTING",
                "embedding_similarity": 0.0,
                "landmark_similarity": 0.0,
                "is_same_person": False,
            }

        if self.benchmark_embedding is None or len(self.benchmark_frames) == 0:
            return {
                "verified": False,
                "reason": "NO_BENCHMARK",
                "embedding_similarity": 0.0,
                "landmark_similarity": 0.0,
                "is_same_person": False,
            }

        # 计算 embedding 相似度
        embedding_sim = self._calculate_embedding_similarity(
            embedding, self.benchmark_embedding
        )

        # 计算 landmarks 相似度
        landmark_sim = self._calculate_landmark_similarity(
            landmarks, self.benchmark_landmarks
        )

        # 综合判定
        is_same_person = (
            embedding_sim >= self.config.embedding_threshold
            and landmark_sim >= self.config.landmark_threshold
        )

        # 记录历史
        self.verification_history.append(
            {
                "embedding_sim": embedding_sim,
                "landmark_sim": landmark_sim,
                "is_same_person": is_same_person,
            }
        )

        # 判定原因
        if is_same_person:
            reason = "VERIFIED"
        else:
            reasons = []
            if embedding_sim < self.config.embedding_threshold:
                reasons.append(f"embedding 过低 ({embedding_sim:.3f})")
            if landmark_sim < self.config.landmark_threshold:
                reasons.append(f"landmarks 过低 ({landmark_sim:.3f})")
            reason = "MISMATCH: " + ", ".join(reasons)

        return {
            "verified": is_same_person,
            "reason": reason,
            "embedding_similarity": round(embedding_sim, 4),
            "landmark_similarity": round(landmark_sim, 4),
            "is_same_person": is_same_person,
        }

    def _calculate_embedding_similarity(
        self,
        embedding: np.ndarray,
        benchmark_embedding: np.ndarray,
    ) -> float:
        """计算 embedding 余弦相似度"""
        from numpy.linalg import norm

        norm_emb = norm(embedding)
        norm_bench = norm(benchmark_embedding)

        if norm_emb < 1e-8 or norm_bench < 1e-8:
            return 0.0

        similarity = np.dot(embedding, benchmark_embedding) / (norm_emb * norm_bench)
        return float(np.clip(similarity, 0.0, 1.0))

    def _calculate_landmark_similarity(
        self,
        landmarks: np.ndarray,
        benchmark_landmarks: np.ndarray,
    ) -> float:
        """
        计算 landmarks 相似度

        使用归一化均方误差（NMSE）转换为相似度
        """
        if landmarks.shape != benchmark_landmarks.shape:
            return 0.0

        # 计算均方误差
        mse = np.mean((landmarks - benchmark_landmarks) ** 2)

        # 转换为相似度（MSE 越小相似度越高）
        # 假设 landmarks 坐标归一化到 [0, 1]，MSE 最大约 1.0
        similarity = 1.0 / (1.0 + mse * 10)  # 放大 MSE 影响

        return float(np.clip(similarity, 0.0, 1.0))

    def get_calibrated_threshold(self, base_threshold: float, metric: str) -> float:
        """
        获取校准后的阈值

        根据基准帧的质量动态调整检测阈值

        Args:
            base_threshold: 基础阈值
            metric: 指标类型 ("quality", "angle", "embedding")

        Returns:
            校准后的阈值
        """
        if not self.config.enable_threshold_calibration:
            return base_threshold

        if self.benchmark_embedding is None:
            return base_threshold

        # 根据基准质量调整阈值
        quality_factor = self.benchmark_quality - 0.5  # [-0.5, 0.5]

        if metric == "quality":
            # 高质量基准 → 提高阈值（更严格）
            # 低质量基准 → 降低阈值（更宽松）
            calibrated = (
                base_threshold + quality_factor * self.config.calibration_factor
            )
        elif metric == "angle":
            # 基准角度越小（越正面）→ 提高阈值
            angle_deviation = (abs(self.benchmark_pitch) + abs(self.benchmark_yaw)) / 2
            angle_factor = 1.0 - min(angle_deviation / self.config.max_face_angle, 1.0)
            calibrated = (
                base_threshold + (angle_factor - 0.5) * self.config.calibration_factor
            )
        else:
            calibrated = base_threshold

        # 限制范围
        return float(np.clip(calibrated, base_threshold - 0.15, base_threshold + 0.15))

    def get_status(self) -> Dict[str, Any]:
        """获取校准器状态"""
        if self.is_collecting:
            elapsed = (
                time.time() - self.collection_start_time
                if self.collection_start_time
                else 0
            )
            return {
                "status": "COLLECTING",
                "elapsed": round(elapsed, 2),
                "duration": self.config.benchmark_duration,
                "frames_collected": len(self.benchmark_frames),
                "min_required": self.config.min_benchmark_frames,
            }
        elif self.benchmark_embedding is not None:
            return {
                "status": "READY",
                "frames_collected": len(self.benchmark_frames),
                "benchmark_quality": round(self.benchmark_quality, 4),
                "benchmark_pitch": round(self.benchmark_pitch, 2),
                "benchmark_yaw": round(self.benchmark_yaw, 2),
                "verification_count": len(self.verification_history),
            }
        else:
            return {
                "status": "NOT_STARTED",
            }

    def is_ready(self) -> bool:
        """基准是否已准备好"""
        return not self.is_collecting and self.benchmark_embedding is not None

    def is_collecting_benchmark(self) -> bool:
        """是否正在采集基准"""
        return self.is_collecting
