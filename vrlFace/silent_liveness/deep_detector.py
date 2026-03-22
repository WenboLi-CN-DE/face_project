"""
启发式 AI 生成图像检测器

基于图像质量特征的启发式检测（无需深度学习模型）：
- 锐度分析（Laplacian 方差）
- 噪声模式估计
- 颜色分布分析
- 人脸对称性检测

相比频域分析，对现代 Diffusion 模型（如豆包 Seedream 4.5）有一定补充作用。

TODO: 后续可集成 UniversalFakeDetect 等预训练模型增强检测能力
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HeuristicDetector:
    """
    启发式 AI 检测器

    基于图像质量特征，无需加载深度学习模型。
    优势：轻量、快速、无额外依赖
    局限：对先进 Diffusion 模型检测率有限（约 60-70%）
    """

    _instance: "HeuristicDetector | None" = None

    def __init__(self) -> None:
        logger.info("初始化启发式检测器...")
        # 图像质量特征权重
        self.weight_sharpness = 0.30
        self.weight_noise = 0.25
        self.weight_color_dist = 0.25
        self.weight_face_symmetry = 0.20

    @classmethod
    def get_instance(cls) -> "HeuristicDetector":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, image: np.ndarray, face_bbox: tuple) -> Dict[str, Any]:
        """
        执行深度学习风格的 AI 检测（当前为启发式方法）

        Args:
            image: OpenCV BGR 图像
            face_bbox: (x1, y1, x2, y2) 人脸边界框

        Returns:
            {
                "is_ai_generated": bool,
                "confidence": float,
                "anomaly_score": float,
                "dl_features": dict,
            }
        """
        result = {
            "is_ai_generated": False,
            "confidence": 0.0,
            "anomaly_score": 0.0,
            "dl_features": {},
        }

        try:
            # 提取人脸 ROI
            x1, y1, x2, y2 = map(int, face_bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                return result

            # 特征提取
            sharpness_score = self._analyze_sharpness(face_roi)
            noise_score = self._analyze_noise_pattern(face_roi)
            color_dist_score = self._analyze_color_distribution(face_roi)
            symmetry_score = self._analyze_face_symmetry(face_roi)

            # 加权融合
            anomaly_score = (
                self.weight_sharpness * sharpness_score
                + self.weight_noise * noise_score
                + self.weight_color_dist * color_dist_score
                + self.weight_face_symmetry * symmetry_score
            )

            # 阈值判定
            is_ai = anomaly_score > 0.45
            confidence = (
                min(anomaly_score / 0.45, 1.0) if is_ai else 1.0 - anomaly_score / 0.45
            )

            result["is_ai_generated"] = is_ai
            result["confidence"] = round(float(confidence), 4)
            result["anomaly_score"] = round(float(anomaly_score), 4)
            result["dl_features"] = {
                "sharpness": round(float(sharpness_score), 4),
                "noise_pattern": round(float(noise_score), 4),
                "color_distribution": round(float(color_dist_score), 4),
                "face_symmetry": round(float(symmetry_score), 4),
            }

            logger.info(
                "深度学习检测 anomaly_score=%.4f is_ai=%s | Sharp=%.4f Noise=%.4f Color=%.4f Sym=%.4f",
                anomaly_score,
                is_ai,
                sharpness_score,
                noise_score,
                color_dist_score,
                symmetry_score,
            )

            return result

        except Exception as e:
            logger.error("深度学习检测失败：%s", str(e), exc_info=True)
            return result

    def _analyze_sharpness(self, face_roi: np.ndarray) -> float:
        """
        分析图像锐度（AI 生成图像通常过度平滑或锐化不自然）

        返回：0-1，越高越可疑
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Laplacian 方差（锐度指标）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)

        # 正常图像：variance 50-500，AI 生成：可能异常高或低
        # 归一化到 0-1（使用对数尺度）
        norm_var = np.log(variance + 1) / np.log(501)

        # 超出正常范围的可疑
        if norm_var < 0.3 or norm_var > 0.8:
            score = min(abs(norm_var - 0.55) * 2, 1.0)
        else:
            score = 0.2  # 正常范围内

        return float(score)

    def _analyze_noise_pattern(self, face_roi: np.ndarray) -> float:
        """
        分析噪声模式（AI 生成图像缺乏真实传感器噪声）

        返回：0-1，越高越可疑
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 估计噪声水平（使用差分法）
        noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        noise_std = np.std(noise)

        # 真实图像：noise_std 5-20，AI 生成：通常<5（过度平滑）
        # 归一化
        noise_level = noise_std / 20.0

        # 噪声过低可疑
        if noise_level < 0.25:
            score = 1.0 - noise_level / 0.25
        else:
            score = 0.2  # 正常

        return float(score)

    def _analyze_color_distribution(self, face_roi: np.ndarray) -> float:
        """
        分析颜色分布（AI 生成图像颜色分布可能异常）

        返回：0-1，越高越可疑
        """
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)

        # 计算各通道的偏度（skewness）
        def calc_skewness(channel):
            mean = np.mean(channel)
            std = np.std(channel)
            if std < 1e-8:
                return 0.0
            return np.mean(((channel - mean) / std) ** 3)

        l_skew = abs(calc_skewness(lab[:, :, 0]))
        a_skew = abs(calc_skewness(lab[:, :, 1]))
        b_skew = abs(calc_skewness(lab[:, :, 2]))

        avg_skew = (l_skew + a_skew + b_skew) / 3

        # 正常图像：skewness 0.5-2.0，AI 生成：可能异常
        if avg_skew < 0.5 or avg_skew > 2.5:
            score = min(abs(avg_skew - 1.25) / 1.25, 1.0)
        else:
            score = 0.3

        return float(score)

    def _analyze_face_symmetry(self, face_roi: np.ndarray) -> float:
        """
        分析人脸对称性（AI 生成的人脸可能过度对称或不对称）

        返回：0-1，越高越可疑
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 左右翻转对比
        left_half = gray[:, : w // 2]
        right_half = cv2.flip(gray[:, w // 2 :], 1)

        # 调整大小使维度匹配
        min_h = min(left_half.shape[0], right_half.shape[0])
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:min_h, :min_w]
        right_half = right_half[:min_h, :min_w]

        # 计算相似度
        correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]

        # 真实人脸：correlation 0.7-0.9（有一定对称性但不完美）
        # AI 生成：可能>0.95（过度对称）或<0.6（不对称）
        if np.isnan(correlation):
            return 0.3

        if correlation > 0.92:
            score = min((correlation - 0.92) / 0.08, 1.0)  # 过度对称
        elif correlation < 0.65:
            score = min((0.65 - correlation) / 0.35, 1.0)  # 不对称
        else:
            score = 0.3  # 正常

        return float(score)
