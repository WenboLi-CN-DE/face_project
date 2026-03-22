"""
频域分析器 — AI 生成图像检测

基于 DCT（离散余弦变换）的 GAN/Diffusion 伪影检测。
核心原理：AI 生成图像在频域会留下周期性伪影。

参考:
- Fighting Deepfakes by Detecting GAN DCT Anomalies (2021)
- Generalizable Deepfake Detection via Frequency Masking (2024)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """频域分析器（单例）"""

    _instance: "FrequencyAnalyzer | None" = None

    def __init__(self) -> None:
        logger.info("初始化频域分析器...")
        # DCT 分析参数
        self.block_size = 8  # 8×8 DCT 块
        self.ac_coefficient_threshold = 0.5  # AC 系数异常阈值

    @classmethod
    def get_instance(cls) -> "FrequencyAnalyzer":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze(self, image: np.ndarray, face_bbox: tuple) -> Dict[str, Any]:
        """
        对人脸区域进行频域分析

        Args:
            image: OpenCV BGR 图像
            face_bbox: (x1, y1, x2, y2) 人脸边界框

        Returns:
            {
                "is_ai_generated": bool,      # 是否 AI 生成
                "confidence": float,          # 置信度 (0-1，越高越可能是 AI)
                "anomaly_score": float,       # DCT 异常分数
                "frequency_features": dict,   # 频域特征（调试用）
            }
        """
        result = {
            "is_ai_generated": False,
            "confidence": 0.0,
            "anomaly_score": 0.0,
            "frequency_features": {},
        }

        try:
            # 提取人脸 ROI
            x1, y1, x2, y2 = map(int, face_bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                logger.warning("人脸 ROI 为空")
                return result

            # 转换为灰度图
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Step 1: 8×8 DCT 分块分析
            dct_anomaly = self._analyze_dct_blocks(gray)

            # Step 2: 全局 FFT 分析
            fft_anomaly = self._analyze_fft_spectrum(gray)

            # Step 3: 融合决策
            # DCT 权重 0.6，FFT 权重 0.4
            anomaly_score = 0.6 * dct_anomaly["score"] + 0.4 * fft_anomaly["score"]

            # 阈值判定（可调整）
            # 0.5 是经验阈值，可根据实际数据调整
            is_ai = anomaly_score > 0.5
            confidence = (
                min(anomaly_score / 0.5, 1.0) if is_ai else 1.0 - anomaly_score / 0.5
            )

            result["is_ai_generated"] = is_ai
            result["confidence"] = round(float(confidence), 4)
            result["anomaly_score"] = round(float(anomaly_score), 4)
            result["frequency_features"] = {
                "dct_score": round(float(dct_anomaly["score"]), 4),
                "fft_score": round(float(fft_anomaly["score"]), 4),
                "dct_high_freq_ratio": round(
                    float(dct_anomaly.get("high_freq_ratio", 0)), 4
                ),
                "fft_radial_mean": round(float(fft_anomaly.get("radial_mean", 0)), 4),
            }

            logger.info(
                "频域分析 anomaly_score=%.4f is_ai=%s confidence=%.4f",
                anomaly_score,
                is_ai,
                confidence,
            )

            return result

        except Exception as e:
            logger.error("频域分析失败：%s", str(e))
            return result

    def _analyze_dct_blocks(self, gray: np.ndarray) -> Dict[str, float]:
        """
        8×8 DCT 分块分析，检测 GAN 棋盘格伪影

        返回：{"score": 0-1, "high_freq_ratio": float}
        """
        h, w = gray.shape
        block_h, block_w = self.block_size, self.block_size

        # 裁剪到 block 的整数倍
        h_crop = (h // block_h) * block_h
        w_crop = (w // block_w) * block_w
        gray_cropped = gray[:h_crop, :w_crop]

        ac_coeffs = []
        dc_coeffs = []

        # 遍历所有 8×8 块
        for i in range(0, h_crop, block_h):
            for j in range(0, w_crop, block_w):
                block = gray_cropped[i : i + block_h, j : j + block_w].astype(
                    np.float32
                )
                dct_block = cv2.dct(block)

                # DC 系数（左上角）
                dc_coeffs.append(abs(dct_block[0, 0]))

                # AC 系数（高频部分）
                ac_block = dct_block[1:, 1:]
                ac_coeffs.append(np.mean(np.abs(ac_block)))

        if not ac_coeffs or not dc_coeffs:
            return {"score": 0.0, "high_freq_ratio": 0.0}

        # 计算 AC/DC 比率（GAN 通常高频异常高）
        ac_mean = np.mean(ac_coeffs)
        dc_mean = np.mean(dc_coeffs)
        high_freq_ratio = ac_mean / (dc_mean + 1e-8)

        # 归一化到 0-1（经验公式）
        # 正常图像：0.1-0.3，GAN 生成：0.3-0.8
        score = min(max((high_freq_ratio - 0.15) / 0.35, 0.0), 1.0)

        return {"score": score, "high_freq_ratio": high_freq_ratio}

    def _analyze_fft_spectrum(self, gray: np.ndarray) -> Dict[str, float]:
        """
        全局 FFT 频谱分析，检测 Diffusion 模型痕迹

        返回：{"score": 0-1, "radial_mean": float}
        """
        # FFT 变换
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1e-8)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # 计算径向平均频谱（从中心到边缘）
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        max_radius = min(center_h, center_w)

        # 分 10 个半径区间计算平均能量
        radial_means = []
        for r in range(1, 11):
            r_low = (r - 1) * max_radius // 10
            r_high = r * max_radius // 10
            mask = (radius >= r_low) & (radius < r_high)
            radial_means.append(np.mean(magnitude[mask]))

        # Diffusion 模型通常高频衰减异常
        # 计算高频/低频比率
        low_freq_mean = np.mean(radial_means[:3])  # 低频（中心）
        high_freq_mean = np.mean(radial_means[-3:])  # 高频（边缘）
        radial_ratio = high_freq_mean / (low_freq_mean + 1e-8)

        # 归一化到 0-1
        # 正常图像：0.3-0.6，Diffusion：0.1-0.3（高频缺失）
        score = min(max((0.5 - radial_ratio) / 0.3, 0.0), 1.0)

        return {"score": score, "radial_mean": radial_ratio}
