"""
频域分析器 — AI 生成图像检测（增强版）

基于多特征融合的 GAN/Diffusion 伪影检测：
1. DCT 8×8 块分析 - 检测 GAN 棋盘格伪影
2. FFT 频谱分析 - 检测 Diffusion 高频缺失
3. 梯度域分析 - 检测生成图像的边缘异常
4. 颜色一致性分析 - 检测 AI 生成的颜色过度平滑

参考:
- Fighting Deepfakes by Detecting GAN DCT Anomalies (2021)
- Generalizable Deepfake Detection via Frequency Masking (2024)
- DiffusionArtifacts: Detecting Diffusion Model Forgeries (2024)
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """频域分析器（单例）"""

    _instance: "FrequencyAnalyzer | None" = None

    def __init__(self) -> None:
        logger.info("初始化频域分析器（增强版）...")
        # DCT 分析参数
        self.block_size = 8  # 8×8 DCT 块

        # 阈值配置（可根据实际数据调优）
        # 针对现代 Diffusion 模型（如豆包 Seedream 4.5）调整
        self.threshold_dct = 0.25  # DCT 异常阈值（降低，因为现代模型没有棋盘格伪影）
        self.threshold_fft = 0.30  # FFT 异常阈值
        self.threshold_gradient = 0.35  # 梯度异常阈值
        self.threshold_color = 0.30  # 颜色一致性阈值

        # 权重配置 - 梯度特征对 Diffusion 更有效
        self.weight_dct = 0.15  # 降低 DCT 权重（GAN 特征不明显）
        self.weight_fft = 0.20  # FFT 权重
        self.weight_gradient = 0.40  # 提高梯度权重（边缘异常更明显）
        self.weight_color = 0.25  # 颜色权重

    @classmethod
    def get_instance(cls) -> "FrequencyAnalyzer":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze(self, image: np.ndarray, face_bbox: tuple) -> Dict[str, Any]:
        """
        对人脸区域进行多特征频域分析

        Args:
            image: OpenCV BGR 图像
            face_bbox: (x1, y1, x2, y2) 人脸边界框

        Returns:
            {
                "is_ai_generated": bool,      # 是否 AI 生成
                "confidence": float,          # 置信度 (0-1，越高越可能是 AI)
                "anomaly_score": float,       # 综合异常分数
                "frequency_features": dict,   # 各特征分数（调试用）
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

            # 确保 ROI 足够大进行分析
            if face_roi.shape[0] < 32 or face_roi.shape[1] < 32:
                logger.warning("人脸 ROI 太小：%s", face_roi.shape)
                return result

            # 转换为灰度图
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # ===================================================================
            # Step 1: 8×8 DCT 分块分析（检测 GAN 棋盘格伪影）
            # ===================================================================
            dct_result = self._analyze_dct_blocks(gray)

            # ===================================================================
            # Step 2: 全局 FFT 频谱分析（检测 Diffusion 高频缺失）
            # ===================================================================
            fft_result = self._analyze_fft_spectrum(gray)

            # ===================================================================
            # Step 3: 梯度域分析（检测生成图像的边缘异常）
            # ===================================================================
            gradient_result = self._analyze_gradient_domain(gray)

            # ===================================================================
            # Step 4: 颜色一致性分析（检测 AI 生成的颜色过度平滑）
            # ===================================================================
            color_result = self._analyze_color_consistency(face_roi)

            # ===================================================================
            # Step 5: 加权融合决策
            # ===================================================================
            anomaly_score = (
                self.weight_dct * dct_result["score"]
                + self.weight_fft * fft_result["score"]
                + self.weight_gradient * gradient_result["score"]
                + self.weight_color * color_result["score"]
            )

            # 阈值判定
            is_ai = anomaly_score > 0.5

            # 置信度计算
            if is_ai:
                confidence = min(anomaly_score / 0.5, 1.0)
            else:
                confidence = 1.0 - anomaly_score / 0.5

            result["is_ai_generated"] = is_ai
            result["confidence"] = round(float(confidence), 4)
            result["anomaly_score"] = round(float(anomaly_score), 4)
            result["frequency_features"] = {
                "dct_score": round(float(dct_result["score"]), 4),
                "dct_high_freq_ratio": round(
                    float(dct_result.get("high_freq_ratio", 0)), 4
                ),
                "fft_score": round(float(fft_result["score"]), 4),
                "fft_radial_ratio": round(float(fft_result.get("radial_ratio", 0)), 4),
                "gradient_score": round(float(gradient_result["score"]), 4),
                "gradient_entropy": round(float(gradient_result.get("entropy", 0)), 4),
                "color_score": round(float(color_result["score"]), 4),
                "color_variance": round(
                    float(color_result.get("variance_ratio", 0)), 4
                ),
            }

            logger.info(
                "频域分析 anomaly_score=%.4f is_ai=%s confidence=%.4f | "
                "DCT=%.4f FFT=%.4f Grad=%.4f Color=%.4f",
                anomaly_score,
                is_ai,
                confidence,
                dct_result["score"],
                fft_result["score"],
                gradient_result["score"],
                color_result["score"],
            )

            return result

        except Exception as e:
            logger.error("频域分析失败：%s", str(e), exc_info=True)
            return result

    def _analyze_dct_blocks(self, gray: np.ndarray) -> Dict[str, float]:
        """
        8×8 DCT 分块分析，检测 GAN 棋盘格伪影

        核心原理：GAN 的上采样操作会在 DCT 频域留下周期性峰值

        返回：{"score": 0-1, "high_freq_ratio": float}
        """
        h, w = gray.shape
        block_h, block_w = self.block_size, self.block_size

        # 裁剪到 block 的整数倍
        h_crop = (h // block_h) * block_h
        w_crop = (w // block_w) * block_w
        if h_crop < block_h or w_crop < block_w:
            return {"score": 0.0, "high_freq_ratio": 0.0}

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

                # DC 系数（左上角，低频）
                dc_coeffs.append(abs(dct_block[0, 0]))

                # AC 系数（高频部分）
                ac_block = dct_block[1:, 1:]
                ac_coeffs.append(np.mean(np.abs(ac_block)))

        if not ac_coeffs or not dc_coeffs:
            return {"score": 0.0, "high_freq_ratio": 0.0}

        # 计算 AC/DC 比率
        ac_mean = np.mean(ac_coeffs)
        dc_mean = np.mean(dc_coeffs)
        high_freq_ratio = ac_mean / (dc_mean + 1e-8)

        # 改进的归一化公式
        # 正常图像：0.2-0.5，GAN/Diffusion：0.5-1.0+
        # 使用 Sigmoid 风格的映射
        score = float(1.0 / (1.0 + np.exp(-(high_freq_ratio - 0.4) * 10)))

        return {"score": score, "high_freq_ratio": float(high_freq_ratio)}

    def _analyze_fft_spectrum(self, gray: np.ndarray) -> Dict[str, float]:
        """
        全局 FFT 频谱分析，检测 Diffusion 模型痕迹

        核心原理：Diffusion 模型生成的图像高频细节不足，频谱衰减快

        返回：{"score": 0-1, "radial_ratio": float}
        """
        # FFT 变换
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1e-8)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # 计算径向平均频谱
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        max_radius = min(center_h, center_w)

        if max_radius < 5:
            return {"score": 0.0, "radial_ratio": 0.0}

        # 分 10 个半径区间计算平均能量
        radial_means = []
        for r in range(1, 11):
            r_low = (r - 1) * max_radius // 10
            r_high = r * max_radius // 10
            mask = (radius >= r_low) & (radius < r_high)
            if np.any(mask):
                radial_means.append(np.mean(magnitude[mask]))
            else:
                radial_means.append(0.0)

        # 计算高频/低频比率
        low_freq_mean = np.mean(radial_means[:3])  # 低频（中心）
        high_freq_mean = np.mean(radial_means[-3:])  # 高频（边缘）

        if low_freq_mean < 1e-8:
            return {"score": 0.0, "radial_ratio": 0.0}

        radial_ratio = high_freq_mean / low_freq_mean

        # 改进的归一化
        # 正常图像：0.4-0.8，Diffusion：0.2-0.4（高频缺失）
        # radial_ratio 越低，越可能是 AI 生成
        score = float(1.0 / (1.0 + np.exp((radial_ratio - 0.45) * 10)))

        return {"score": score, "radial_ratio": float(radial_ratio)}

    def _analyze_gradient_domain(self, gray: np.ndarray) -> Dict[str, float]:
        """
        梯度域分析，检测生成图像的边缘异常

        核心原理：AI 生成图像的边缘梯度分布异常（过度平滑或不连续）

        返回：{"score": 0-1, "entropy": float}
        """
        # 计算 Sobel 梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 梯度直方图
        hist, _ = np.histogram(gradient_magnitude.flatten(), bins=64, range=(0, 500))
        hist_norm = hist / (hist.sum() + 1e-8)

        # 计算梯度熵（AI 生成图像通常熵较低）
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-8))
        max_entropy = np.log(64)  # 最大熵

        # 归一化熵
        norm_entropy = entropy / max_entropy

        # 计算梯度方向一致性（AI 生成图像方向更随机）
        grad_angle = np.arctan2(grad_y, grad_x + 1e-8)
        angle_hist, _ = np.histogram(
            grad_angle.flatten(), bins=36, range=(-np.pi, np.pi)
        )
        angle_norm = angle_hist / (angle_hist.sum() + 1e-8)
        angle_entropy = -np.sum(angle_norm * np.log(angle_norm + 1e-8))
        angle_norm_entropy = angle_entropy / np.log(36)

        # 综合分数：熵越低越可疑
        combined_entropy = (norm_entropy + angle_norm_entropy) / 2
        score = 1.0 - combined_entropy  # 反转：低熵=高分数

        return {"score": float(score), "entropy": float(combined_entropy)}

    def _analyze_color_consistency(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        颜色一致性分析，检测 AI 生成的颜色过度平滑

        核心原理：AI 生成图像在局部区域颜色过度一致，缺乏自然噪声

        返回：{"score": 0-1, "variance_ratio": float}
        """
        # 转换为 LAB 颜色空间
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 分块计算颜色方差
        block_size = 16
        h, w = lab.shape[:2]
        variances = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = lab[i : i + block_size, j : j + block_size, :]
                # 计算 L 通道（亮度）的方差
                var_l = np.var(block[:, :, 0])
                variances.append(var_l)

        if not variances:
            return {"score": 0.0, "variance_ratio": 0.0}

        # 平均方差
        mean_variance = np.mean(variances)

        # 正常图像：方差 50-200，AI 生成：方差 10-50（过度平滑）
        # 方差越低，越可疑
        variance_ratio = mean_variance / 100.0  # 归一化

        # 反转分数：低方差=高分数
        score = 1.0 / (1.0 + np.exp((variance_ratio - 0.5) * 5))

        return {"score": float(score), "variance_ratio": float(variance_ratio)}
