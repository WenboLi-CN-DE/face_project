"""
静默活体检测器 — DeepFaceAntiSpoofing 封装

核心逻辑：三信号融合
1. analyze_image() → 人脸检测 + Deepfake 检测（容忍度高）
2. analyze_deepface() → 展示攻击检测（严格，防打印/翻拍）
3. FFT 频域分析 → 莫尔纹检测（屏幕翻拍特征）
"""

import logging
from typing import Dict, Any
import cv2
import numpy as np

from deepface_antispoofing import DeepFaceAntiSpoofing

logger = logging.getLogger(__name__)


class SilentLivenessDetector:
    """静默活体检测器（单例）"""

    _instance: "SilentLivenessDetector | None" = None

    def __init__(self) -> None:
        logger.info("初始化 DeepFaceAntiSpoofing 模型...")
        self._analyzer = DeepFaceAntiSpoofing()
        logger.info("DeepFaceAntiSpoofing 模型加载完成")

    @classmethod
    def get_instance(cls) -> "SilentLivenessDetector":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _detect_moire_fft(self, image_path: str) -> float:
        """
        FFT 频域检测莫尔纹（改进版：检测周期性峰值）

        Returns:
            moire_score: 莫尔纹分数，越高越可能是翻拍（>0.15 疑似翻拍）
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0

            # 缩放到固定尺寸以标准化
            img = cv2.resize(img, (256, 256))

            # FFT 变换
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            # 去除中心低频（DC 分量）
            h, w = magnitude.shape
            center_mask = np.ones((h, w), dtype=bool)
            center_mask[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10] = False
            magnitude_filtered = magnitude * center_mask

            # 检测周期性峰值：莫尔纹会在特定频率产生明显峰值
            # 计算频谱的峰值突出度
            mean_mag = np.mean(magnitude_filtered)
            max_mag = np.max(magnitude_filtered)

            # 峰值突出度：最大值与均值的比值
            peak_prominence = (max_mag - mean_mag) / (mean_mag + 1e-6)

            # 归一化到 0-1 范围
            moire_score = min(peak_prominence / 100.0, 1.0)

            return float(moire_score)
        except Exception as e:
            logger.warning("莫尔纹检测失败: %s", str(e))
            return 0.0

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        执行静默活体检测（双模型投票）

        Args:
            image_path: 图片文件绝对路径

        Returns:
            {
                "is_liveness": 1/0,
                "confidence": float,
                "is_face_exist": 1/0,
                "face_exist_confidence": float,
            }
        """
        # 默认结果（检测失败）
        result: Dict[str, Any] = {
            "is_liveness": 0,
            "confidence": 0.0,
            "is_face_exist": 0,
            "face_exist_confidence": 0.0,
        }

        # Step 1: analyze_image（容忍度高）
        try:
            image_result = self._analyzer.analyze_image(image_path)
        except Exception as e:
            logger.error("analyze_image 失败: %s", str(e))
            return result

        # 检查是否检测到人脸
        if not image_result or "age" not in image_result:
            logger.info("未检测到人脸: %s", image_path)
            return result

        spoof_info = image_result.get("spoof", {})
        image_real_prob = float(spoof_info.get("Real", 0.0))
        result["is_face_exist"] = 1
        result["face_exist_confidence"] = round(image_real_prob, 4)

        logger.info("=== analyze_image 结果 ===")
        logger.info("  完整返回: %s", image_result)
        logger.info("  spoof 分析: %s", spoof_info)
        logger.info("  Real 概率: %.4f", image_real_prob)

        # Step 2: analyze_deepface（严格，防打印/翻拍）
        try:
            deepface_result = self._analyzer.analyze_deepface(image_path)
            dominant_printed = deepface_result.get("dominant_printed", "Printed")
            printed_analysis = deepface_result.get("printed_analysis", {})
            deepface_real_prob = float(printed_analysis.get("Real", 0.0))
            deepface_vote = dominant_printed == "Real"

            logger.info("=== analyze_deepface 结果 ===")
            logger.info("  完整返回: %s", deepface_result)
            logger.info("  printed_analysis: %s", printed_analysis)
            logger.info("  dominant_printed: %s", dominant_printed)
            logger.info("  Real 概率: %.4f", deepface_real_prob)
        except Exception as e:
            logger.warning("analyze_deepface 失败，仅用 analyze_image 兜底: %s", str(e))
            # 兜底：仅用 analyze_image
            result["is_liveness"] = 1 if image_real_prob > 0.5 else 0
            result["confidence"] = round(image_real_prob, 4)
            return result

        # Step 3: FFT 莫尔纹检测
        moire_score = self._detect_moire_fft(image_path)
        has_moire = moire_score > 0.15  # 阈值：>0.15 疑似翻拍（改进后的阈值）

        logger.info("=== FFT 莫尔纹检测 ===")
        logger.info("  莫尔纹分数: %.4f", moire_score)
        logger.info("  是否检测到莫尔纹: %s", has_moire)

        # Step 4: 三信号融合策略
        image_vote = image_real_prob > 0.5
        deepface_vote = deepface_real_prob > 0.5

        # 莫尔纹强制拒绝：如果检测到明显莫尔纹，直接判定为翻拍
        if has_moire and image_real_prob > 0.9:
            is_liveness = 0
            final_confidence = 1.0 - (moire_score / 10.0)  # 转换为置信度
            logger.warning(
                "⚠️  莫尔纹强制拒绝：moire_score=%.4f，判定为翻拍",
                moire_score,
            )
        else:
            # 检测模型严重冲突
            severe_conflict = (
                image_real_prob >= 0.6 and deepface_real_prob < 0.15
            ) or (deepface_real_prob >= 0.6 and image_real_prob < 0.15)

            if severe_conflict:
                # 严重冲突时：降低不可靠模型的权重
                if image_real_prob >= 0.6 and deepface_real_prob < 0.15:
                    # image 确信真 + deepface 强烈认为假 → 可能是 deepface 误判
                    # 降低 deepface 权重
                    final_confidence = 0.7 * image_real_prob + 0.3 * deepface_real_prob
                    # 宽松阈值
                    is_liveness = (
                        1
                        if (
                            final_confidence > 0.45
                            or (image_real_prob > 0.75 and deepface_real_prob > 0.05)
                        )
                        else 0
                    )
                    logger.warning(
                        "⚠️  冲突：image 确信真(%.4f) vs deepface 强烈假(%.4f)，降低 deepface 权重",
                        image_real_prob,
                        deepface_real_prob,
                    )
                else:
                    # deepface 确信真 + image 强烈认为假 → 罕见情况，保守拒绝
                    final_confidence = 0.5 * image_real_prob + 0.5 * deepface_real_prob
                    is_liveness = 1 if final_confidence > 0.6 else 0
                    logger.warning(
                        "⚠️  冲突：deepface 确信真(%.4f) vs image 强烈假(%.4f)，保守判断",
                        deepface_real_prob,
                        image_real_prob,
                    )
            else:
                # 无严重冲突：标准加权融合
                final_confidence = 0.4 * image_real_prob + 0.6 * deepface_real_prob
                is_liveness = (
                    1
                    if (
                        final_confidence > 0.55
                        or (image_real_prob > 0.85 and deepface_real_prob > 0.2)
                        or (image_real_prob > 0.65 and deepface_real_prob > 0.5)
                    )
                    else 0
                )

        result["is_liveness"] = is_liveness
        result["confidence"] = round(final_confidence, 4)

        logger.info("=== 最终判定 ===")
        logger.info("  文件: %s", image_path)
        logger.info("  image_vote: %s (Real=%.4f)", image_vote, image_real_prob)
        logger.info(
            "  deepface_vote: %s (Real=%.4f)", deepface_vote, deepface_real_prob
        )
        logger.info("  is_liveness: %d", is_liveness)
        logger.info("  confidence: %.4f", final_confidence)

        return result
