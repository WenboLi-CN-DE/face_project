"""
静默活体检测器 — DeepFaceAntiSpoofing 封装

核心逻辑：
1. analyze_deepface() → 展示攻击检测（打印照片/屏幕翻拍）
2. analyze_image() → 人脸检测 + Deepfake 检测
3. 融合两个信号输出最终活体判定
"""

import logging
from typing import Dict, Any

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

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        执行静默活体检测

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
        # 默认结果（无人脸）
        result: Dict[str, Any] = {
            "is_liveness": 0,
            "confidence": 0.0,
            "is_face_exist": 0,
            "face_exist_confidence": 0.0,
        }

        # Step 1: analyze_image — 获取人脸检测 + deepfake 判定
        try:
            image_result = self._analyzer.analyze_image(image_path)
        except Exception as e:
            logger.warning("analyze_image 失败: %s", str(e))
            return result

        # 检查是否检测到人脸（如果返回了 age/gender，说明有人脸）
        if not image_result or "age" not in image_result:
            logger.info("未检测到人脸: %s", image_path)
            return result

        # 人脸存在
        spoof_info = image_result.get("spoof", {})
        real_prob = float(spoof_info.get("Real", 0.0))
        result["is_face_exist"] = 1
        result["face_exist_confidence"] = round(real_prob, 4)

        # Step 2: analyze_deepface — 展示攻击检测（打印/翻拍）
        try:
            deepface_result = self._analyzer.analyze_deepface(image_path)
        except Exception as e:
            logger.warning("analyze_deepface 失败: %s", str(e))
            # 仅用 image 结果兜底
            result["confidence"] = round(real_prob, 4)
            result["is_liveness"] = 1 if real_prob > 0.5 else 0
            return result

        # 提取防伪置信度
        anti_spoof_confidence = float(deepface_result.get("confidence", 0.0))
        dominant_printed = deepface_result.get("dominant_printed", "Printed")
        printed_analysis = deepface_result.get("printed_analysis", {})
        printed_real_prob = float(printed_analysis.get("Real", 0.0))

        logger.info(
            "analyze_deepface 原始返回: %s | analyze_image spoof: %s",
            deepface_result,
            spoof_info,
        )

        # 加权融合置信度
        # 问题：当 printed 模型强烈误判时（printed_real_prob 极低），
        # 0.6 的权重会导致融合分数被拉低，保护条件也难以触发
        # 解决：冲突时降低 printed 权重，并放宽保护条件

        # 检测模型冲突：analyze_image 认为是真人，但 printed 强烈认为是 Spoof
        model_conflict = real_prob >= 0.6 and printed_real_prob < 0.15

        if model_conflict:
            # 冲突时降低 printed 权重到 0.3，更相信 analyze_image
            final_confidence = 0.7 * real_prob + 0.3 * printed_real_prob
            # 冲突时使用更宽松的阈值
            is_liveness = (
                1
                if (
                    final_confidence > 0.45
                    or (real_prob > 0.75 and printed_real_prob > 0.05)
                )
                else 0
            )
        else:
            # 正常情况：标准加权融合
            final_confidence = 0.4 * real_prob + 0.6 * printed_real_prob
            is_liveness = (
                1
                if (
                    final_confidence > 0.55
                    or (real_prob > 0.85 and printed_real_prob > 0.2)
                    or (real_prob > 0.65 and printed_real_prob > 0.5)
                )
                else 0
            )

        result["is_liveness"] = is_liveness
        result["confidence"] = round(final_confidence, 4)
        result["face_exist_confidence"] = round(
            max(real_prob, anti_spoof_confidence), 4
        )

        logger.info(
            "静默活体检测 path=%s is_liveness=%d confidence=%.4f "
            "deepfake_real=%.4f anti_spoof=%.4f",
            image_path,
            is_liveness,
            final_confidence,
            real_prob,
            anti_spoof_confidence,
        )

        return result
