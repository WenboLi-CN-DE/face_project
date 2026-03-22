"""
静默活体检测器 — DeepFaceAntiSpoofing 封装

核心逻辑：双模型投票
1. analyze_image() → 人脸检测 + Deepfake 检测（容忍度高）
2. analyze_deepface() → 展示攻击检测（严格，防打印/翻拍）
3. 两个模型都判断为真人才通过
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

        # Step 2: analyze_deepface（严格，防打印/翻拍）
        try:
            deepface_result = self._analyzer.analyze_deepface(image_path)
            dominant_printed = deepface_result.get("dominant_printed", "Printed")
            printed_analysis = deepface_result.get("printed_analysis", {})
            deepface_real_prob = float(printed_analysis.get("Real", 0.0))
            deepface_vote = dominant_printed == "Real"
        except Exception as e:
            logger.warning("analyze_deepface 失败，仅用 analyze_image 兜底: %s", str(e))
            # 兜底：仅用 analyze_image
            result["is_liveness"] = 1 if image_real_prob > 0.5 else 0
            result["confidence"] = round(image_real_prob, 4)
            return result

        # Step 3: 双模型投票
        image_vote = image_real_prob > 0.5

        # 取两者中较低的置信度作为最终置信度（保守策略）
        final_confidence = min(image_real_prob, deepface_real_prob)

        # 两个模型都认为是真人才通过
        is_liveness = 1 if (image_vote and deepface_vote) else 0

        result["is_liveness"] = is_liveness
        result["confidence"] = round(final_confidence, 4)

        logger.info(
            "静默活体检测 path=%s is_liveness=%d confidence=%.4f "
            "image_vote=%s(%.4f) deepface_vote=%s(%.4f)",
            image_path,
            is_liveness,
            final_confidence,
            image_vote,
            image_real_prob,
            deepface_vote,
            deepface_real_prob,
        )

        return result
