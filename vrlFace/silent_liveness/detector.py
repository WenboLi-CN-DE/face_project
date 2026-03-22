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
        is_real_str = str(deepface_result.get("is_real", "False"))
        is_real = is_real_str.lower() == "true"

        # Step 3: 融合判定
        # 综合 deepfake 检测 (real_prob) 和展示攻击检测 (anti_spoof_confidence)
        # 展示攻击检测权重更高（更直接的防伪信号）
        final_confidence = 0.4 * real_prob + 0.6 * anti_spoof_confidence
        is_liveness = 1 if (is_real and final_confidence > 0.5) else 0

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
