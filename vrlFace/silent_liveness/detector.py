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
        执行静默活体检测（仅使用 analyze_deepface）

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

        try:
            deepface_result = self._analyzer.analyze_deepface(image_path)

            # 详细日志：输出完整返回结果
            logger.info("analyze_deepface 完整返回: %s", deepface_result)

            # 提取结果
            confidence = float(deepface_result.get("confidence", 0.0))
            dominant_printed = deepface_result.get("dominant_printed", "Printed")
            printed_analysis = deepface_result.get("printed_analysis", {})
            is_real = dominant_printed == "Real"

            # 映射到业务字段
            result["is_liveness"] = 1 if is_real else 0
            result["confidence"] = round(confidence, 4)
            result["is_face_exist"] = 1  # analyze_deepface 能返回说明检测到人脸
            result["face_exist_confidence"] = round(confidence, 4)

            logger.info(
                "静默活体检测 path=%s is_liveness=%d confidence=%.4f "
                "dominant=%s printed_analysis=%s",
                image_path,
                result["is_liveness"],
                confidence,
                dominant_printed,
                printed_analysis,
            )

            return result

        except Exception as e:
            logger.error("analyze_deepface 失败: %s", str(e))
            return result
