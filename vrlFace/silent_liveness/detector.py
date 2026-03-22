"""
静默活体检测器 — UniFace MiniFASNet 封装

核心逻辑：
使用 UniFace 的 MiniFASNet 模型进行活体检测
"""

import logging
from typing import Dict, Any
import cv2

from uniface import AntiSpoofing

logger = logging.getLogger(__name__)


class SilentLivenessDetector:
    """静默活体检测器（单例）"""

    _instance: "SilentLivenessDetector | None" = None

    def __init__(self) -> None:
        logger.info("初始化 UniFace AntiSpoofing 模型...")
        self._detector = AntiSpoofing(model_name="minifasnet_v2")
        logger.info("UniFace AntiSpoofing 模型加载完成")

    @classmethod
    def get_instance(cls) -> "SilentLivenessDetector":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        执行静默活体检测（使用 UniFace MiniFASNet）

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
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error("无法读取图片: %s", image_path)
                return result

            # UniFace 检测
            uniface_result = self._detector.predict(image)

            logger.info("=== UniFace 检测结果 ===")
            logger.info("  完整返回: %s", uniface_result)

            # 提取结果
            is_real = uniface_result.get("is_real", False)
            confidence = float(uniface_result.get("confidence", 0.0))

            # 映射到业务字段
            result["is_liveness"] = 1 if is_real else 0
            result["confidence"] = round(confidence, 4)
            result["is_face_exist"] = 1  # UniFace 能返回说明检测到人脸
            result["face_exist_confidence"] = round(confidence, 4)

            logger.info(
                "静默活体检测 path=%s is_liveness=%d confidence=%.4f",
                image_path,
                result["is_liveness"],
                confidence,
            )

            return result

        except Exception as e:
            logger.error("UniFace 检测失败: %s", str(e))
            return result
