"""
静默活体检测器 — UniFace MiniFASNet 封装

核心逻辑：
使用 UniFace 的 MiniFASNet 模型进行活体检测
"""

import logging
from typing import Dict, Any
import cv2

from uniface.detection import RetinaFace
from uniface.spoofing import MiniFASNet

logger = logging.getLogger(__name__)


class SilentLivenessDetector:
    """静默活体检测器（单例）"""

    _instance: "SilentLivenessDetector | None" = None

    def __init__(self) -> None:
        logger.info("初始化 UniFace 模型...")
        self._face_detector = RetinaFace()
        self._spoofer = MiniFASNet()
        logger.info("UniFace 模型加载完成")

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

            # Step 1: 人脸检测
            faces = self._face_detector.detect(image)

            logger.info("=== RetinaFace 检测结果 ===")
            logger.info("  检测到 %d 张人脸", len(faces))

            if not faces:
                logger.info("未检测到人脸: %s", image_path)
                return result

            result["is_face_exist"] = 1

            # Step 2: 对第一张人脸进行活体检测
            face = faces[0]
            spoof_result = self._spoofer.predict(image, face.bbox)

            logger.info("=== MiniFASNet 检测结果 ===")
            logger.info("  完整返回: %s", spoof_result)
            logger.info("  is_real: %s", spoof_result.is_real)
            logger.info("  confidence: %.4f", spoof_result.confidence)

            # 映射到业务字段
            result["is_liveness"] = 1 if spoof_result.is_real else 0
            result["confidence"] = round(float(spoof_result.confidence), 4)
            result["face_exist_confidence"] = round(float(spoof_result.confidence), 4)

            logger.info(
                "静默活体检测 path=%s is_liveness=%d confidence=%.4f",
                image_path,
                result["is_liveness"],
                result["confidence"],
            )

            return result

        except Exception as e:
            logger.error("UniFace 检测失败: %s", str(e))
            return result
