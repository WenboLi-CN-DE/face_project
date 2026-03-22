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
                "success": "True"/"False",
                "is_real": "True"/"False",
                "confidence": float,
                "spoof_type": str,
                "processing_time": float,
            }
        """
        import time

        start_time = time.time()

        try:
            deepface_result = self._analyzer.analyze_deepface(image_path)
            processing_time = round(time.time() - start_time, 2)

            # 提取结果
            confidence = float(deepface_result.get("confidence", 0.0))
            dominant_printed = deepface_result.get("dominant_printed", "Printed")
            is_real = dominant_printed == "Real"

            result = {
                "success": "True",
                "is_real": "True" if is_real else "False",
                "confidence": round(confidence, 2),
                "spoof_type": "Real Face" if is_real else "Spoof",
                "processing_time": processing_time,
            }

            logger.info(
                "静默活体检测 path=%s is_real=%s confidence=%.2f time=%.2fs",
                image_path,
                result["is_real"],
                confidence,
                processing_time,
            )

            return result

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            logger.error("analyze_deepface 失败: %s", str(e))
            return {
                "success": "False",
                "is_real": "False",
                "confidence": 0.0,
                "spoof_type": "Error",
                "processing_time": processing_time,
            }
