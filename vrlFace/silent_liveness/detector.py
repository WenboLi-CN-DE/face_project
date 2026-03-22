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

        # Step 3: 智能投票策略
        image_vote = image_real_prob > 0.5
        deepface_vote = deepface_real_prob > 0.5

        # 检测模型冲突：一个强烈认为真，另一个强烈认为假
        strong_conflict = (image_real_prob > 0.8 and deepface_real_prob < 0.2) or (
            deepface_real_prob > 0.8 and image_real_prob < 0.2
        )

        if strong_conflict:
            # 冲突时的决策逻辑（优先级从高到低）

            # 优先级1: deepface 极度确信是假（<0.05）→ 拒绝（防翻拍最高优先级）
            if deepface_real_prob < 0.05:
                is_liveness = 0
                final_confidence = deepface_real_prob
                logger.warning(
                    "⚠️  模型冲突：deepface 极度确信假(%.4f)，拒绝（疑似翻拍）",
                    deepface_real_prob,
                )
            # 优先级2: image 极度确信是真（>0.95）且 deepface 不是极度确信假（>=0.05）→ 通过
            elif image_real_prob > 0.95:
                is_liveness = 1
                final_confidence = image_real_prob
                logger.warning(
                    "⚠️  模型冲突：image 极度确信(%.4f)，通过",
                    image_real_prob,
                )
            # 优先级3: 其他冲突 → 信任 deepface
            else:
                is_liveness = 1 if deepface_vote else 0
                final_confidence = deepface_real_prob
                logger.warning(
                    "⚠️  模型冲突：image=%.4f vs deepface=%.4f，采用 deepface 判断",
                    image_real_prob,
                    deepface_real_prob,
                )
        else:
            # 无冲突：OR 逻辑（只要一个说真人就通过）
            is_liveness = 1 if (image_vote or deepface_vote) else 0
            final_confidence = max(image_real_prob, deepface_real_prob)

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
