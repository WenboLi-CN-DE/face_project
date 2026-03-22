"""
静默活体检测器 — 串行双检测方案

核心逻辑：
1. UniFace MiniFASNet — 防传统物理攻击（打印/屏幕/面具）
2. 频域分析器 — 防 AI 生成图像（GAN/Diffusion 伪影）

串行流程：
  UniFace 检测 → 失败 → 返回 "traditional_spoof"
       ↓ 通过
  频域分析 → 异常 → 返回 "ai_spoof"
       ↓ 通过
  返回 "real"
"""

import logging
from typing import Dict, Any, Optional
import cv2

from uniface.detection import RetinaFace
from uniface.spoofing import MiniFASNet
from .frequency_analyzer import FrequencyAnalyzer
from .deep_detector import HeuristicDetector

logger = logging.getLogger(__name__)


class SilentLivenessDetector:
    """静默活体检测器（单例）"""

    _instance: "SilentLivenessDetector | None" = None

    def __init__(self) -> None:
        logger.info("初始化 UniFace 模型...")
        self._face_detector = RetinaFace()
        self._spoofer = MiniFASNet()
        self._frequency_analyzer = FrequencyAnalyzer.get_instance()
        self._heuristic_detector = HeuristicDetector.get_instance()
        logger.info("UniFace 模型加载完成")
        logger.info("频域分析器初始化完成")
        logger.info("启发式检测器初始化完成")

    @classmethod
    def get_instance(cls) -> "SilentLivenessDetector":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        执行静默活体检测（串行双检测）

        检测流程：
        1. UniFace MiniFASNet — 检测传统物理攻击
        2. 频域分析器 — 检测 AI 生成图像

        Args:
            image_path: 图片文件绝对路径

        Returns:
            {
                "is_liveness": 1/0,           # 1=真人，0=伪造
                "confidence": float,          # 综合置信度
                "is_face_exist": 1/0,         # 是否检测到人脸
                "face_exist_confidence": float,
                "reject_reason": str | None,  # 拒绝原因：null/"traditional_spoof"/"ai_spoof"/"no_face"
                "details": {
                    "uniface_passed": bool,   # UniFace 是否通过
                    "frequency_check_passed": bool,  # 频域分析是否通过
                    "dl_check_passed": bool,  # 深度学习检测是否通过
                }
            }
        """
        # 默认结果（检测失败）
        result: Dict[str, Any] = {
            "is_liveness": 0,
            "confidence": 0.0,
            "is_face_exist": 0,
            "face_exist_confidence": 0.0,
            "reject_reason": None,
            "details": {
                "uniface_passed": False,
                "ai_check_passed": False,
            },
        }

        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error("无法读取图片：%s", image_path)
                return result

            # ===================================================================
            # Step 1: UniFace 传统活体检测（防打印/屏幕/面具）
            # ===================================================================
            faces = self._face_detector.detect(image)

            logger.info("=== RetinaFace 检测结果 ===")
            logger.info("  检测到 %d 张人脸", len(faces))

            if not faces:
                logger.info("未检测到人脸：%s", image_path)
                result["reject_reason"] = "no_face"
                return result

            result["is_face_exist"] = 1
            face = faces[0]

            spoof_result = self._spoofer.predict(image, face.bbox)

            logger.info("=== MiniFASNet 检测结果 ===")
            logger.info("  完整返回：%s", spoof_result)
            logger.info("  is_real: %s", spoof_result.is_real)
            logger.info("  confidence: %.4f", spoof_result.confidence)

            # UniFace 检测失败 → 传统攻击
            if not spoof_result.is_real:
                logger.warning(
                    "UniFace 检测失败：传统物理攻击 path=%s confidence=%.4f",
                    image_path,
                    spoof_result.confidence,
                )
                result["is_liveness"] = 0
                result["confidence"] = round(float(spoof_result.confidence), 4)
                result["face_exist_confidence"] = round(
                    float(spoof_result.confidence), 4
                )
                result["reject_reason"] = "traditional_spoof"
                result["details"]["uniface_passed"] = False
                return result

            # UniFace 检测通过
            logger.info("UniFace 检测通过：path=%s", image_path)
            result["details"]["uniface_passed"] = True

            # ===================================================================
            # Step 2: 频域分析（防 AI 生成图像）
            # ===================================================================
            # 将 bbox numpy 数组转换为 tuple
            bbox_tuple = tuple(map(int, face.bbox))
            frequency_result = self._frequency_analyzer.analyze(image, bbox_tuple)

            logger.info("=== 频域分析结果 ===")
            logger.info("  is_ai_generated: %s", frequency_result["is_ai_generated"])
            logger.info("  anomaly_score: %.4f", frequency_result["anomaly_score"])
            logger.info("  confidence: %.4f", frequency_result["confidence"])

            # AI 检测失败 → AI 生成
            if frequency_result["is_ai_generated"]:
                logger.warning(
                    "频域分析失败：AI 生成图像 path=%s anomaly_score=%.4f",
                    image_path,
                    frequency_result["anomaly_score"],
                )
                result["is_liveness"] = 0
                result["confidence"] = round(float(frequency_result["confidence"]), 4)
                result["face_exist_confidence"] = round(
                    float(spoof_result.confidence), 4
                )
                result["reject_reason"] = "ai_spoof"
                result["details"]["frequency_check_passed"] = False
                result["details"]["dl_check_passed"] = True  # 未检测
                return result

            # AI 检测通过
            logger.info("频域分析通过：path=%s", image_path)
            result["details"]["frequency_check_passed"] = True

            # ===================================================================
            # Step 3: 启发式检测（增强 AI 检测，针对现代 Diffusion 模型）
            # ===================================================================
            heuristic_result = self._heuristic_detector.detect(image, bbox_tuple)

            logger.info("=== 启发式检测结果 ===")
            logger.info("  is_ai_generated: %s", heuristic_result["is_ai_generated"])
            logger.info("  anomaly_score: %.4f", heuristic_result["anomaly_score"])

            if heuristic_result["is_ai_generated"]:
                logger.warning(
                    "启发式检测失败：AI 生成图像 path=%s anomaly_score=%.4f",
                    image_path,
                    heuristic_result["anomaly_score"],
                )
                result["is_liveness"] = 0
                result["confidence"] = round(float(heuristic_result["confidence"]), 4)
                result["face_exist_confidence"] = round(
                    float(spoof_result.confidence), 4
                )
                result["reject_reason"] = "ai_spoof"
                result["details"]["frequency_check_passed"] = True
                result["details"]["heuristic_check_passed"] = False
                return result

            logger.info("启发式检测通过：path=%s", image_path)
            result["details"]["heuristic_check_passed"] = True

            # ===================================================================
            # 三步都通过 → 真人
            # ===================================================================
            # 综合置信度：取三者的较小值（保守策略）
            final_confidence = min(
                float(spoof_result.confidence),
                float(frequency_result["confidence"]),
                float(heuristic_result["confidence"]),
            )

            result["is_liveness"] = 1
            result["confidence"] = round(final_confidence, 4)
            result["face_exist_confidence"] = round(float(spoof_result.confidence), 4)
            result["reject_reason"] = None

            logger.info(
                "✅ 静默活体检测通过 path=%s confidence=%.4f",
                image_path,
                final_confidence,
            )

            return result

        except Exception as e:
            logger.error("静默活体检测失败：%s", str(e))
            return result
