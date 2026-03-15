"""
人脸质量评估子包

包含基于 InsightFace 的人脸质量检测器。
注意：InsightFaceQualityDetector 虽依赖 InsightFace（人脸域），
但其职责是为活体检测提供质量评分，故归属于 liveness/quality。
"""

from ..insightface_quality import InsightFaceQualityDetector

__all__ = ["InsightFaceQualityDetector"]

