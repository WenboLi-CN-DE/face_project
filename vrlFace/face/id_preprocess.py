"""
证件照预处理模块

提供 preprocess_id_photo() 函数，对证件照执行：
    1. 检测人脸（多张取面积最大者）
    2. 基于双眼关键点对齐旋转
    3. 按扩展比例裁剪人脸 ROI
    4. 可选 CLAHE 增强 + 高斯去噪

未检测到人脸或任何异常均返回原图。
"""

import logging
import math

import cv2
import numpy as np

from .config import config
from .recognizer import get_recognizer

logger = logging.getLogger(__name__)


def preprocess_id_photo(img: np.ndarray) -> np.ndarray:
    """对证件照执行人脸对齐、裁剪与增强预处理。

    Args:
        img: 输入图像，BGR 格式的 numpy 数组（单通道灰度图会自动转换）。

    Returns:
        预处理后的 BGR 图像。若未检测到人脸或发生异常，返回原图。
    """
    try:
        if img is None or img.size == 0:
            return img

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        recognizer = get_recognizer()
        faces = recognizer.get(img)

        if not faces:
            logger.warning("证件照预处理：未检测到人脸，返回原图")
            return img

        face = _select_largest_face(faces)

        aligned = _align_face(img, face)
        cropped = _crop_face_roi(aligned, face, img.shape)
        enhanced = _enhance_image(cropped)
        return enhanced

    except Exception as exc:
        logger.warning("证件照预处理异常，返回原图: %s", exc)
        return img


def _select_largest_face(faces):
    """从检测结果中选取 bbox 面积最大的人脸。"""

    def _area(face):
        bbox = face.bbox
        return max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

    return max(faces, key=_area)


def _align_face(img: np.ndarray, face) -> np.ndarray:
    """基于双眼关键点计算旋转角度并校正图像。

    Args:
        img: 原始 BGR 图像。
        face: InsightFace 人脸对象，需含 kps 属性（shape=(5,2)）。

    Returns:
        旋转后的图像；若 kps 不可用则返回原图。
    """
    try:
        if not hasattr(face, "kps") or face.kps is None:
            return img

        kps = face.kps
        if kps.shape[0] < 2:
            return img

        left_eye = kps[0]
        right_eye = kps[1]

        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle = math.degrees(math.atan2(dy, dx))

        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        return aligned

    except Exception as exc:
        logger.warning("人脸对齐失败，跳过对齐: %s", exc)
        return img


def _crop_face_roi(img: np.ndarray, face, original_shape: tuple) -> np.ndarray:
    """按 config.id_crop_expand_ratio 扩展 bbox 后裁剪人脸区域。

    Args:
        img: 对齐后的 BGR 图像。
        face: InsightFace 人脸对象，含 bbox 属性 [x1, y1, x2, y2]。
        original_shape: 原始图像的 shape（用于兜底边界裁剪）。

    Returns:
        裁剪后的人脸 ROI；若裁剪区域无效则返回原图。
    """
    try:
        bbox = face.bbox
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        expand = config.id_crop_expand_ratio
        bw = x2 - x1
        bh = y2 - y1

        x1 = x1 - bw * expand
        y1 = y1 - bh * expand
        x2 = x2 + bw * expand
        y2 = y2 + bh * expand

        h, w = img.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            logger.warning("裁剪区域无效（bbox=%s），返回原图", bbox)
            return img

        return img[y1:y2, x1:x2]

    except Exception as exc:
        logger.warning("人脸裁剪失败，返回原图: %s", exc)
        return img


def _enhance_image(img: np.ndarray) -> np.ndarray:
    """CLAHE 直方图均衡化 + 高斯模糊去噪。

    仅在 config.id_enhance_enabled=True 时执行。

    Args:
        img: 输入 BGR 图像。

    Returns:
        增强后的图像；若增强开关关闭或发生异常则返回原图。
    """
    if not config.id_enhance_enabled:
        return img

    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return denoised

    except Exception as exc:
        logger.warning("图像增强失败，返回原图: %s", exc)
        return img
