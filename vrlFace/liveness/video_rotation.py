"""
视频旋转检测和修正模块

处理移动设备录制的旋转视频
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_rotation(video_path: str) -> int:
    """
    检测视频的旋转角度

    参数:
        video_path: 视频文件路径

    返回:
        旋转角度 (0, 90, 180, 270)
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.warning(f"无法打开视频检测旋转: {video_path}")
        return 0

    # 尝试从元数据获取旋转信息
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    cap.release()

    # OpenCV 返回的旋转角度
    if rotation in [0, 90, 180, 270]:
        logger.info(f"检测到视频旋转: {rotation}度")
        return int(rotation)

    logger.info("视频无旋转信息")
    return 0


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    根据旋转角度修正帧

    参数:
        frame: 原始帧
        rotation: 旋转角度 (0, 90, 180, 270)

    返回:
        修正后的帧
    """
    if rotation == 0:
        return frame

    if rotation == 90:
        # 顺时针旋转 90 度
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        # 旋转 180 度
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        # 逆时针旋转 90 度（等于顺时针 270 度）
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        logger.warning(f"不支持的旋转角度: {rotation}")
        return frame


def get_rotated_dimensions(width: int, height: int, rotation: int) -> Tuple[int, int]:
    """
    计算旋转后的分辨率

    参数:
        width: 原始宽度
        height: 原始高度
        rotation: 旋转角度

    返回:
        (新宽度, 新高度)
    """
    if rotation in [90, 270]:
        # 90 度或 270 度旋转会交换宽高
        return (height, width)
    else:
        return (width, height)


def auto_detect_rotation_from_face(frame: np.ndarray) -> int:
    """
    通过人脸检测自动判断视频旋转角度

    当元数据不可用时，尝试在不同旋转角度下检测人脸，
    选择检测到最大人脸的角度作为正确方向。

    参数:
        frame: 视频帧

    返回:
        推荐的旋转角度 (0, 90, 180, 270)
    """
    try:
        # 使用 OpenCV 的人脸检测器
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        best_rotation = 0
        max_face_size = 0

        for rotation in [0, 90, 180, 270]:
            rotated = rotate_frame(frame, rotation)
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # 找到最大的人脸
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                face_size = largest_face[2] * largest_face[3]

                if face_size > max_face_size:
                    max_face_size = face_size
                    best_rotation = rotation

        if max_face_size > 0:
            logger.info(f"通过人脸检测推断旋转角度: {best_rotation}度")
            return best_rotation
        else:
            logger.warning("未检测到人脸，无法自动判断旋转")
            return 0

    except Exception as e:
        logger.error(f"自动检测旋转失败: {e}")
        return 0


class RotationHandler:
    """视频旋转处理器"""

    def __init__(self, video_path: str, auto_detect: bool = True):
        """
        初始化旋转处理器

        参数:
            video_path: 视频路径
            auto_detect: 当元数据不可用时，是否启用基于人脸的自动检测（默认 True）
        """
        self.video_path = video_path
        self.rotation = detect_rotation(video_path)

        # 如果元数据没有旋转信息且启用自动检测
        if self.rotation == 0 and auto_detect:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            ret, frame = cap.read()
            cap.release()

            if ret:
                detected = auto_detect_rotation_from_face(frame)
                if detected != 0:
                    logger.info(
                        f"元数据无旋转信息，通过人脸检测推断需要旋转 {detected}度"
                    )
                    self.rotation = detected

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理帧（应用旋转修正）

        参数:
            frame: 原始帧

        返回:
            修正后的帧
        """
        return rotate_frame(frame, self.rotation)

    def get_rotation(self) -> int:
        """获取旋转角度"""
        return self.rotation

    def needs_rotation(self) -> bool:
        """是否需要旋转"""
        return self.rotation != 0
