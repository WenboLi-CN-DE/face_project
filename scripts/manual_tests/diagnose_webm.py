"""
WebM 视频处理诊断脚本

用于诊断 WebM 视频是否能被正确读取和处理
"""

import cv2
import sys
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def diagnose_video(video_path: str):
    """诊断视频文件"""
    logger.info(f"开始诊断视频: {video_path}")

    # 1. 检查文件是否存在
    import os

    if not os.path.exists(video_path):
        logger.error(f"文件不存在: {video_path}")
        return False

    file_size = os.path.getsize(video_path)
    logger.info(f"文件大小: {file_size / 1024:.2f} KB")

    # 2. 尝试打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("无法打开视频文件")
        return False

    logger.info("✓ 视频文件成功打开")

    # 3. 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"视频属性:")
    logger.info(f"  - FPS: {fps}")
    logger.info(f"  - 总帧数: {total_frames}")
    logger.info(f"  - 分辨率: {width}x{height}")
    logger.info(f"  - 时长: {duration:.2f} 秒")

    # 4. 尝试读取前几帧
    logger.info("尝试读取前 10 帧...")
    success_count = 0
    for i in range(min(10, total_frames)):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            if i == 0:
                logger.info(f"  第 1 帧: shape={frame.shape}, dtype={frame.dtype}")
        else:
            logger.warning(f"  第 {i + 1} 帧读取失败")
            break

    logger.info(f"成功读取 {success_count}/10 帧")

    cap.release()

    # 5. 测试 MediaPipe 处理
    logger.info("\n测试 MediaPipe 人脸检测...")
    try:
        from vrlFace.liveness.mediapipe_detector import MediaPipeLivenessDetector

        detector = MediaPipeLivenessDetector(max_faces=1)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("无法读取第一帧")
            return False

        logger.info(f"第一帧 shape: {frame.shape}")

        lm_data = detector.extract_landmarks(frame)

        if lm_data is None:
            logger.warning("⚠ MediaPipe 未检测到人脸")
        else:
            logger.info("✓ MediaPipe 成功检测到人脸")
            logger.info(f"  - 质量分数: {lm_data.get('quality_score', 0):.4f}")
            logger.info(f"  - 关键点数量: {len(lm_data.get('landmarks', []))}")

        detector.close()

    except Exception as e:
        logger.error(f"MediaPipe 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    logger.info("\n✓ 诊断完成")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_webm.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    success = diagnose_video(video_path)

    sys.exit(0 if success else 1)
