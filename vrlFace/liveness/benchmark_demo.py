"""
基准帧校准功能演示脚本

功能：
- 读取视频文件
- 实时显示基准帧采集状态
- 显示每帧的 embedding 相似度
- 显示活体检测结果

使用方式:
    uv run python -m vrlFace.liveness.benchmark_demo --video path/to/video.mp4
    uv run python -m vrlFace.liveness.benchmark_demo --camera 0  # 摄像头模式
"""

import cv2
import argparse
import time
import numpy as np
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from .config import LivenessConfig
from .fusion_engine import LivenessFusionEngine
from .benchmark_calibrator import BenchmarkCalibrator, BenchmarkConfig
from .fast_detector import FastLivenessDetector
from .utils import build_fast_detector_config, resolve_current_action


class BenchmarkVisualizer:
    """基准帧可视化器"""

    def __init__(self):
        self.font = self._load_font()

    def _load_font(self):
        """加载中文字体"""
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, 18)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    def draw_benchmark_info(
        self, frame: np.ndarray, calibrator: BenchmarkCalibrator
    ) -> np.ndarray:
        """在帧上绘制基准帧信息"""
        h, w = frame.shape[:2]

        # 获取基准状态
        status = calibrator.get_status()

        # 创建 PIL 图像用于中文渲染
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 背景面板
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 380, 120

        # 绘制半透明背景
        overlay = np.zeros((panel_h, panel_w, 4), dtype=np.uint8)
        overlay[:, :] = (0, 0, 0, 180)

        # 绘制基准状态
        y_offset = 25
        status_text = f"基准状态：{status.get('status', 'UNKNOWN')}"
        draw.text((15, y_offset), status_text, font=self.font, fill=(255, 255, 255))

        if status.get("status") == "COLLECTING":
            # 采集中
            elapsed = status.get("elapsed", 0)
            duration = status.get("duration", 2.0)
            frames = status.get("frames_collected", 0)
            min_req = status.get("min_required", 3)

            progress = min(elapsed / duration, 1.0)
            bar_w = int(progress * 300)

            draw.text(
                (15, y_offset + 25),
                f"采集进度：{elapsed:.1f}s / {duration:.1f}s",
                font=self.font,
                fill=(200, 200, 0),
            )
            draw.text(
                (15, y_offset + 50),
                f"已收集：{frames} 帧 (最少 {min_req} 帧)",
                font=self.font,
                fill=(200, 200, 0),
            )

            # 进度条
            draw.rectangle(
                [(15, y_offset + 80), (315, y_offset + 90)], fill=(50, 50, 50)
            )
            draw.rectangle(
                [(15, y_offset + 80), (15 + bar_w, y_offset + 90)], fill=(0, 255, 0)
            )

        elif status.get("status") == "READY":
            # 已就绪
            quality = status.get("benchmark_quality", 0)
            pitch = status.get("benchmark_pitch", 0)
            yaw = status.get("benchmark_yaw", 0)
            verified = status.get("verification_count", 0)

            draw.text(
                (15, y_offset + 25),
                f"基准质量：{quality:.3f}",
                font=self.font,
                fill=(0, 255, 0),
            )
            draw.text(
                (15, y_offset + 50),
                f"基准角度：pitch={pitch:.1f}°, yaw={yaw:.1f}°",
                font=self.font,
                fill=(0, 255, 0),
            )
            draw.text(
                (15, y_offset + 75),
                f"已验证：{verified} 帧",
                font=self.font,
                fill=(0, 255, 0),
            )

            # 验证历史状态
            if calibrator.verification_history:
                recent = list(calibrator.verification_history)[-5:]
                all_verified = all(v.get("is_same_person", False) for v in recent)

                if all_verified:
                    draw.text(
                        (15, y_offset + 100),
                        "✅ 身份验证通过",
                        font=self.font,
                        fill=(0, 255, 0),
                    )
                else:
                    draw.text(
                        (15, y_offset + 100),
                        "⚠️ 检测到身份不匹配",
                        font=self.font,
                        fill=(255, 0, 0),
                    )

        # 转换回 OpenCV 格式
        frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        return frame

    def draw_verification_details(
        self, frame: np.ndarray, verification: dict
    ) -> np.ndarray:
        """绘制验证详细信息"""
        h, w = frame.shape[:2]

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 右侧面板
        panel_x = w - 220
        panel_y = 10

        # 显示 embedding 相似度
        emb_sim = verification.get("embedding_similarity", 0)
        lm_sim = verification.get("landmark_similarity", 0)
        is_same = verification.get("is_same_person", False)

        # 背景
        draw.rectangle(
            [(panel_x, panel_y), (w - 10, panel_y + 100)],
            fill=(0, 0, 0, 180),
            outline=(100, 100, 100),
        )

        # 标题
        draw.text(
            (panel_x + 10, panel_y + 5),
            "身份验证",
            font=self.font,
            fill=(255, 255, 255),
        )

        # Embedding 相似度
        emb_color = (0, 255, 0) if emb_sim > 0.3 else (255, 0, 0)
        draw.text(
            (panel_x + 10, panel_y + 30),
            f"Embedding: {emb_sim:.4f}",
            font=self.font,
            fill=emb_color,
        )

        # 相似度条
        bar_w = int(emb_sim * 180)
        draw.rectangle(
            [(panel_x + 10, panel_y + 55), (panel_x + 190, panel_y + 62)],
            fill=(50, 50, 50),
        )
        draw.rectangle(
            [(panel_x + 10, panel_y + 55), (panel_x + 10 + bar_w, panel_y + 62)],
            fill=emb_color,
        )

        # Landmarks 相似度
        lm_color = (0, 255, 0) if lm_sim > 0.5 else (255, 0, 0)
        draw.text(
            (panel_x + 10, panel_y + 70),
            f"Landmarks: {lm_sim:.4f}",
            font=self.font,
            fill=lm_color,
        )

        # 判定结果
        result_text = "✅ 同一人" if is_same else "❌ 不同人"
        result_color = (0, 255, 0) if is_same else (255, 0, 0)
        draw.text(
            (panel_x + 10, panel_y + 95), result_text, font=self.font, fill=result_color
        )

        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


def run_video_benchmark(video_path: str, enable_benchmark: bool = True):
    """视频文件基准帧测试"""
    print("=" * 60)
    print("基准帧校准功能 - 视频测试")
    print("=" * 60)

    # 初始化配置
    config = LivenessConfig.video_fast_config()
    config.enable_benchmark = enable_benchmark

    # 初始化引擎
    engine = LivenessFusionEngine(config)
    fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

    # 初始化可视化器
    visualizer = BenchmarkVisualizer()

    # 打开视频
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"视频信息：{total_frames} 帧 @ {fps:.1f}fps, 时长={duration:.2f}秒")
    print("按 'ESC' 退出，'R' 重置基准，'B' 开关基准功能\n")

    frame_idx = 0
    last_verification = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n视频播放完成")
            break

        frame_idx += 1

        # 推理
        lm_data = engine.mp_detector.extract_landmarks(frame)

        if lm_data is None:
            result = {
                "is_live": False,
                "score": 0.0,
                "quality_score": 0.0,
                "reason": "NO_FACE_DETECTED",
            }
        else:
            landmarks = lm_data["landmarks"]
            quality_score = lm_data["quality_score"]

            fd_result = fast_detector.detect_liveness(
                landmarks, lm_data.get("frame_shape", frame.shape)
            )

            motion_score = fd_result["score"]
            engine.score_history.append(motion_score)
            smoothed = float(
                sum(list(engine.score_history)[-config.smooth_window :])
                / min(len(engine.score_history), config.smooth_window)
            )

            is_live = smoothed > config.threshold

            result = {
                "is_live": is_live,
                "score": smoothed,
                "quality_score": quality_score,
                "reason": "LIVE" if is_live else "SPOOF",
            }

            # 基准帧校准
            if enable_benchmark and engine.calibrator is not None:
                embedding = lm_data.get("embedding")
                pitch = fd_result.get("pitch", 0.0)
                yaw = fd_result.get("yaw", 0.0)
                face_bbox = lm_data.get("face_bbox")

                if embedding is not None and face_bbox is not None:
                    if engine.calibrator.is_collecting_benchmark():
                        engine.calibrator.add_candidate_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            quality_score=quality_score,
                            face_bbox=face_bbox,
                            pitch=pitch,
                            yaw=yaw,
                            frame_index=frame_idx,
                        )
                        last_verification = {"status": "collecting"}
                    else:
                        verification = engine.calibrator.verify_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            pitch=pitch,
                            yaw=yaw,
                        )
                        last_verification = verification

        # 绘制 UI
        if enable_benchmark and engine.calibrator is not None:
            frame = visualizer.draw_benchmark_info(frame, engine.calibrator)

            if (
                last_verification
                and last_verification.get("embedding_similarity") is not None
            ):
                frame = visualizer.draw_verification_details(frame, last_verification)

        # 显示活体状态
        status_color = (0, 255, 0) if result.get("is_live", False) else (0, 0, 255)
        cv2.putText(
            frame,
            f"{result.get('reason', 'UNKNOWN')} score={result.get('score', 0):.2f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        # 显示帧信息
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{total_frames}",
            (frame.shape[1] - 200, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # 显示窗口
        cv2.imshow("Benchmark Demo", frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n用户退出")
            break
        elif key == ord("r") or key == ord("R"):
            engine.reset()
            last_verification = {}
            print("\n重置基准")
        elif key == ord("b") or key == ord("B"):
            enable_benchmark = not enable_benchmark
            print(f"\n基准功能：{'ON' if enable_benchmark else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    engine.close()

    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    if engine.calibrator is not None:
        status = engine.calibrator.get_status()
        print(f"基准状态：{status.get('status', 'UNKNOWN')}")
        if status.get("status") == "READY":
            print(f"基准质量：{status.get('benchmark_quality', 0):.3f}")
            print(f"验证帧数：{status.get('verification_count', 0)}")
    print("=" * 60)


def run_camera_benchmark(camera_id: int = 0, enable_benchmark: bool = True):
    """摄像头实时基准帧测试"""
    print("=" * 60)
    print("基准帧校准功能 - 摄像头测试")
    print("=" * 60)

    config = LivenessConfig.realtime_config()
    config.enable_benchmark = enable_benchmark

    engine = LivenessFusionEngine(config)
    fast_detector = FastLivenessDetector(**build_fast_detector_config(config))
    visualizer = BenchmarkVisualizer()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}")
        return

    print("✅ 摄像头已打开")
    print("提示:")
    print("  - 请面对摄像头，保持静止 2 秒以采集基准")
    print("  - 基准采集完成后，系统会自动验证后续帧")
    print("  - 按 'ESC' 退出  'R' 重置基准  'B' 开关基准功能")
    print()

    last_verification = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        # 推理（与视频模式相同）
        lm_data = engine.mp_detector.extract_landmarks(frame)

        if lm_data is None:
            result = {"is_live": False, "score": 0.0, "reason": "NO_FACE_DETECTED"}
        else:
            landmarks = lm_data["landmarks"]
            quality_score = lm_data["quality_score"]

            fd_result = fast_detector.detect_liveness(
                landmarks, lm_data.get("frame_shape", frame.shape)
            )

            motion_score = fd_result["score"]
            engine.score_history.append(motion_score)
            smoothed = float(
                sum(list(engine.score_history)[-config.smooth_window :])
                / min(len(engine.score_history), config.smooth_window)
            )

            is_live = smoothed > config.threshold
            result = {
                "is_live": is_live,
                "score": smoothed,
                "quality_score": quality_score,
                "reason": "LIVE" if is_live else "SPOOF",
            }

            # 基准帧校准
            if enable_benchmark and engine.calibrator is not None:
                embedding = lm_data.get("embedding")
                pitch = fd_result.get("pitch", 0.0)
                yaw = fd_result.get("yaw", 0.0)
                face_bbox = lm_data.get("face_bbox")

                if embedding is not None and face_bbox is not None:
                    if engine.calibrator.is_collecting_benchmark():
                        engine.calibrator.add_candidate_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            quality_score=quality_score,
                            face_bbox=face_bbox,
                            pitch=pitch,
                            yaw=yaw,
                            frame_index=engine.frame_count,
                        )
                        last_verification = {"status": "collecting"}
                    else:
                        verification = engine.calibrator.verify_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            pitch=pitch,
                            yaw=yaw,
                        )
                        last_verification = verification

        # 绘制 UI
        if enable_benchmark and engine.calibrator is not None:
            frame = visualizer.draw_benchmark_info(frame, engine.calibrator)
            if (
                last_verification
                and last_verification.get("embedding_similarity") is not None
            ):
                frame = visualizer.draw_verification_details(frame, last_verification)

        status_color = (0, 255, 0) if result.get("is_live", False) else (0, 0, 255)
        cv2.putText(
            frame,
            f"{result.get('reason', 'UNKNOWN')} score={result.get('score', 0):.2f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        cv2.imshow("Benchmark Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\n用户退出")
            break
        elif key == ord("r") or key == ord("R"):
            engine.reset()
            last_verification = {}
            print("\n重置基准")
        elif key == ord("b") or key == ord("B"):
            enable_benchmark = not enable_benchmark
            print(f"\n基准功能：{'ON' if enable_benchmark else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    engine.close()


def main():
    parser = argparse.ArgumentParser(description="基准帧校准功能演示")
    parser.add_argument("--video", "-v", type=str, default=None, help="视频文件路径")
    parser.add_argument("--camera", "-c", type=int, default=None, help="摄像头 ID")
    parser.add_argument("--no-benchmark", action="store_true", help="禁用基准帧功能")
    args = parser.parse_args()

    if args.video:
        run_video_benchmark(args.video, enable_benchmark=not args.no_benchmark)
    elif args.camera is not None:
        run_camera_benchmark(args.camera, enable_benchmark=not args.no_benchmark)
    else:
        print("请指定视频文件或摄像头:")
        print("  uv run python -m vrlFace.liveness.benchmark_demo --video test.mp4")
        print("  uv run python -m vrlFace.liveness.benchmark_demo --camera 0")


if __name__ == "__main__":
    main()
