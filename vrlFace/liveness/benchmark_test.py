"""
基准帧校准功能 - 无头模式测试（无需 GUI）

功能：
- 读取视频文件
- 输出基准帧采集和验证的详细信息
- 适合服务器/WSL 环境使用

使用方式:
    uv run python -m vrlFace.liveness.benchmark_test --video path/to/video.mp4
"""

import argparse
import time
import numpy as np
from typing import Dict, Any
from datetime import datetime

from .config import LivenessConfig
from .fusion_engine import LivenessFusionEngine
from .fast_detector import FastLivenessDetector
from .utils import build_fast_detector_config


def format_timestamp() -> str:
    """格式化时间戳"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("\n" + "=" * 60)


def run_headless_test(video_path: str, enable_benchmark: bool = True):
    """无头模式测试"""
    print_separator("基准帧校准功能 - 无头模式测试")
    print(f"测试时间：{format_timestamp()}")
    print(f"视频文件：{video_path}")
    print(f"基准功能：{'启用' if enable_benchmark else '禁用'}")

    # 初始化配置 - 针对 6 秒短视频优化
    config = LivenessConfig.video_fast_config()
    config.enable_benchmark = enable_benchmark
    if enable_benchmark:
        config.benchmark_duration = 2.0  # 前 2 秒采集
        config.benchmark_min_frames = 2  # 最少 2 帧即可
        config.benchmark_min_quality = 0.4  # 降低质量阈值适配低质量视频
        config.benchmark_max_angle = 20.0  # 放宽角度限制

    # 初始化引擎
    print("\n[初始化] 加载活体检测引擎...")
    engine = LivenessFusionEngine(config)
    fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

    # 打开视频
    import cv2

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 修复 webm 格式时长检测问题
    if total_frames <= 0 or total_frames > 1000000:
        total_frames = 180  # 6 秒@30fps 默认值

    duration = total_frames / fps if fps > 0 else 0

    print(f"[视频信息] {total_frames} 帧 @ {fps:.1f}fps, 时长={duration:.2f}秒")
    print(f"[预期结构] 0-2s 静止 (基准采集), 2-4s 动作 1, 4-6s 动作 2")

    # 统计信息
    stats = {
        "total_frames": 0,
        "face_detected": 0,
        "benchmark_collected": 0,
        "benchmark_verified": 0,
        "verification_passed": 0,
        "verification_failed": 0,
        "liveness_passed": 0,
        "liveness_failed": 0,
    }

    # 验证历史记录
    verification_history = []

    print_separator("开始处理视频")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stats["total_frames"] += 1
        frame_idx = stats["total_frames"]

        # 推理
        lm_data = engine.mp_detector.extract_landmarks(frame)

        if lm_data is None:
            # 未检测到人脸
            if frame_idx <= 10 or frame_idx % 30 == 0:
                print(f"[{format_timestamp()}] 帧 {frame_idx}: 未检测到人脸")
        else:
            stats["face_detected"] += 1

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
            if is_live:
                stats["liveness_passed"] += 1
            else:
                stats["liveness_failed"] += 1

            # 基准帧校准 - 需要从 InsightFace 获取 embedding
            # 注意：extract_landmarks 返回的是 MediaPipe 数据，不含 embedding
            # 需要使用 engine.mp_detector._face_analyzer 直接获取
            if enable_benchmark and engine.calibrator is not None:
                # 使用 InsightFace 获取 embedding
                embedding = None
                face_bbox = None

                if engine.mp_detector._face_analyzer is not None:
                    try:
                        faces = engine.mp_detector._face_analyzer.get(frame)
                        if faces:
                            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                            embedding = face.embedding
                            face_bbox = tuple(face.bbox.astype(int))
                    except Exception:
                        pass

                pitch = fd_result.get("pitch", 0.0)
                yaw = fd_result.get("yaw", 0.0)

                # 调试输出
                if frame_idx <= 5:
                    print(
                        f"[调试] 帧 {frame_idx}: embedding={embedding is not None}, bbox={face_bbox is not None}, quality={quality_score:.3f}"
                    )

                if embedding is not None and face_bbox is not None:
                    if engine.calibrator.is_collecting_benchmark():
                        added = engine.calibrator.add_candidate_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            quality_score=quality_score,
                            face_bbox=face_bbox,
                            pitch=pitch,
                            yaw=yaw,
                            frame_index=frame_idx,
                        )
                        if added:
                            stats["benchmark_collected"] += 1

                            if (
                                stats["benchmark_collected"] <= 5
                                or stats["benchmark_collected"] % 10 == 0
                            ):
                                status = engine.calibrator.get_status()
                                print(
                                    f"[{format_timestamp()}] 帧 {frame_idx}: 采集基准帧 #{stats['benchmark_collected']} "
                                    f"(质量={quality_score:.3f}, pitch={pitch:.1f}°, yaw={yaw:.1f}°)"
                                )
                    else:
                        verification = engine.calibrator.verify_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            pitch=pitch,
                            yaw=yaw,
                        )
                        stats["benchmark_verified"] += 1

                        is_same = verification.get("is_same_person", False)
                        emb_sim = verification.get("embedding_similarity", 0)
                        lm_sim = verification.get("landmark_similarity", 0)

                        if is_same:
                            stats["verification_passed"] += 1
                        else:
                            stats["verification_failed"] += 1

                        # 记录验证历史
                        verification_history.append(
                            {
                                "frame": frame_idx,
                                "is_same": is_same,
                                "emb_sim": emb_sim,
                                "lm_sim": lm_sim,
                            }
                        )

                        # 输出关键帧
                        if (
                            stats["benchmark_verified"] <= 5
                            or stats["benchmark_verified"] % 30 == 0
                        ):
                            status_str = "✅ 通过" if is_same else "❌ 失败"
                            print(
                                f"[{format_timestamp()}] 帧 {frame_idx}: 身份验证 {status_str} "
                                f"(embedding={emb_sim:.4f}, landmarks={lm_sim:.4f})"
                            )

                        # 检测首次失败
                        if stats["verification_failed"] == 1:
                            print(f"\n⚠️  首次检测到身份不匹配！帧 {frame_idx}")
                            print(f"   原因：{verification.get('reason', 'UNKNOWN')}")

        # 进度显示
        if frame_idx % 60 == 0:
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames * 100
            print(
                f"[{format_timestamp()}] 进度：{frame_idx}/{total_frames} ({progress:.1f}%), "
                f"耗时={elapsed:.1f}s, 人脸检出={stats['face_detected']}"
            )

    elapsed_time = time.time() - start_time
    cap.release()
    engine.close()

    # 输出统计结果
    print_separator("测试完成 - 统计结果")
    print(f"总耗时：{elapsed_time:.2f}秒")
    print(f"处理帧率：{stats['total_frames'] / elapsed_time:.1f} fps")

    print(f"\n【帧统计】")
    print(f"  总帧数：{stats['total_frames']}")
    print(
        f"  人脸检出：{stats['face_detected']} ({stats['face_detected'] / stats['total_frames'] * 100:.1f}%)"
    )

    print(f"\n【活体检测】")
    print(f"  通过：{stats['liveness_passed']} 帧")
    print(f"  失败：{stats['liveness_failed']} 帧")

    if enable_benchmark and engine.calibrator is not None:
        print(f"\n【基准帧校准】")
        status = engine.calibrator.get_status()
        print(f"  基准状态：{status.get('status', 'UNKNOWN')}")

        if status.get("status") == "READY":
            print(f"  采集帧数：{len(engine.calibrator.benchmark_frames)}")
            print(f"  基准质量：{engine.calibrator.benchmark_quality:.4f}")
            print(
                f"  基准角度：pitch={engine.calibrator.benchmark_pitch:.2f}°, yaw={engine.calibrator.benchmark_yaw:.2f}°"
            )

            print(f"\n【身份验证】")
            print(f"  验证帧数：{stats['benchmark_verified']}")
            print(
                f"  通过：{stats['verification_passed']} 帧 ({stats['verification_passed'] / stats['benchmark_verified'] * 100:.1f}%)"
            )
            print(
                f"  失败：{stats['verification_failed']} 帧 ({stats['verification_failed'] / stats['benchmark_verified'] * 100:.1f}%)"
            )

            # 验证趋势分析
            if verification_history:
                print(f"\n【验证趋势】")
                # 分段统计
                segment_size = max(len(verification_history) // 5, 1)
                for i in range(0, len(verification_history), segment_size):
                    segment = verification_history[i : i + segment_size]
                    passed = sum(1 for v in segment if v["is_same"])
                    avg_emb = np.mean([v["emb_sim"] for v in segment])
                    print(
                        f"  帧 {i + 1}-{i + len(segment)}: 通过率={passed / len(segment) * 100:.0f}%, "
                        f"平均 embedding={avg_emb:.4f}"
                    )

            # 最终判定
            all_passed = stats["verification_failed"] == 0
            print(f"\n{'✅ 身份验证全程通过' if all_passed else '⚠️  检测到身份变化'}")
        else:
            print(f"  ⚠️  基准采集未完成")

    print_separator()


def main():
    parser = argparse.ArgumentParser(description="基准帧校准功能 - 无头模式测试")
    parser.add_argument("--video", "-v", type=str, required=True, help="视频文件路径")
    parser.add_argument("--no-benchmark", action="store_true", help="禁用基准帧功能")
    args = parser.parse_args()

    run_headless_test(args.video, enable_benchmark=not args.no_benchmark)


if __name__ == "__main__":
    main()
