#!/usr/bin/env python3
"""
诊断 is_liveness=0 的问题
分析分数分布和阈值判断
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.fusion_engine import LivenessFusionEngine
from vrlFace.liveness.utils import build_fast_detector_config
from vrlFace.liveness.fast_detector import FastLivenessDetector


def analyze_video_detailed(video_path: str):
    """详细分析视频，打印每一帧的分数"""
    print(f"\n{'=' * 70}")
    print(f"诊断视频：{video_path}")
    print(f"{'=' * 70}")

    config = LivenessConfig.video_fast_config()
    print(f"\n【配置参数】")
    print(f"  threshold: {config.threshold}")
    print(f"  smooth_window: {config.smooth_window}")
    print(f"  window_size: {config.window_size}")

    # 手动分析视频，打印详细数据
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频")
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 修复：如果帧数为负数，通过实际读取计算
    if total_frames <= 0:
        print("检测到帧数无效，通过实际读取计算...")
        temp_cap = cv2.VideoCapture(video_path)
        actual_frames = 0
        while temp_cap.read()[0]:
            actual_frames += 1
        temp_cap.release()
        total_frames = actual_frames

    print(f"\n【视频信息】")
    print(f"  总帧数：{total_frames}")
    print(f"  FPS: {fps}")
    print(f"  分辨率：{width}x{height}")

    # 初始化检测器
    engine = LivenessFusionEngine(config)
    fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

    # 收集数据
    frame_scores = []  # 每帧的原始 motion_score
    smoothed_scores = []  # 平滑后的分数
    quality_scores = []  # 质量分数
    face_detected_frames = 0

    frame_idx = 0
    max_frames = min(total_frames, 180)  # 最多分析 180 帧

    print(f"\n【逐帧分析】（前 {max_frames} 帧）")
    print(
        f"{'帧号':^6} | {'人脸':^4} | {'原始分':^8} | {'平滑分':^8} | {'质量分':^8} | {'累计帧':^6}"
    )
    print(f"{'-' * 70}")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 缩小帧以加速处理
        max_w = config.max_width
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (max_w, int(frame.shape[0] * scale)))

        # 提取 landmarks
        lm_data = engine.mp_detector.extract_landmarks(frame)

        if lm_data is None:
            # 未检测到人脸
            frame_scores.append(0.0)
            smoothed_scores.append(0.0)
            quality_scores.append(0.0)
            print(
                f"{frame_idx:^6} | {'✗':^4} | {'0.0000':^8} | {'0.0000':^8} | {'0.0000':^8} | {len(engine.score_history):^6}"
            )
            continue

        face_detected_frames += 1
        landmarks = lm_data["landmarks"]
        quality_score = lm_data["quality_score"]

        # fast_detector 检测
        fd_result = fast_detector.detect_liveness(
            landmarks, lm_data.get("frame_shape", frame.shape)
        )

        motion_score = fd_result["score"]
        engine.score_history.append(motion_score)

        # 计算平滑分数
        smoothed = float(
            sum(list(engine.score_history)[-config.smooth_window :])
            / min(len(engine.score_history), config.smooth_window)
        )

        frame_scores.append(motion_score)
        smoothed_scores.append(smoothed)
        quality_scores.append(quality_score)

        face_flag = "✓" if lm_data else "✗"
        print(
            f"{frame_idx:^6} | {face_flag:^4} | {motion_score:^8.4f} | {smoothed:^8.4f} | {quality_score:^8.4f} | {len(engine.score_history):^6}"
        )

    cap.release()
    engine.close()

    # 统计分析
    print(f"\n{'=' * 70}")
    print(f"【统计分析】")
    print(f"{'=' * 70}")

    # 人脸检测统计
    print(f"\n人脸检测:")
    print(
        f"  检出帧数：{face_detected_frames}/{frame_idx} ({face_detected_frames / frame_idx * 100:.1f}%)"
    )

    # 原始分数统计
    nonzero_scores = [s for s in frame_scores if s > 0]
    if nonzero_scores:
        print(f"\n原始 motion_score (仅人脸帧):")
        print(f"  最高分：{max(nonzero_scores):.4f}")
        print(f"  平均分：{np.mean(nonzero_scores):.4f}")
        print(f"  最低分：{min(nonzero_scores):.4f}")

    # 平滑分数统计
    nonzero_smoothed = [s for s in smoothed_scores if s > 0]
    if nonzero_smoothed:
        print(f"\n平滑分数 (仅人脸帧):")
        print(f"  最高分：{max(nonzero_smoothed):.4f}")
        print(f"  平均分：{np.mean(nonzero_smoothed):.4f}")
        print(f"  最低分：{min(nonzero_smoothed):.4f}")

    # 质量分数统计
    nonzero_quality = [q for q in quality_scores if q > 0]
    if nonzero_quality:
        print(f"\n质量分数 (仅人脸帧):")
        print(f"  最高分：{max(nonzero_quality):.4f}")
        print(f"  平均分：{np.mean(nonzero_quality):.4f}")
        print(f"  最低分：{min(nonzero_quality):.4f}")

    # 关键判断逻辑
    print(f"\n{'=' * 70}")
    print(f"【活体判定逻辑】")
    print(f"{'=' * 70}")

    best_score = max(smoothed_scores) if smoothed_scores else 0.0
    is_liveness = 1 if best_score >= config.threshold else 0

    print(f"\n代码逻辑:")
    print(f"  best_score = max(all_scores) = {best_score:.4f}")
    print(f"  threshold = {config.threshold}")
    print(f"  is_liveness = 1 if best_score >= threshold else 0")
    print(f"  is_liveness = {is_liveness}")

    if is_liveness == 0:
        print(f"\n⚠️  问题诊断:")
        if best_score < config.threshold:
            print(f"  ❌ 最高平滑分 ({best_score:.4f}) < 阈值 ({config.threshold})")
            print(f"  💡 建议：降低阈值到 {best_score * 0.9:.2f} 或提高动作幅度")
    else:
        print(f"\n✅ 活体判定通过")

    # 分数分布
    print(f"\n{'=' * 70}")
    print(f"【分数分布】")
    print(f"{'=' * 70}")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    print(f"\n平滑分数分布:")
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = sum(1 for s in nonzero_smoothed if low <= s < high)
        pct = count / len(nonzero_smoothed) * 100 if nonzero_smoothed else 0
        bar = "█" * int(pct / 2)
        print(f"  {bin_labels[i]:^8} | {count:^4} ({pct:5.1f}%) {bar}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    # 测试第一个视频
    videos_dir = Path(__file__).parent / "videos"
    test_video = videos_dir / "9a49613f-6a1d-465c-9171-7c6f2f28a203_1773582414.webm"

    if test_video.exists():
        analyze_video_detailed(str(test_video))
    else:
        print(f"视频不存在：{test_video}")
