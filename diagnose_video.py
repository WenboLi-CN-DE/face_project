"""
诊断视频检测过程 - 记录所有关键参数

输出CSV文件，包含每一帧的：
- pitch, yaw 原始值
- baseline_pitch, baseline_yaw
- dp, dy (相对基线的偏移)
- 检测到的动作
- 是否触发事件
"""

import csv
import cv2
from pathlib import Path
from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.fast_detector import FastLivenessDetector
from vrlFace.liveness.fusion_engine import LivenessFusionEngine
from vrlFace.liveness.utils import build_fast_detector_config


def diagnose_video(video_path: str, output_csv: str):
    """诊断单个视频"""
    print(f"诊断视频: {video_path}")
    print(f"输出CSV: {output_csv}")

    config = LivenessConfig.video_fast_config()
    engine = LivenessFusionEngine(config)
    fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {fps:.1f} FPS, {total_frames} 帧")

    csv_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        lm_data = engine.mp_detector.extract_landmarks(frame)
        if lm_data is None:
            csv_data.append(
                {
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps,
                    "face_detected": False,
                }
            )
            continue

        landmarks = lm_data["landmarks"]
        fd_result = fast_detector.detect_liveness(
            landmarks, lm_data.get("frame_shape", frame.shape)
        )

        pitch = fd_result.get("pitch", 0.0)
        yaw = fd_result.get("yaw", 0.0)
        current_action = fd_result.get("head_action", "none")

        head_detector = fast_detector._head_action_detector

        pitch_history = (
            list(head_detector.pitch_history)
            if len(head_detector.pitch_history) > 0
            else []
        )
        yaw_history = (
            list(head_detector.yaw_history)
            if len(head_detector.yaw_history) > 0
            else []
        )

        pitch_range = (
            max(pitch_history) - min(pitch_history) if len(pitch_history) > 0 else 0.0
        )
        yaw_range = max(yaw_history) - min(yaw_history) if len(yaw_history) > 0 else 0.0

        csv_data.append(
            {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 2),
                "face_detected": True,
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "pitch_range": round(pitch_range, 2),
                "yaw_range": round(yaw_range, 2),
                "action": current_action,
                "cooldown": head_detector._cooldown,
                "pending_action": head_detector._pending_action,
                "pending_count": head_detector._pending_count,
            }
        )

        if frame_idx % 30 == 0:
            print(f"  处理中... {frame_idx}/{total_frames} 帧")

    cap.release()
    engine.close()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"✅ 诊断完成，共 {len(csv_data)} 帧")
    print(f"📊 CSV文件已保存: {output_csv}")

    analyze_csv(output_csv)


def analyze_csv(csv_path: str):
    """分析CSV数据，输出统计信息"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df_face = df[df["face_detected"] == True]

    if len(df_face) == 0:
        print("⚠️  没有检测到人脸")
        return

    print("\n" + "=" * 60)
    print("统计分析")
    print("=" * 60)

    print(f"\n总帧数: {len(df)}")
    print(f"检测到人脸: {len(df_face)} 帧 ({len(df_face) / len(df) * 100:.1f}%)")

    print(f"\n角度范围:")
    print(
        f"  Pitch: {df_face['pitch'].min():.1f}° ~ {df_face['pitch'].max():.1f}° (范围: {df_face['pitch'].max() - df_face['pitch'].min():.1f}°)"
    )
    print(
        f"  Yaw:   {df_face['yaw'].min():.1f}° ~ {df_face['yaw'].max():.1f}° (范围: {df_face['yaw'].max() - df_face['yaw'].min():.1f}°)"
    )

    print(f"\n偏移量统计:")
    print(f"  |dp| 最大: {df_face['abs_dp'].max():.1f}°")
    print(f"  |dy| 最大: {df_face['abs_dy'].max():.1f}°")
    print(f"  |dp| > 12°: {len(df_face[df_face['abs_dp'] > 12])} 帧")
    print(f"  |dy| > 12°: {len(df_face[df_face['abs_dy'] > 12])} 帧")

    actions = df_face[df_face["action"] != "none"]["action"].value_counts()
    if len(actions) > 0:
        print(f"\n检测到的动作:")
        for action, count in actions.items():
            print(f"  {action}: {count} 次")
    else:
        print(f"\n⚠️  未检测到任何动作")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: uv run python diagnose_video.py <视频路径> [输出CSV路径]")
        print("\n示例:")
        print(
            "  uv run python diagnose_video.py 15/5cef26ef-6b3f-4ffe-bdaf-34faa7726c5f_1773557272.webm"
        )
        sys.exit(1)

    video_path = sys.argv[1]
    output_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else video_path.replace(".webm", "_diagnosis.csv")
    )

    diagnose_video(video_path, output_csv)
