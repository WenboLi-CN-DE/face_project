#!/usr/bin/env python3
"""
CSV 数据分析脚本

分析 recorder.py 生成的逐帧 CSV 文件，生成详细统计和可视化数据

使用方法:
    uv run python scripts/workflow/analyze_csv_detailed.py --csv output/diagnose/csv/video.webm.frames.csv
    uv run python scripts/workflow/analyze_csv_detailed.py --csv output/diagnose/csv/video.webm.frames.csv --plot
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def analyze_csv(csv_path: str) -> Dict[str, Any]:
    """分析 CSV 数据"""
    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(csv_path)

        # 基础统计
        total_frames = len(df)
        face_detected_frames = df["face_detected"].sum()
        face_detected_rate = face_detected_frames / total_frames * 100

        # 活体统计
        is_live_frames = df["is_live"].sum()
        is_live_rate = is_live_frames / total_frames * 100
        best_score = float(df["score_smoothed"].max())
        avg_score = float(df["score_smoothed"].mean())

        # EAR 统计（眨眼）
        ear = df["ear"]
        ear_stats = {
            "min": float(ear.min()),
            "max": float(ear.max()),
            "mean": float(ear.mean()),
            "std": float(ear.std()),
            "below_0.15": int((ear < 0.15).sum()),
            "below_0.20": int((ear < 0.20).sum()),
            "below_0.25": int((ear < 0.25).sum()),
        }

        # MAR 统计（张嘴）
        mar = df["mar"]
        mar_stats = {
            "min": float(mar.min()),
            "max": float(mar.max()),
            "mean": float(mar.mean()),
            "std": float(mar.std()),
            "above_0.40": int((mar > 0.40).sum()),
            "above_0.50": int((mar > 0.50).sum()),
            "above_0.60": int((mar > 0.60).sum()),
        }

        # Pitch/Yaw 统计
        pitch = df["pitch"]
        yaw = df["yaw"]

        pitch_stats = {
            "min": float(pitch.min()),
            "max": float(pitch.max()),
            "mean": float(pitch.mean()),
            "range": float(pitch.max() - pitch.min()),
        }

        yaw_stats = {
            "min": float(yaw.min()),
            "max": float(yaw.max()),
            "mean": float(yaw.mean()),
            "range": float(yaw.max() - yaw.min()),
        }

        # 动作事件统计
        blink_frames = df["blink_detected"].sum()
        mouth_frames = df["mouth_open"].sum()
        head_action_frames = (df["head_action"] != "none").sum()

        # 眨眼片段分析
        blink_segments = find_consecutive_segments((ear < 0.20).values, min_length=3)
        mouth_segments = find_consecutive_segments((mar > 0.40).values, min_length=3)

        return {
            "file": csv_path,
            "timestamp": datetime.now().isoformat(),
            "total_frames": total_frames,
            "face_detected_rate": face_detected_rate,
            "is_live_rate": is_live_rate,
            "best_score": best_score,
            "avg_score": avg_score,
            "ear": ear_stats,
            "mar": mar_stats,
            "pitch": pitch_stats,
            "yaw": yaw_stats,
            "action_frames": {
                "blink": int(blink_frames),
                "mouth_open": int(mouth_frames),
                "head_action": int(head_action_frames),
            },
            "segments": {
                "blink_segments": len(blink_segments),
                "mouth_segments": len(mouth_segments),
            },
        }
    except Exception as e:
        return {"error": str(e)}


def find_consecutive_segments(arr, min_length=3):
    """查找连续片段"""
    segments = []
    in_seg = False
    start = 0

    for i, v in enumerate(arr):
        if v and not in_seg:
            in_seg = True
            start = i
        elif not v and in_seg:
            in_seg = False
            if i - start >= min_length:
                segments.append((start, i - 1))

    if in_seg and len(arr) - start >= min_length:
        segments.append((start, len(arr) - 1))

    return segments


def generate_diagnosis(analysis: Dict[str, Any]) -> list:
    """生成诊断建议"""
    recommendations = []

    if "error" in analysis:
        return [f"分析失败：{analysis['error']}"]

    # 人脸检出率
    if analysis["face_detected_rate"] < 80:
        recommendations.append(
            f"⚠️  人脸检出率过低 ({analysis['face_detected_rate']:.1f}%)，建议检查视频质量或光线条件"
        )

    # 活体判定
    if analysis["is_live_rate"] < 50:
        recommendations.append(
            f"⚠️  活体通过率过低 ({analysis['is_live_rate']:.1f}%)，建议检查动作幅度或调整阈值"
        )

    # EAR 分析
    ear = analysis["ear"]
    if ear["below_0.20"] / analysis["total_frames"] < 0.30:
        recommendations.append(
            f"⚠️  眨眼帧占比过低 ({ear['below_0.20'] / analysis['total_frames'] * 100:.1f}%)，"
            f"EAR 范围 {ear['min']:.3f}-{ear['max']:.3f}，建议降低 ear_threshold"
        )

    # MAR 分析
    mar = analysis["mar"]
    if mar["above_0.40"] / analysis["total_frames"] < 0.25:
        recommendations.append(
            f"⚠️  张嘴帧占比过低 ({mar['above_0.40'] / analysis['total_frames'] * 100:.1f}%)，"
            f"MAR 范围 {mar['min']:.3f}-{mar['max']:.3f}，建议降低 mar_threshold"
        )

    # Pitch 分析
    pitch = analysis["pitch"]
    if pitch["range"] < 8.0:
        recommendations.append(
            f"⚠️  Pitch 峰峰值过小 ({pitch['range']:.1f}° < 8.0°)，点头动作不明显"
        )

    # Yaw 分析
    yaw = analysis["yaw"]
    if yaw["range"] < 8.0:
        recommendations.append(
            f"⚠️  Yaw 峰峰值过小 ({yaw['range']:.1f}° < 8.0°)，转头动作不明显"
        )

    if not recommendations:
        recommendations.append("✅ 各项指标正常")

    return recommendations


def print_report(analysis: Dict[str, Any]):
    """打印分析报告"""
    print("=" * 80)
    print("CSV 数据分析报告")
    print("=" * 80)

    if "error" in analysis:
        print(f"❌ 分析失败：{analysis['error']}")
        return

    print(f"文件：{analysis['file']}")
    print(f"分析时间：{analysis['timestamp']}")
    print()

    print("【基础统计】")
    print(f"  总帧数：{analysis['total_frames']}")
    print(f"  人脸检出率：{analysis['face_detected_rate']:.1f}%")
    print(f"  活体通过率：{analysis['is_live_rate']:.1f}%")
    print(f"  最佳分数：{analysis['best_score']:.4f}")
    print(f"  平均分数：{analysis['avg_score']:.4f}")
    print()

    print("【EAR 统计（眨眼）】")
    ear = analysis["ear"]
    print(f"  范围：{ear['min']:.3f} - {ear['max']:.3f}")
    print(f"  平均：{ear['mean']:.3f} ± {ear['std']:.3f}")
    print(f"  EAR < 0.15: {ear['below_0.15']} 帧")
    print(f"  EAR < 0.20: {ear['below_0.20']} 帧")
    print(f"  EAR < 0.25: {ear['below_0.25']} 帧")
    print()

    print("【MAR 统计（张嘴）】")
    mar = analysis["mar"]
    print(f"  范围：{mar['min']:.3f} - {mar['max']:.3f}")
    print(f"  平均：{mar['mean']:.3f} ± {mar['std']:.3f}")
    print(f"  MAR > 0.40: {mar['above_0.40']} 帧")
    print(f"  MAR > 0.50: {mar['above_0.50']} 帧")
    print(f"  MAR > 0.60: {mar['above_0.60']} 帧")
    print()

    print("【头部姿态】")
    pitch = analysis["pitch"]
    yaw = analysis["yaw"]
    print(
        f"  Pitch: {pitch['min']:+.1f}° ~ {pitch['max']:+.1f}° (范围：{pitch['range']:.1f}°)"
    )
    print(
        f"  Yaw:   {yaw['min']:+.1f}° ~ {yaw['max']:+.1f}° (范围：{yaw['range']:.1f}°)"
    )
    print()

    print("【动作事件】")
    action = analysis["action_frames"]
    print(f"  眨眼：{action['blink']} 帧")
    print(f"  张嘴：{action['mouth_open']} 帧")
    print(f"  头部动作：{action['head_action']} 帧")
    print()

    print("【诊断建议】")
    recommendations = generate_diagnosis(analysis)
    for rec in recommendations:
        print(f"  {rec}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="CSV 数据分析脚本")
    parser.add_argument("--csv", required=True, help="CSV 文件路径")
    parser.add_argument("--output", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--plot", action="store_true", help="生成可视化图表")

    args = parser.parse_args()

    # 分析 CSV
    analysis = analyze_csv(args.csv)

    # 打印报告
    print_report(analysis)

    # 保存 JSON 报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n📄 JSON 报告已保存：{output_path}")

    # 生成图表
    if args.plot:
        try:
            generate_plots(args.csv, analysis)
        except Exception as e:
            print(f"\n⚠️  图表生成失败：{e}")


def generate_plots(csv_path: str, analysis: Dict[str, Any]):
    """生成可视化图表"""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # 1. EAR 曲线
    ax = axes[0]
    ax.plot(df.index, df["ear"], label="EAR", color="blue", linewidth=0.5)
    ax.axhline(y=0.20, color="red", linestyle="--", label="Threshold (0.20)")
    ax.fill_between(
        df.index, df["ear"], 0.20, where=(df["ear"] < 0.20), alpha=0.3, color="green"
    )
    ax.set_ylabel("EAR")
    ax.set_title("Eye Aspect Ratio (Blink Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MAR 曲线
    ax = axes[1]
    ax.plot(df.index, df["mar"], label="MAR", color="orange", linewidth=0.5)
    ax.axhline(y=0.40, color="red", linestyle="--", label="Threshold (0.40)")
    ax.fill_between(
        df.index, df["mar"], 0.40, where=(df["mar"] > 0.40), alpha=0.3, color="green"
    )
    ax.set_ylabel("MAR")
    ax.set_title("Mouth Aspect Ratio (Yawn Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Pitch/Yaw 曲线
    ax = axes[2]
    ax.plot(df.index, df["pitch"], label="Pitch", color="green", linewidth=0.5)
    ax.plot(df.index, df["yaw"], label="Yaw", color="purple", linewidth=0.5)
    ax.axhline(y=8.0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=-8.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Angle (°)")
    ax.set_title("Head Pose (Pitch/Yaw)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 活体分数曲线
    ax = axes[3]
    ax.plot(
        df.index,
        df["score_smoothed"],
        label="Smoothed Score",
        color="red",
        linewidth=0.5,
    )
    ax.axhline(y=0.45, color="blue", linestyle="--", label="Threshold (0.45)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Frame")
    ax.set_title("Liveness Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = Path(csv_path).with_suffix(".png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 图表已保存：{output_path}")


if __name__ == "__main__":
    main()
