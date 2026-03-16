#!/usr/bin/env python3
"""
活体检测自动化诊断工作流

功能：
1. 批量测试视频，识别问题视频
2. 自动保存问题视频的逐帧 CSV
3. 自动分析 CSV 数据
4. 生成诊断报告和建议

使用方法：
    uv run python scripts/workflow/auto_diagnose.py --video-dir 15
    uv run python scripts/workflow/auto_diagnose.py --video-dir 15 --actions blink mouth_open
    uv run python scripts/workflow/auto_diagnose.py --video-dir 15 --csv --report
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.recorder import run_video_detection_with_csv


# ============================================================================
# 配置
# ============================================================================


class DiagnoseConfig:
    """诊断配置"""

    # 阈值配置
    LIVENESS_THRESHOLD = 0.45  # 活体判定阈值
    ACTION_THRESHOLD = 0.75  # 动作通过阈值

    # EAR 阈值（眨眼）
    EAR_THRESHOLD = 0.20  # 眨眼触发阈值
    EAR_LOW_RATE = 0.30  # 眨眼帧占比 < 30% 判定为不足

    # MAR 阈值（张嘴）
    MAR_THRESHOLD = 0.28  # 张嘴触发阈值
    MAR_LOW_RATE = 0.25  # 张嘴帧占比 < 25% 判定为不足

    # 头部动作阈值
    PITCH_THRESHOLD = 8.0  # 点头角度阈值
    YAW_THRESHOLD = 8.0  # 转头角度阈值

    # 输出配置
    OUTPUT_DIR = Path("output/diagnose")
    CSV_DIR = OUTPUT_DIR / "csv"
    REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================================
# 数据类
# ============================================================================


class VideoTestResult:
    """单个视频测试结果"""

    def __init__(self, video_name: str, expected_actions: List[str]):
        self.video_name = video_name
        self.expected_actions = expected_actions
        self.is_liveness = False
        self.liveness_confidence = 0.0
        self.action_results: Dict[str, Dict] = {}
        self.csv_path: Optional[str] = None
        self.error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "video_name": self.video_name,
            "expected_actions": self.expected_actions,
            "is_liveness": self.is_liveness,
            "liveness_confidence": self.liveness_confidence,
            "action_results": self.action_results,
            "csv_path": self.csv_path,
            "error": self.error,
        }


class DiagnoseReport:
    """诊断报告"""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.total_videos = 0
        self.passed_videos = 0
        self.failed_videos = 0
        self.video_results: List[VideoTestResult] = []
        self.action_stats: Dict[str, Dict] = {}
        self.recommendations: List[str] = []

    def add_result(self, result: VideoTestResult):
        self.video_results.append(result)
        if result.error:
            return

        self.total_videos += 1
        if result.is_liveness and all(
            r.get("passed", False) for r in result.action_results.values()
        ):
            self.passed_videos += 1
        else:
            self.failed_videos += 1

    def generate_recommendations(self):
        """生成改进建议"""
        self.recommendations = []

        # 统计各动作通过率
        action_pass_rates = {}
        for result in self.video_results:
            if result.error:
                continue
            for action, data in result.action_results.items():
                if action not in action_pass_rates:
                    action_pass_rates[action] = {
                        "passed": 0,
                        "total": 0,
                        "confidences": [],
                    }
                action_pass_rates[action]["total"] += 1
                if data.get("passed", False):
                    action_pass_rates[action]["passed"] += 1
                action_pass_rates[action]["confidences"].append(
                    data.get("confidence", 0)
                )

        # 分析低通过率动作
        for action, stats in action_pass_rates.items():
            pass_rate = (
                stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            avg_conf = (
                sum(stats["confidences"]) / len(stats["confidences"])
                if stats["confidences"]
                else 0
            )

            if pass_rate < 60:
                self.recommendations.append(
                    f"⚠️  动作 '{action}' 通过率过低 ({pass_rate:.1f}%)，平均置信度 {avg_conf:.2f}"
                )

                # 针对性建议
                if action == "blink":
                    self.recommendations.append(
                        f"   → 建议：降低 ear_threshold (当前 {DiagnoseConfig.EAR_THRESHOLD:.2f})"
                    )
                elif action == "mouth_open":
                    self.recommendations.append(
                        f"   → 建议：降低 mar_threshold (当前 {DiagnoseConfig.MAR_THRESHOLD:.2f})"
                    )
                elif action in ["nod", "shake_head"]:
                    self.recommendations.append(
                        f"   → 建议：降低 pitch/yaw_threshold (当前 {DiagnoseConfig.PITCH_THRESHOLD:.1f}°/{DiagnoseConfig.YAW_THRESHOLD:.1f}°)"
                    )

        # 活体判定通过率
        if self.total_videos > 0:
            liveness_pass_rate = self.passed_videos / self.total_videos * 100
            if liveness_pass_rate < 70:
                self.recommendations.append(
                    f"⚠️  总体活体通过率过低 ({liveness_pass_rate:.1f}%)，建议检查视频质量或调整 threshold"
                )

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "total_videos": self.total_videos,
            "passed_videos": self.passed_videos,
            "failed_videos": self.failed_videos,
            "pass_rate": self.passed_videos / self.total_videos * 100
            if self.total_videos > 0
            else 0,
            "video_results": [r.to_dict() for r in self.video_results],
            "recommendations": self.recommendations,
        }


# ============================================================================
# 核心功能
# ============================================================================


def test_single_video(
    video_path: str, expected_actions: List[str], save_csv: bool = False
) -> VideoTestResult:
    """测试单个视频"""
    result = VideoTestResult(Path(video_path).name, expected_actions)

    try:
        # 1. 使用 VideoLivenessAnalyzer 测试
        config = LivenessConfig.video_fast_config()
        analyzer = VideoLivenessAnalyzer(
            liveness_config=config,
            liveness_threshold=DiagnoseConfig.LIVENESS_THRESHOLD,
            action_threshold=DiagnoseConfig.ACTION_THRESHOLD,
            enable_benchmark=False,
        )

        analyze_result = analyzer.analyze(
            video_path=video_path,
            actions=expected_actions,
        )

        result.is_liveness = bool(analyze_result.is_liveness)
        result.liveness_confidence = analyze_result.liveness_confidence

        # 提取每个动作的结果
        for detail in analyze_result.action_verify.action_details:
            result.action_results[detail.action] = {
                "passed": detail.passed,
                "confidence": detail.confidence,
                "msg": detail.msg,
            }

        # 2. 如果是失败视频，保存逐帧 CSV
        if save_csv and not result.is_liveness:
            csv_path = DiagnoseConfig.CSV_DIR / f"{result.video_name}.frames.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            run_video_detection_with_csv(
                video_path=video_path,
                output_csv=str(csv_path),
                config=config,
                show_ui=False,
            )
            result.csv_path = str(csv_path)

    except Exception as e:
        result.error = str(e)

    return result


def analyze_csv_data(csv_path: str) -> Dict[str, Any]:
    """分析 CSV 数据"""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        # EAR 统计
        ear = df["ear"]
        ear_below_threshold = (ear < DiagnoseConfig.EAR_THRESHOLD).sum()
        ear_below_rate = ear_below_threshold / len(ear) * 100 if len(ear) > 0 else 0

        # MAR 统计
        mar = df["mar"]
        mar_above_threshold = (mar > DiagnoseConfig.MAR_THRESHOLD).sum()
        mar_above_rate = mar_above_threshold / len(mar) * 100 if len(mar) > 0 else 0

        # Pitch/Yaw 峰峰值
        pitch_range = df["pitch"].max() - df["pitch"].min()
        yaw_range = df["yaw"].max() - df["yaw"].min()

        # 人脸检出率
        face_detected_rate = (
            df["face_detected"].sum() / len(df) * 100 if len(df) > 0 else 0
        )

        return {
            "frame_count": len(df),
            "face_detected_rate": face_detected_rate,
            "ear": {
                "min": float(ear.min()),
                "max": float(ear.max()),
                "mean": float(ear.mean()),
                "below_threshold_frames": int(ear_below_threshold),
                "below_threshold_rate": ear_below_rate,
            },
            "mar": {
                "min": float(mar.min()),
                "max": float(mar.max()),
                "mean": float(mar.mean()),
                "above_threshold_frames": int(mar_above_threshold),
                "above_threshold_rate": mar_above_rate,
            },
            "pitch_range": float(pitch_range),
            "yaw_range": float(yaw_range),
        }
    except Exception as e:
        return {"error": str(e)}


def write_diagnose_report(report: DiagnoseReport, output_path: str):
    """生成诊断报告"""
    report.generate_recommendations()

    # JSON 报告
    json_path = (
        Path(output_path)
        / f"diagnose_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 80)
    print("诊断报告摘要")
    print("=" * 80)
    print(f"测试时间：{report.timestamp}")
    print(f"测试视频数：{report.total_videos}")
    print(
        f"通过：{report.passed_videos} ({report.passed_videos / report.total_videos * 100:.1f}%)"
    )
    print(
        f"失败：{report.failed_videos} ({report.failed_videos / report.total_videos * 100:.1f}%)"
    )

    if report.recommendations:
        print("\n改进建议:")
        for rec in report.recommendations:
            print(rec)

    print(f"\n完整报告：{json_path}")
    print("=" * 80)
    print(f"测试时间：{report.timestamp}")
    print(f"测试视频数：{report.total_videos}")
    print(
        f"通过：{report.passed_videos} ({report.passed_videos / report.total_videos * 100:.1f}%)"
    )
    print(
        f"失败：{report.failed_videos} ({report.failed_videos / report.total_videos * 100:.1f}%)"
    )

    if report.recommendations:
        print("\n改进建议:")
        for rec in report.recommendations:
            print(rec)

    print(f"\n完整报告：{json_path}")
    print("=" * 80)


# ============================================================================
# 主流程
# ============================================================================


def run_workflow(
    video_dir: str,
    action_file: Optional[str] = None,
    save_csv: bool = True,
    generate_report: bool = True,
):
    """运行诊断工作流"""

    # 初始化
    video_dir_path = Path(video_dir)
    action_file_path = (
        Path(action_file) if action_file else video_dir_path / "动作对应.txt"
    )

    if not action_file_path.exists():
        print(f"❌ 找不到动作配置文件：{action_file_path}")
        return

    # 解析动作配置
    ACTION_MAP = {
        "点头": "nod",
        "抬头": "nod_up",
        "低头": "nod_down",
        "摇头": "shake_head",
        "左摇头": "turn_left",
        "右摇头": "turn_right",
        "眨眼": "blink",
        "张嘴": "mouth_open",
    }

    video_actions = []
    with open(action_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ".webm" not in line:
                continue
            import re

            match = re.match(r"^(.+\.webm)\s+(.+)$", line)
            if not match:
                continue
            video_name = match.group(1).strip()
            actions_str = match.group(2).strip()
            actions = [
                ACTION_MAP[cn] for cn, en in ACTION_MAP.items() if cn in actions_str
            ]
            if actions:
                video_actions.append((video_name, actions))

    print("=" * 80)
    print("活体检测自动化诊断工作流")
    print("=" * 80)
    print(f"视频目录：{video_dir_path}")
    print(f"动作配置：{action_file_path}")
    print(f"待测试视频：{len(video_actions)} 个")
    print()

    # 创建输出目录
    DiagnoseConfig.CSV_DIR.mkdir(parents=True, exist_ok=True)
    DiagnoseConfig.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 测试每个视频
    report = DiagnoseReport()

    for idx, (video_name, actions) in enumerate(video_actions, 1):
        video_path = video_dir_path / video_name

        if not video_path.exists():
            print(f"[{idx}/{len(video_actions)}] ⚠️  跳过（文件不存在）: {video_name}")
            continue

        print(f"[{idx}/{len(video_actions)}] 测试：{video_name}")
        print(f"  期望动作：{actions}")

        result = test_single_video(str(video_path), actions, save_csv=save_csv)
        report.add_result(result)

        if result.error:
            print(f"  ❌ 错误：{result.error}")
        else:
            status = "✅" if result.is_liveness else "❌"
            print(f"  活体：{status} ({result.liveness_confidence:.2%})")

            for action, data in result.action_results.items():
                action_status = "✅" if data["passed"] else "❌"
                print(
                    f"    {action_status} {action:12s} 置信度：{data['confidence']:.2%}"
                )

            if result.csv_path:
                print(f"  📊 CSV 已保存：{result.csv_path}")

                # 分析 CSV 数据
                csv_analysis = analyze_csv_data(result.csv_path)
                if "error" not in csv_analysis:
                    print(f"  📈 CSV 分析:")
                    print(f"     人脸检出率：{csv_analysis['face_detected_rate']:.1f}%")
                    print(
                        f"     EAR: {csv_analysis['ear']['min']:.3f} - {csv_analysis['ear']['max']:.3f}"
                    )
                    print(
                        f"     MAR: {csv_analysis['mar']['min']:.3f} - {csv_analysis['mar']['max']:.3f}"
                    )
                    print(f"     Pitch 范围：{csv_analysis['pitch_range']:.1f}°")
                    print(f"     Yaw 范围：{csv_analysis['yaw_range']:.1f}°")

        print()

    # 生成报告
    if generate_report:
        write_diagnose_report(report, str(DiagnoseConfig.REPORT_DIR))
    else:
        # 简单汇总
        print("=" * 80)
        print("汇总统计")
        print("=" * 80)
        print(f"测试视频数：{report.total_videos}")
        print(
            f"通过：{report.passed_videos} ({report.passed_videos / report.total_videos * 100:.1f}%)"
        )
        print(
            f"失败：{report.failed_videos} ({report.failed_videos / report.total_videos * 100:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="活体检测自动化诊断工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  uv run python scripts/workflow/auto_diagnose.py --video-dir 15
  
  # 指定动作
  uv run python scripts/workflow/auto_diagnose.py --video-dir 15 --actions blink mouth_open
  
  # 不保存 CSV
  uv run python scripts/workflow/auto_diagnose.py --video-dir 15 --no-csv
  
  # 仅生成报告
  uv run python scripts/workflow/auto_diagnose.py --video-dir 15 --report-only
        """,
    )

    parser.add_argument("--video-dir", required=True, help="视频目录")
    parser.add_argument(
        "--action-file",
        default=None,
        help="动作配置文件（默认：<video-dir>/动作对应.txt）",
    )
    parser.add_argument("--actions", nargs="+", default=None, help="指定测试动作列表")
    parser.add_argument("--no-csv", action="store_true", help="不保存逐帧 CSV")
    parser.add_argument("--no-report", action="store_true", help="不生成诊断报告")
    parser.add_argument(
        "--report-only", action="store_true", help="仅生成报告（不重新测试）"
    )

    args = parser.parse_args()

    run_workflow(
        video_dir=args.video_dir,
        action_file=args.action_file,
        save_csv=not args.no_csv,
        generate_report=not args.no_report and not args.report_only,
    )


if __name__ == "__main__":
    main()
