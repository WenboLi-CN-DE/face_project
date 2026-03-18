#!/usr/bin/env python3
"""
测试质量分数修复效果

对比修复前后的质量分数变化
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig


def test_video(video_path: str, expected_actions: list, video_name: str):
    """测试单个视频"""
    print(f"\n{'=' * 60}")
    print(f"测试视频：{video_name}")
    print(f"路径：{video_path}")
    print(f"期望动作：{expected_actions}")
    print(f"{'=' * 60}")

    # 使用修复后的配置：阈值 0.50
    config = LivenessConfig.video_fast_config()
    config.threshold = 0.50  # 降低阈值

    analyzer = VideoLivenessAnalyzer(
        liveness_config=config,
        liveness_threshold=0.50,  # 降低活体阈值
        action_threshold=0.75,
        enable_benchmark=False,  # 禁用基准帧校准，简化测试
    )

    try:
        result = analyzer.analyze(
            video_path=video_path,
            actions=expected_actions,
        )

        print(f"\n【结果】")
        print(f"  活体判定：{'通过' if result.is_liveness == 1 else '失败'}")
        print(f"  活体置信度：{result.liveness_confidence:.4f}")
        print(
            f"  人脸质量分数：{result.face_info.quality_score if result.face_info else 'N/A':.4f}"
        )
        print(
            f"  人脸置信度：{result.face_info.confidence if result.face_info else 'N/A':.4f}"
        )

        print(f"\n【动作详情】")
        for detail in result.action_verify.action_details:
            status = "✓" if detail.passed else "✗"
            print(
                f"  {status} {detail.action}: confidence={detail.confidence:.4f}, msg={detail.msg}"
            )

        return {
            "video": video_name,
            "is_liveness": result.is_liveness,
            "liveness_confidence": result.liveness_confidence,
            "quality_score": result.face_info.quality_score
            if result.face_info
            else 0.0,
            "face_confidence": result.face_info.confidence if result.face_info else 0.0,
            "action_details": [
                {
                    "action": d.action,
                    "passed": d.passed,
                    "confidence": d.confidence,
                }
                for d in result.action_verify.action_details
            ],
            "all_actions_passed": result.action_verify.passed,
        }

    except Exception as e:
        print(f"\n【错误】{e}")
        import traceback

        traceback.print_exc()
        return {
            "video": video_name,
            "error": str(e),
        }


def main():
    """批量测试所有视频"""
    videos_dir = Path(__file__).parent / "videos"

    # 解析动作对应文件
    action_file = Path(__file__).parent / "动作对应.txt"
    video_actions = []

    if action_file.exists():
        ACTION_MAP = {
            "nod": "nod",
            "nod_up": "nod_up",
            "nod_down": "nod_down",
            "shake_head": "shake_head",
            "turn_left": "turn_left",
            "turn_right": "turn_right",
            "blink": "blink",
            "mouth_open": "mouth_open",
        }

        with open(action_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ".webm" not in line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    video_name = parts[0].strip()
                    actions = [ACTION_MAP[a] for a in parts[1:] if a in ACTION_MAP]
                    if actions:
                        video_actions.append((video_name, actions))

    print(f"找到 {len(video_actions)} 个测试视频")
    print(f"测试开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置：threshold=0.50, action_threshold=0.75, enable_benchmark=False")

    results = []
    for video_name, actions in video_actions:
        video_path = videos_dir / video_name
        if not video_path.exists():
            print(f"\n⚠️  视频不存在：{video_path}")
            continue

        result = test_video(str(video_path), actions, video_name)
        results.append(result)

    # 生成汇总报告
    print(f"\n\n{'=' * 60}")
    print(f"【汇总报告】")
    print(f"{'=' * 60}")

    total = len(results)
    passed = sum(1 for r in results if r.get("is_liveness") == 1)
    quality_scores = [
        r.get("quality_score", 0) for r in results if "quality_score" in r
    ]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print(f"测试视频数：{total}")
    print(f"活体通过数：{passed} ({passed / total * 100:.1f}%)")
    print(f"平均质量分数：{avg_quality:.4f}")

    # 保存结果
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "liveness_threshold": 0.50,
            "action_threshold": 0.75,
            "enable_benchmark": False,
        },
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_quality_score": avg_quality,
        },
        "results": results,
    }

    report_path = Path(__file__).parent / "quality_fix_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存：{report_path}")

    # 与之前的结果对比
    prev_report_path = Path(__file__).parent / "test_comparison_report.json"
    if prev_report_path.exists():
        print(f"\n{'=' * 60}")
        print(f"【与修复前对比】")
        print(f"{'=' * 60}")

        with open(prev_report_path, "r", encoding="utf-8") as f:
            prev_data = json.load(f)

        prev_liveness_rate = prev_data["summary"].get("liveness_match_rate", 0)
        curr_liveness_rate = passed / total if total > 0 else 0

        print(f"修复前活体匹配率：{prev_liveness_rate * 100:.1f}%")
        print(f"修复后活体通过率：{curr_liveness_rate * 100:.1f}%")
        print(f"提升：{(curr_liveness_rate - prev_liveness_rate) * 100:+.1f}%")


if __name__ == "__main__":
    main()
