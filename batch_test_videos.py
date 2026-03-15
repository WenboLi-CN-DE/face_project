"""
批量测试视频动作检测

基于 15/动作对应.txt 文件批量测试所有视频
"""

import os
import re
from pathlib import Path
from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig

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


def parse_action_file(file_path: str):
    """解析动作对应文件"""
    video_actions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not ".webm" in line:
                continue

            match = re.match(r"^(.+\.webm)\s+(.+)$", line)
            if not match:
                continue

            video_name = match.group(1).strip()
            actions_str = match.group(2).strip()

            actions = []
            for cn_action, en_action in ACTION_MAP.items():
                if cn_action in actions_str:
                    if en_action not in actions:
                        actions.append(en_action)

            if actions:
                video_actions.append((video_name, actions))

    return video_actions


def test_single_video(video_path: str, expected_actions: list, video_name: str):
    """测试单个视频"""
    config = LivenessConfig.video_fast_config()
    analyzer = VideoLivenessAnalyzer(
        liveness_config=config,
        liveness_threshold=0.45,
        action_threshold=0.75,
        enable_benchmark=True,
    )

    try:
        result = analyzer.analyze(
            video_path=video_path,
            actions=expected_actions,
        )

        return {
            "video": video_name,
            "expected": expected_actions,
            "is_liveness": result.is_liveness,
            "liveness_confidence": result.liveness_confidence,
            "action_details": result.action_verify.action_details,
            "all_passed": result.action_verify.passed,
            "error": None,
        }
    except Exception as e:
        return {"video": video_name, "expected": expected_actions, "error": str(e)}


def main():
    video_dir = Path("15")
    action_file = video_dir / "动作对应.txt"

    if not action_file.exists():
        print(f"❌ 找不到文件: {action_file}")
        return

    print("=" * 80)
    print("批量测试视频动作检测")
    print("=" * 80)
    print()

    video_actions = parse_action_file(str(action_file))
    print(f"找到 {len(video_actions)} 个视频待测试")
    print()

    results = []
    total_videos = len(video_actions)

    for idx, (video_name, actions) in enumerate(video_actions, 1):
        video_path = video_dir / video_name

        if not video_path.exists():
            print(f"[{idx}/{total_videos}] ⚠️  跳过（文件不存在）: {video_name}")
            continue

        print(f"[{idx}/{total_videos}] 测试: {video_name}")
        print(f"  期望动作: {actions}")

        result = test_single_video(str(video_path), actions, video_name)
        results.append(result)

        if result.get("error"):
            print(f"  ❌ 错误: {result['error']}")
        else:
            print(
                f"  活体: {'✅' if result['is_liveness'] else '❌'} ({result['liveness_confidence']:.2%})"
            )
            print(f"  动作验证:")
            for detail in result["action_details"]:
                status = "✅" if detail.passed else "❌"
                print(
                    f"    {status} {detail.action:12s} {detail.confidence:.2%}  {detail.msg}"
                )
        print()

    print("=" * 80)
    print("汇总统计")
    print("=" * 80)

    valid_results = [r for r in results if not r.get("error")]
    if not valid_results:
        print("没有有效的测试结果")
        return

    total_actions = sum(len(r["action_details"]) for r in valid_results)
    passed_actions = sum(
        sum(1 for d in r["action_details"] if d.passed) for r in valid_results
    )

    print(f"测试视频数: {len(valid_results)}/{total_videos}")
    print(f"总动作数: {total_actions}")
    print(f"通过动作数: {passed_actions}")
    print(f"动作通过率: {passed_actions / total_actions * 100:.1f}%")
    print()

    action_stats = {}
    for result in valid_results:
        for detail in result["action_details"]:
            action = detail.action
            if action not in action_stats:
                action_stats[action] = {"total": 0, "passed": 0, "confidences": []}
            action_stats[action]["total"] += 1
            if detail.passed:
                action_stats[action]["passed"] += 1
            action_stats[action]["confidences"].append(detail.confidence)

    print("各动作统计:")
    print("-" * 80)
    for action, stats in sorted(action_stats.items()):
        pass_rate = stats["passed"] / stats["total"] * 100
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"])
        print(
            f"  {action:12s}  通过率: {pass_rate:5.1f}%  ({stats['passed']}/{stats['total']})  平均置信度: {avg_conf:.2%}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
