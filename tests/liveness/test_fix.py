"""
测试修复后的阈值配置效果

运行:
    uv run python tests/liveness/test_fix.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.schemas import ThresholdConfig
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_default_config():
    """测试修复后的默认配置"""
    print("\n" + "=" * 70)
    print("测试默认阈值配置 (修复后)")
    print("=" * 70)

    # 检查 schemas 中的默认值
    threshold_config = ThresholdConfig()
    print(f"\nThresholdConfig 默认值:")
    print(f"  liveness_threshold: {threshold_config.liveness_threshold}")
    print(f"  action_threshold: {threshold_config.action_threshold}")

    # 检查 config 中的默认值
    config = LivenessConfig.video_fast_config()
    print(f"\nLivenessConfig.video_fast_config():")
    print(f"  threshold: {config.threshold}")
    print(f"  ear_threshold: {config.ear_threshold}")
    print(f"  mar_threshold: {config.mar_threshold}")
    print(f"  yaw_threshold: {config.yaw_threshold}")
    print(f"  pitch_threshold: {config.pitch_threshold}")


def test_videos():
    """测试 15 文件夹中的视频"""
    print("\n" + "=" * 70)
    print("使用修复后配置测试视频")
    print("=" * 70)

    video_folder = Path(__file__).parent.parent.parent / "15"
    if not video_folder.exists():
        logger.error(f"视频文件夹不存在：{video_folder}")
        return

    video_files = list(video_folder.glob("*.webm"))[:5]
    logger.info(f"测试 {len(video_files)} 个视频文件")

    # 使用修复后的默认配置
    config = LivenessConfig.video_fast_config()
    logger.info(f"使用配置：threshold={config.threshold}")

    results = []

    for video_file in video_files:
        # 使用默认配置创建分析器
        analyzer = VideoLivenessAnalyzer(
            liveness_config=config,
            liveness_threshold=0.45,  # 修复后的推荐值
            action_threshold=0.75,  # 修复后的推荐值
        )

        # 测试常见动作组合
        test_actions = [
            ["blink", "mouth_open"],
            ["nod_up", "mouth_open"],
            ["turn_left", "mouth_open"],
        ]

        # 选择一个动作组合测试
        actions = test_actions[video_files.index(video_file) % len(test_actions)]

        try:
            result = analyzer.analyze(
                video_path=str(video_file),
                actions=actions,
            )

            passed_actions = sum(
                1 for d in result.action_verify.action_details if d.passed
            )
            total_actions = len(result.action_verify.action_details)

            results.append(
                {
                    "video": video_file.name[:50],
                    "is_liveness": result.is_liveness,
                    "confidence": result.liveness_confidence,
                    "actions_passed": passed_actions,
                    "actions_total": total_actions,
                }
            )

            status = "✓" if result.is_liveness else "✗"
            print(f"\n{status} {video_file.name[:50]}")
            print(
                f"   活体：{'通过' if result.is_liveness else '失败'} (confidence={result.liveness_confidence:.4f})"
            )
            print(f"   动作：{passed_actions}/{total_actions}")
            for detail in result.action_verify.action_details:
                a_status = "✓" if detail.passed else "✗"
                print(
                    f"     {a_status} {detail.action}: {detail.msg} (conf={detail.confidence:.4f})"
                )

        except Exception as e:
            logger.error(f"测试失败 {video_file.name}: {e}")
            results.append(
                {
                    "video": video_file.name[:50],
                    "is_liveness": 0,
                    "confidence": 0.0,
                    "actions_passed": 0,
                    "actions_total": len(actions),
                    "error": str(e),
                }
            )

    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    liveness_passed = sum(1 for r in results if r["is_liveness"])
    action_passed = sum(r["actions_passed"] for r in results)
    action_total = sum(r["actions_total"] for r in results)

    print(f"视频数量：{len(results)}")
    print(
        f"活体通过率：{liveness_passed}/{len(results)} ({liveness_passed / len(results) * 100:.1f}%)"
    )
    print(
        f"动作通过率：{action_passed}/{action_total} ({action_passed / action_total * 100:.1f}%)"
    )

    print("\n详细结果:")
    print(f"{'视频':<50} {'活体':<8} {'置信度':<10} {'动作':<10}")
    print("-" * 78)
    for r in results:
        status = "通过" if r["is_liveness"] else "失败"
        conf = f"{r['confidence']:.4f}"
        actions = f"{r['actions_passed']}/{r['actions_total']}"
        print(f"{r['video']:<50} {status:<8} {conf:<10} {actions:<10}")


if __name__ == "__main__":
    test_default_config()
    test_videos()
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
