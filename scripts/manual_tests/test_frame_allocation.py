"""
测试帧分配修复效果

使用方法：
    uv run python test_frame_allocation.py --video <视频路径> --actions nod mouth_open

支持的动作：
    blink        - 眨眼
    mouth_open   - 张嘴
    shake_head   - 摇头（左右）
    nod          - 点头（上下）
    nod_down     - 低头
    nod_up       - 抬头
    turn_left    - 向左转头
    turn_right   - 向右转头
"""

import sys
import argparse
from vrlFace.liveness.video_analyzer import VideoLivenessAnalyzer
from vrlFace.liveness.config import LivenessConfig


def test_video_analysis(video_path: str, actions: list):
    print("=" * 60)
    print("测试帧分配修复效果")
    print("=" * 60)
    print(f"视频路径: {video_path}")
    print()

    config = LivenessConfig.video_fast_config()
    analyzer = VideoLivenessAnalyzer(
        liveness_config=config,
        liveness_threshold=0.45,
        action_threshold=0.75,
        enable_benchmark=True,
    )

    print(f"测试动作: {actions}")
    print()

    print("开始分析...")
    result = analyzer.analyze(
        video_path=video_path,
        actions=actions,
    )

    print()
    print("=" * 60)
    print("分析结果")
    print("=" * 60)
    print(f"活体判定: {'✅ 通过' if result.is_liveness else '❌ 不通过'}")
    print(f"活体置信度: {result.liveness_confidence:.2%}")
    print(f"人脸检测: {'✅ 检测到' if result.is_face_exist else '❌ 未检测到'}")
    print()

    print("动作验证结果:")
    print("-" * 60)
    for detail in result.action_verify.action_details:
        status = "✅ 通过" if detail.passed else "❌ 失败"
        print(
            f"  {detail.action:15s} {status:10s} 置信度: {detail.confidence:.2%}  {detail.msg}"
        )
    print()

    print(
        f"总体动作验证: {'✅ 全部通过' if result.action_verify.passed else '❌ 部分失败'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="测试帧分配修复效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的动作：
  blink        - 眨眼
  mouth_open   - 张嘴
  shake_head   - 摇头（左右）
  nod          - 点头（上下）
  nod_down     - 低头
  nod_up       - 抬头
  turn_left    - 向左转头
  turn_right   - 向右转头

示例：
  uv run python test_frame_allocation.py --video test.mp4 --actions nod mouth_open
  uv run python test_frame_allocation.py --video test.mp4 --actions blink shake_head nod
        """,
    )
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["nod", "mouth_open"],
        help="动作列表（默认：nod mouth_open）",
    )
    args = parser.parse_args()

    test_video_analysis(args.video, args.actions)
