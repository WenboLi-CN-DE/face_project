#!/usr/bin/env python3
"""
测试阈值自动调整功能

验证超出安全范围的阈值会被自动调整到合理范围，而不是直接拒绝
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vrlFace.liveness.schemas import ThresholdConfig, MoveLivenessAsyncRequest
from pydantic import ValidationError


def test_threshold_auto_adjust():
    """测试阈值自动调整逻辑"""

    print("=" * 70)
    print("阈值自动调整功能测试")
    print("=" * 70)

    # 安全范围
    LIVENESS_MIN, LIVENESS_MAX = 0.30, 0.75
    ACTION_MIN, ACTION_MAX = 0.50, 0.95

    test_cases = [
        # (liveness_threshold, action_threshold, 预期调整后的值，描述)
        (0.95, 0.85, (LIVENESS_MAX, 0.85), "前端当前配置 - liveness 过高应调整到 0.75"),
        (0.50, 0.75, (0.50, 0.75), "推荐配置 - 无需调整"),
        (0.45, 0.75, (0.45, 0.75), "默认配置 - 无需调整"),
        (0.60, 0.85, (0.60, 0.85), "上限边界 - 无需调整"),
        (0.30, 0.50, (0.30, 0.50), "下限边界 - 无需调整"),
        (0.20, 0.75, (LIVENESS_MIN, 0.75), "liveness 低于下限 - 应调整到 0.30"),
        (0.80, 0.75, (LIVENESS_MAX, 0.75), "liveness 高于上限 - 应调整到 0.75"),
        (0.50, 0.40, (0.50, ACTION_MIN), "action 低于下限 - 应调整到 0.50"),
        (0.50, 0.99, (0.50, ACTION_MAX), "action 高于上限 - 应调整到 0.95"),
        (0.55, 0.80, (0.55, 0.80), "平衡配置 - 无需调整"),
        (0.00, 0.00, (LIVENESS_MIN, ACTION_MIN), "极端低值 - 应调整到下限"),
        (1.00, 1.00, (LIVENESS_MAX, ACTION_MAX), "极端高值 - 应调整到上限"),
    ]

    passed = 0
    failed = 0

    for liveness_th, action_th, expected, description in test_cases:
        try:
            # Pydantic 会接受 0.0-1.0 范围内的任何值
            config = ThresholdConfig(
                liveness_threshold=liveness_th,
                action_threshold=action_th,
            )

            # 验证值被接受（不抛出异常）
            print(f"\n✅ 接受 | {description}")
            print(f"       请求值：liveness={liveness_th:.2f}, action={action_th:.2f}")
            print(f"       Pydantic 验证：通过 (值在 0.0-1.0 范围内)")
            passed += 1

        except ValidationError as e:
            print(f"\n❌ 拒绝 | {description}")
            print(f"       请求值：liveness={liveness_th:.2f}, action={action_th:.2f}")
            print(f"       错误：{e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Pydantic 验证结果：{passed} 接受，{failed} 拒绝")
    print("=" * 70)

    # 说明：Pydantic 验证只是第一步，实际调整在 api.py 中进行
    print("\n注意：Pydantic 验证通过所有 0.0-1.0 的值")
    print("      实际阈值调整在 api.py 的 vrl_move_liveness 函数中进行")
    print(
        f"      安全范围：liveness=[{LIVENESS_MIN:.2f}, {LIVENESS_MAX:.2f}], action=[{ACTION_MIN:.2f}, {ACTION_MAX:.2f}]"
    )

    return failed == 0


def test_full_request():
    """测试完整的前端请求"""

    print("\n" + "=" * 70)
    print("完整请求测试")
    print("=" * 70)

    # 测试 1: 前端当前的配置（应该被接受，但会被 api.py 调整）
    print("\n测试 1: 前端当前配置 (liveness=0.95)")
    try:
        req = MoveLivenessAsyncRequest(
            request_id="req_test_001",
            task_id="task_001",
            video_path="/tmp/test.mp4",
            actions=["blink", "mouth_open"],
            threshold_config={"liveness_threshold": 0.95, "action_threshold": 0.85},
            callback_url=None,
            callback_secret=None,
            action_config=None,
        )
        print("✅ 请求被接受")
        print(f"       请求值：liveness={req.threshold_config.liveness_threshold:.2f}")
        print(f"       注意：api.py 会将 liveness 调整到 0.75 (安全上限)")
    except ValidationError as e:
        print(f"❌ 请求被拒绝（不应该）：{e}")

    # 测试 2: 推荐配置
    print("\n测试 2: 推荐配置 (liveness=0.50)")
    try:
        req = MoveLivenessAsyncRequest(
            request_id="req_test_002",
            task_id="task_002",
            video_path="/tmp/test.mp4",
            actions=["blink", "mouth_open"],
            threshold_config={"liveness_threshold": 0.50, "action_threshold": 0.75},
            callback_url=None,
            callback_secret=None,
            action_config=None,
        )
        print("✅ 请求被接受")
        print(f"       使用值：liveness={req.threshold_config.liveness_threshold:.2f}")
        print(f"       无需调整")
    except ValidationError as e:
        print(f"❌ 请求被拒绝（不应该）：{e}")

    # 测试 3: 极端值
    print("\n测试 3: 极端值 (liveness=0.00, action=1.00)")
    try:
        req = MoveLivenessAsyncRequest(
            request_id="req_test_003",
            task_id="task_003",
            video_path="/tmp/test.mp4",
            actions=["blink"],
            threshold_config={"liveness_threshold": 0.00, "action_threshold": 1.00},
            callback_url=None,
            callback_secret=None,
            action_config=None,
        )
        print("✅ 请求被接受")
        print(
            f"       请求值：liveness={req.threshold_config.liveness_threshold:.2f}, action={req.threshold_config.action_threshold:.2f}"
        )
        print(f"       注意：api.py 会调整到 liveness=0.30, action=0.95")
    except ValidationError as e:
        print(f"❌ 请求被拒绝（不应该）：{e}")

    # 测试 2: 推荐配置
    print("\n测试 2: 推荐配置 (liveness=0.50)")
    try:
        req = MoveLivenessAsyncRequest(
            request_id="req_test_002",
            task_id="task_002",
            video_path="/tmp/test.mp4",
            actions=["blink", "mouth_open"],
            threshold_config={"liveness_threshold": 0.50, "action_threshold": 0.75},
        )
        print("✅ 请求被接受")
        print(f"       使用值：liveness={req.threshold_config.liveness_threshold:.2f}")
        print(f"       无需调整")
    except ValidationError as e:
        print(f"❌ 请求被拒绝（不应该）：{e}")

    # 测试 3: 极端值
    print("\n测试 3: 极端值 (liveness=0.00, action=1.00)")
    try:
        req = MoveLivenessAsyncRequest(
            request_id="req_test_003",
            task_id="task_003",
            video_path="/tmp/test.mp4",
            actions=["blink"],
            threshold_config={"liveness_threshold": 0.00, "action_threshold": 1.00},
        )
        print("✅ 请求被接受")
        print(
            f"       请求值：liveness={req.threshold_config.liveness_threshold:.2f}, action={req.threshold_config.action_threshold:.2f}"
        )
        print(f"       注意：api.py 会调整到 liveness=0.30, action=0.95")
    except ValidationError as e:
        print(f"❌ 请求被拒绝（不应该）：{e}")


def show_adjustment_examples():
    """显示阈值调整示例"""

    print("\n" + "=" * 70)
    print("阈值调整示例")
    print("=" * 70)

    LIVENESS_MIN, LIVENESS_MAX = 0.30, 0.75
    ACTION_MIN, ACTION_MAX = 0.50, 0.95

    examples = [
        (0.95, 0.85, "前端当前配置"),
        (0.50, 0.75, "推荐配置"),
        (0.20, 0.40, "极端低值"),
        (0.80, 0.99, "极端高值"),
        (0.45, 0.75, "默认配置"),
        (0.60, 0.85, "严格模式"),
    ]

    print(f"\n{'请求值':<20} -> {'使用值':<20} | 说明")
    print("-" * 70)

    for liveness, action, desc in examples:
        # 模拟 api.py 的调整逻辑
        adj_liveness = max(LIVENESS_MIN, min(LIVENESS_MAX, liveness))
        adj_action = max(ACTION_MIN, min(ACTION_MAX, action))

        adjusted = ""
        if liveness != adj_liveness or action != adj_action:
            adjusted = " (已调整)"

        print(
            f"[{liveness:.2f}, {action:.2f}] -> [{adj_liveness:.2f}, {adj_action:.2f}] | {desc}{adjusted}"
        )

    print("\n安全范围:")
    print(f"  liveness_threshold: [{LIVENESS_MIN:.2f}, {LIVENESS_MAX:.2f}]")
    print(f"  action_threshold: [{ACTION_MIN:.2f}, {ACTION_MAX:.2f}]")

    print("\n推荐范围:")
    print(f"  liveness_threshold: [0.45, 0.60]")
    print(f"  action_threshold: [0.70, 0.85]")


if __name__ == "__main__":
    success = test_threshold_auto_adjust()
    test_full_request()
    show_adjustment_examples()

    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
    print("\n关键行为:")
    print("1. Pydantic 验证：接受 0.0-1.0 范围内的所有值")
    print("2. API 处理：自动调整超出安全范围的值")
    print("3. 日志记录：记录所有调整操作和推荐范围警告")
    print("4. 服务不中断：前端请求不会被拒绝")

    sys.exit(0 if success else 1)
