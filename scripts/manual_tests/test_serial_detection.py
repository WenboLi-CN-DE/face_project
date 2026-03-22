"""
串行双检测方案测试脚本

测试用例：
1. 真实人脸照片 → 应该通过
2. 打印照片攻击 → 应该被 UniFace 拦截 (traditional_spoof)
3. AI 生成人脸 → 应该被频域分析拦截 (ai_spoof)
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrlFace.silent_liveness.detector import SilentLivenessDetector


def test_real_face():
    """测试真实人脸"""
    print("\n" + "=" * 60)
    print("测试用例 1: 真实人脸照片")
    print("=" * 60)

    detector = SilentLivenessDetector.get_instance()
    result = detector.detect("data/test/real_face.jpg")

    print(f"is_liveness: {result['is_liveness']}")
    print(f"confidence: {result['confidence']}")
    print(f"reject_reason: {result['reject_reason']}")
    print(f"details: {result['details']}")

    if result["is_liveness"] == 1 and result["reject_reason"] is None:
        print("✅ 通过：正确识别为真人")
    else:
        print("❌ 失败：真实人脸被误判")

    return result


def test_print_attack():
    """测试打印照片攻击"""
    print("\n" + "=" * 60)
    print("测试用例 2: 打印照片攻击")
    print("=" * 60)

    detector = SilentLivenessDetector.get_instance()
    result = detector.detect("data/test/print_attack.jpg")

    print(f"is_liveness: {result['is_liveness']}")
    print(f"confidence: {result['confidence']}")
    print(f"reject_reason: {result['reject_reason']}")
    print(f"details: {result['details']}")

    if result["is_liveness"] == 0 and result["reject_reason"] == "traditional_spoof":
        print("✅ 通过：正确拦截传统攻击")
    else:
        print("❌ 失败：打印照片未被识别")

    return result


def test_ai_generated():
    """测试 AI 生成人脸"""
    print("\n" + "=" * 60)
    print("测试用例 3: AI 生成人脸（StyleGAN/Midjourney）")
    print("=" * 60)

    detector = SilentLivenessDetector.get_instance()
    result = detector.detect("data/test/ai_face.jpg")

    print(f"is_liveness: {result['is_liveness']}")
    print(f"confidence: {result['confidence']}")
    print(f"reject_reason: {result['reject_reason']}")
    print(f"details: {result['details']}")

    if result["is_liveness"] == 0 and result["reject_reason"] == "ai_spoof":
        print("✅ 通过：正确拦截 AI 生成图像")
    else:
        print("⚠️  注意：AI 生成图像未被识别（可能需要调整阈值）")

    return result


def test_no_face():
    """测试无人脸图片"""
    print("\n" + "=" * 60)
    print("测试用例 4: 无人脸图片")
    print("=" * 60)

    detector = SilentLivenessDetector.get_instance()
    result = detector.detect("data/test/no_face.jpg")

    print(f"is_liveness: {result['is_liveness']}")
    print(f"is_face_exist: {result['is_face_exist']}")
    print(f"reject_reason: {result['reject_reason']}")

    if result["is_face_exist"] == 0 and result["reject_reason"] == "no_face":
        print("✅ 通过：正确检测无人脸")
    else:
        print("❌ 失败：无人脸检测逻辑异常")

    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("串行双检测方案测试")
    print("UniFace MiniFASNet + 频域分析器")
    print("=" * 60)

    # 检查测试图片是否存在
    test_images = {
        "real_face.jpg": "真实人脸",
        "print_attack.jpg": "打印攻击",
        "ai_face.jpg": "AI 生成人脸",
        "no_face.jpg": "无人脸",
    }

    print("\n测试图片检查:")
    for img, desc in test_images.items():
        path = f"data/test/{img}"
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {desc}: {path}")

    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    try:
        # 运行测试（如果图片存在）
        if os.path.exists("data/test/real_face.jpg"):
            test_real_face()

        if os.path.exists("data/test/print_attack.jpg"):
            test_print_attack()

        if os.path.exists("data/test/ai_face.jpg"):
            test_ai_generated()

        if os.path.exists("data/test/no_face.jpg"):
            test_no_face()

    except Exception as e:
        print(f"\n❌ 测试出错：{e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
