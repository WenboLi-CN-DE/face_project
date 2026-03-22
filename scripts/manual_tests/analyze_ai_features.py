"""
AI 生成图像特征分析工具

用于分析真实图像和 AI 生成图像的频域特征差异，帮助调优检测阈值。

使用方法:
    uv run python scripts/manual_tests/analyze_ai_features.py \
        --real path/to/real/image.jpg \
        --ai path/to/ai/image.jpg
"""

import sys
import os
import argparse
import json

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import numpy as np
from vrlFace.silent_liveness.frequency_analyzer import FrequencyAnalyzer
from vrlFace.silent_liveness.detector import SilentLivenessDetector


def analyze_image(analyzer: FrequencyAnalyzer, image_path: str, label: str):
    """分析单张图像的特征"""
    print(f"\n{'=' * 70}")
    print(f"{label}: {image_path}")
    print(f"{'=' * 70}")

    if not os.path.exists(image_path):
        print(f"❌ 文件不存在：{image_path}")
        return None

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像：{image_path}")
        return None

    print(f"图像尺寸：{image.shape[1]}x{image.shape[0]}")

    # 使用完整检测器（需要人脸）
    detector = SilentLivenessDetector.get_instance()
    faces = detector._face_detector.detect(image)

    if not faces:
        print("⚠️  未检测到人脸，使用整图分析")
        h, w = image.shape[:2]
        face_bbox = (0, 0, w, h)
    else:
        face = faces[0]
        face_bbox = face.bbox
        print(f"人脸位置：{face_bbox}")

    # 频域分析
    result = analyzer.analyze(image, face_bbox)

    print(f"\n📊 频域特征分析结果:")
    print(f"  anomaly_score:     {result['anomaly_score']:.4f}")
    print(f"  is_ai_generated:   {result['is_ai_generated']}")
    print(f"  confidence:        {result['confidence']:.4f}")
    print(f"\n  各维度分数:")
    features = result["frequency_features"]
    print(
        f"    DCT 分数：        {features.get('dct_score', 0):.4f} (高频比：{features.get('dct_high_freq_ratio', 0):.4f})"
    )
    print(
        f"    FFT 分数：        {features.get('fft_score', 0):.4f} (径向比：{features.get('fft_radial_ratio', 0):.4f})"
    )
    print(
        f"    梯度分数：        {features.get('gradient_score', 0):.4f} (熵：{features.get('gradient_entropy', 0):.4f})"
    )
    print(
        f"    颜色分数：        {features.get('color_score', 0):.4f} (方差比：{features.get('color_variance', 0):.4f})"
    )

    # 判定建议
    print(f"\n💡 分析:")
    if result["anomaly_score"] < 0.3:
        print("  → 特征接近真实图像")
    elif result["anomaly_score"] < 0.5:
        print("  → 特征模糊，需要结合其他检测")
    else:
        print("  → 特征异常，疑似 AI 生成")

    return result


def compare_features(real_result, ai_result):
    """对比真实和 AI 图像的特征差异"""
    if not real_result or not ai_result:
        return

    print(f"\n{'=' * 70}")
    print("📈 特征对比分析")
    print(f"{'=' * 70}")

    real_feat = real_result["frequency_features"]
    ai_feat = ai_result["frequency_features"]

    features_to_compare = [
        ("dct_score", "DCT 分数"),
        ("fft_score", "FFT 分数"),
        ("gradient_score", "梯度分数"),
        ("color_score", "颜色分数"),
        ("dct_high_freq_ratio", "DCT 高频比"),
        ("fft_radial_ratio", "FFT 径向比"),
        ("gradient_entropy", "梯度熵"),
        ("color_variance", "颜色方差比"),
    ]

    print(f"\n{'特征':<20} {'真实图像':<15} {'AI 生成':<15} {'差异':<10}")
    print("-" * 60)

    for key, name in features_to_compare:
        real_val = real_feat.get(key, 0)
        ai_val = ai_feat.get(key, 0)
        diff = ai_val - real_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{name:<20} {real_val:<15.4f} {ai_val:<15.4f} {diff_str:<10}")

    print(f"\n{'=' * 70}")
    print(
        f"anomaly_score:  真实={real_result['anomaly_score']:.4f}  AI={ai_result['anomaly_score']:.4f}"
    )
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="AI 生成图像特征分析工具")
    parser.add_argument("--real", type=str, help="真实图像路径")
    parser.add_argument("--ai", type=str, help="AI 生成图像路径")
    parser.add_argument("--batch", type=str, help="批量分析目录")
    args = parser.parse_args()

    analyzer = FrequencyAnalyzer.get_instance()

    if args.batch:
        # 批量分析
        print(f"批量分析目录：{args.batch}")
        results = []

        for filename in sorted(os.listdir(args.batch)):
            if filename.endswith((".jpg", ".png", ".jpeg", ".webp")):
                filepath = os.path.join(args.batch, filename)
                result = analyze_image(analyzer, filepath, filename)
                if result:
                    results.append(
                        {
                            "filename": filename,
                            "anomaly_score": result["anomaly_score"],
                            "is_ai": result["is_ai_generated"],
                            "features": result["frequency_features"],
                        }
                    )

        # 输出统计
        print(f"\n{'=' * 70}")
        print("📊 批量分析统计")
        print(f"{'=' * 70}")
        print(f"总图片数：{len(results)}")

        if results:
            scores = [r["anomaly_score"] for r in results]
            ai_count = sum(1 for r in results if r["is_ai"])
            print(f"AI 生成疑似：{ai_count} ({ai_count / len(results) * 100:.1f}%)")
            print(f"平均 anomaly_score: {np.mean(scores):.4f}")
            print(f"最高分：{max(scores):.4f}")
            print(f"最低分：{min(scores):.4f}")

            # 输出 JSON
            output_file = "output/ai_analysis_results.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存到：{output_file}")

    elif args.real and args.ai:
        # 对比分析
        real_result = analyze_image(analyzer, args.real, "真实图像")
        ai_result = analyze_image(analyzer, args.ai, "AI 生成图像")
        compare_features(real_result, ai_result)

    else:
        parser.print_help()
        print("\n示例:")
        print("  # 对比分析")
        print("  uv run python scripts/manual_tests/analyze_ai_features.py \\")
        print("      --real data/test/real_face.jpg \\")
        print("      --ai data/test/ai_face.jpg")
        print("")
        print("  # 批量分析")
        print("  uv run python python scripts/manual_tests/analyze_ai_features.py \\")
        print("      --batch output/remote_fetch/images/")


if __name__ == "__main__":
    main()
