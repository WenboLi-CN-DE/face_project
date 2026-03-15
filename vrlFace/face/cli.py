"""
人脸识别命令行工具 & 演示脚本

使用示例:
    python -m vrlFace.face.cli
    python -m vrlFace.face.cli --demo
    python -m vrlFace.face.cli --img1 a.jpg --img2 b.jpg
"""

import argparse
import cv2
from pathlib import Path

from .recognizer import face_detection, gen_verify_res
from .config import config


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def draw_face_boxes(img, faces, title=""):
    """在图片上绘制人脸框"""
    result = img.copy()
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        score_text = (
            f"Face {i + 1}: {face.det_score:.2f}"
            if hasattr(face, "det_score")
            else f"Face {i + 1}"
        )
        cv2.putText(
            result, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    if title:
        cv2.putText(
            result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )
    return result


# ---------------------------------------------------------------------------
# 演示函数
# ---------------------------------------------------------------------------

def demo_basic_comparison(data_dir: Path):
    """演示：基础 1:1 人脸比对"""
    print("=" * 60)
    print("演示 1: 基础人脸比对")
    print("=" * 60)

    img1 = data_dir / "t1.jpg"
    img2 = data_dir / "t2.jpg"

    if not img1.exists() or not img2.exists():
        print(f"⚠️  测试图片不存在：{data_dir}")
        return

    print(f"\n图片 1: {img1.name}")
    print(f"图片 2: {img2.name}")

    result = gen_verify_res(str(img1), str(img2))

    print(f"\n结果:")
    print(f"  是否同一人：{'是' if result['is_same_face'] == 1 else '否'}")
    print(f"  置信度：{result['confidence']:.2%}")


def demo_detection(data_dir: Path):
    """演示：人脸检测"""
    print("\n" + "=" * 60)
    print("演示 2: 人脸检测")
    print("=" * 60)

    img_path = data_dir / "t1.jpg"

    if not img_path.exists():
        print("⚠️  测试图片不存在")
        return

    result = face_detection(str(img_path))

    print(f"\n检测结果:")
    print(f"  是否有人脸：{'是' if result['is_face_exist'] else '否'}")
    print(f"  人脸数量：{result['face_num']}")

    if result["face_num"] > 0:
        for i, face in enumerate(result["faces_detected"], 1):
            print(f"  人脸{i}: 置信度 {face['confidence']:.2%}")


def demo_batch(data_dir: Path):
    """演示：批量测试"""
    print("\n" + "=" * 60)
    print("演示 3: 批量测试")
    print("=" * 60)

    test_pairs = [
        ("t1.jpg", "t2.jpg", "同一人测试"),
    ]

    for img1_name, img2_name, desc in test_pairs:
        img1 = data_dir / img1_name
        img2 = data_dir / img2_name

        if img1.exists() and img2.exists():
            result = gen_verify_res(str(img1), str(img2))
            print(f"\n{desc}:")
            print(f"  {img1.name} vs {img2.name}")
            print(f"  结果：{'同一人' if result['is_same_face'] == 1 else '不同人'}")
            print(f"  置信度：{result['confidence']:.2%}")


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def run_compare(img1_path: str, img2_path: str):
    """命令行：执行两张图片比对"""
    print("=" * 60)
    print("人脸识别 — 1:1 比对")
    print("=" * 60)
    print(f"\n配置信息:")
    print(f"  模型：{config.model_name}")
    print(f"  检测尺寸：{config.det_size}")
    print(f"  阈值：{config.similarity_threshold}")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("❌ 无法加载图片")
        return

    print("\n开始人脸比对...")
    try:
        result = gen_verify_res(img1_path, img2_path)

        print(f"\n比对结果:")
        print(f"  是否都有人脸：{'是' if result['is_face_exist'] else '否'}")
        print(f"  是否同一人：{'是' if result['is_same_face'] == 1 else '否'}")
        print(f"  置信度：{result['confidence']:.2%}")
        print(f"  检测结果：{result['detection_result']}")

    except Exception as e:
        print(f"\n❌ 错误：{e}")

    print("\n" + "=" * 60)


def run_demo():
    """运行所有演示"""
    print("\n人脸识别系统演示")
    print(f"模型：{config.model_name}")
    print(f"阈值：{config.similarity_threshold}")
    print()

    # 演示数据目录：vrlFace/face/../../data/  或  vrlFace/data/
    data_dir = Path(__file__).parent.parent / "data"

    demo_basic_comparison(data_dir)
    demo_detection(data_dir)
    demo_batch(data_dir)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="人脸识别命令行工具")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--img1", type=str, help="图片 1 路径")
    parser.add_argument("--img2", type=str, help="图片 2 路径")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.img1 and args.img2:
        run_compare(args.img1, args.img2)
    else:
        run_demo()


if __name__ == "__main__":
    main()


