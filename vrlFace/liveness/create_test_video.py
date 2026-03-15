"""
生成测试视频 - 用于基准帧校准功能测试

生成一个模拟的人脸视频（使用色块代替）
"""

import cv2
import numpy as np


def create_test_video(
    output_path: str = "test_video.mp4", duration: int = 10, fps: int = 30
):
    """生成测试视频"""
    print(f"生成测试视频：{output_path}")
    print(f"时长：{duration}秒 @ {fps}fps = {duration * fps} 帧")

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration * fps

    for i in range(total_frames):
        # 创建背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 模拟人脸区域（前 2 秒静止，后面移动）
        t = i / fps

        if t < 2.0:
            # 前 2 秒：静止（基准采集阶段）
            center_x, center_y = width // 2, height // 2
            size = 100
        else:
            # 2 秒后：轻微移动（模拟真实场景）
            phase = (t - 2.0) * 0.5
            center_x = width // 2 + int(np.sin(phase) * 30)
            center_y = height // 2 + int(np.cos(phase * 0.7) * 20)
            size = 100 + int(np.sin(phase * 1.3) * 10)

        # 绘制"人脸"（椭圆形）
        cv2.ellipse(
            frame,
            (center_x, center_y),
            (size, int(size * 1.3)),
            0,
            0,
            360,
            (100, 100, 200),
            -1,
        )

        # 绘制"眼睛"
        eye_offset = int(size * 0.3)
        cv2.circle(
            frame, (center_x - eye_offset, center_y - int(size * 0.3)), 8, (0, 0, 0), -1
        )
        cv2.circle(
            frame, (center_x + eye_offset, center_y - int(size * 0.3)), 8, (0, 0, 0), -1
        )

        # 绘制"嘴巴"（开合动作）
        mouth_y = center_y + int(size * 0.5)
        if i % 60 < 30:  # 每 2 秒开合一次
            mouth_height = 5 + int(np.sin(i * 0.2) * 5)
            cv2.ellipse(
                frame, (center_x, mouth_y), (15, mouth_height), 0, 0, 360, (0, 0, 0), 2
            )
        else:
            cv2.line(
                frame, (center_x - 15, mouth_y), (center_x + 15, mouth_y), (0, 0, 0), 2
            )

        # 显示帧信息
        cv2.putText(
            frame,
            f"Frame: {i + 1}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Time: {t:.1f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if t < 2.0:
            cv2.putText(
                frame,
                "[基准采集阶段]",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "[身份验证阶段]",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        out.write(frame)

        # 进度显示
        if (i + 1) % 30 == 0:
            print(
                f"  进度：{i + 1}/{total_frames} ({(i + 1) / total_frames * 100:.1f}%)"
            )

    out.release()
    print(f"✅ 测试视频生成完成：{output_path}")


if __name__ == "__main__":
    create_test_video("test_benchmark.mp4", duration=10, fps=30)
    print("\n使用方式:")
    print(
        "  uv run python -m vrlFace.liveness.benchmark_demo --video test_benchmark.mp4"
    )
    print(
        "  uv run python -m vrlFace.liveness.benchmark_test --video test_benchmark.mp4"
    )
