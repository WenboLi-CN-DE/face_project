"""
向后兼容层 — 请使用 vrlFace.liveness.cli

此文件保留是为了不破坏直接调用 `python -m vrlFace.liveness_main` 的旧命令。
新命令请使用:
    python -m vrlFace.liveness.cli --camera 0
    python -m vrlFace.liveness.cli --video path/to/video.mp4
"""

from .liveness.cli import (  # noqa: F401
    run_camera_detection,
    run_video_detection,
    main,
)

if __name__ == "__main__":
    main()
