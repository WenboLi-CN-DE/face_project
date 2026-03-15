"""
向后兼容层 — 请使用 vrlFace.liveness.recorder

此文件保留是为了不破坏直接调用 `python -m vrlFace.save_csv` 的旧命令。
新命令请使用:
    python -m vrlFace.liveness.recorder --video path/to/video.mp4
"""

from .liveness.recorder import (  # noqa: F401
    run_video_detection_with_csv,
    main,
)

if __name__ == "__main__":
    main()

