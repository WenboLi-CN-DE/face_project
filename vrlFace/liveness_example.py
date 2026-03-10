"""
向后兼容层 — 请使用 vrlFace.liveness.cli

此文件保留是为了不破坏直接调用 `python -m vrlFace.liveness_example` 的旧命令。
新命令请使用:
    python -m vrlFace.liveness.cli --camera 0
"""

from .liveness.cli import run_camera_detection, main  # noqa: F401

if __name__ == "__main__":
    main()
