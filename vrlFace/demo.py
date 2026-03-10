"""
向后兼容层 — 请使用 vrlFace.face.cli

此文件保留是为了不破坏直接调用 `python -m vrlFace.demo` 的旧命令。
新命令请使用:
    python -m vrlFace.face.cli --demo
"""

from .face.cli import run_demo, main  # noqa: F401

if __name__ == "__main__":
    run_demo()
