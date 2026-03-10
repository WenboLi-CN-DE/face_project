"""
向后兼容层 — 测试已移动到 tests/liveness/test_liveness.py

运行测试请使用:
    pytest tests/liveness/test_liveness.py -v
"""

import sys
from pathlib import Path

# 兼容旧的直接运行方式
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest",
                    "tests/liveness/test_liveness.py", "-v"])
