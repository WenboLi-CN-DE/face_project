"""
静默活体检测配置

环境变量:
    SILENT_ALLOWED_PATH_PREFIXES: 允许的路径前缀列表，用逗号分隔
    示例：/data/videos,/opt/test,/opt/test2026
"""

import os
from typing import List, Optional


class SilentConfig:
    """静默活体检测配置类"""

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    DEFAULT_ALLOWED_PREFIXES: List[str] = [
        "/data/videos",
        "/opt/test",
        "/opt/test2026",
    ]

    def __init__(self) -> None:
        self._allowed_prefixes: List[str] = []
        self._load_from_env()

    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        allowed_prefixes_env = os.getenv("SILENT_ALLOWED_PATH_PREFIXES", "")
        if allowed_prefixes_env:
            self._allowed_prefixes = [
                p.strip() for p in allowed_prefixes_env.split(",") if p.strip()
            ]
        else:
            self._allowed_prefixes = self.DEFAULT_ALLOWED_PREFIXES.copy()

    def is_path_allowed(self, picture_path: str) -> bool:
        for prefix in self._allowed_prefixes:
            if picture_path.startswith(prefix):
                return True
        return False

    @property
    def allowed_prefixes(self) -> List[str]:
        return self._allowed_prefixes.copy()


_config: Optional[SilentConfig] = None


def get_config() -> SilentConfig:
    global _config
    if _config is None:
        _config = SilentConfig()
    return _config
