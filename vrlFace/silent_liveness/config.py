"""
静默活体检测配置 - 支持多路径映射

环境变量:
    SILENT_PICTURE_PATHS: 路径映射配置，格式：外部路径=内部路径，多个路径用分号分隔
    示例：/opt/test=/data/videos;/opt/test2026=/data/videos

    SILENT_ALLOWED_PATH_PREFIXES: 允许的路径前缀列表，用逗号分隔
    示例：/data/videos,/opt/test,/opt/test2026
"""

import os
from typing import Dict, List, Optional


class SilentConfig:
    """静默活体检测配置类"""

    # 默认允许的文件扩展名
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 默认路径映射（外部路径 -> 内部路径）
    # 注意：更具体的路径必须放在前面（优先匹配）
    DEFAULT_PATH_MAPPING: Dict[str, str] = {
        "/opt/test2026": "/data/videos",  # 更具体，优先
        "/opt/test": "/data/videos",
    }

    # 默认允许的路径前缀
    DEFAULT_ALLOWED_PREFIXES: List[str] = [
        "/data/videos",
        "/opt/test",
        "/opt/test2026",
    ]

    def __init__(self) -> None:
        self._path_mapping: Dict[str, str] = {}
        self._allowed_prefixes: List[str] = []
        self._load_from_env()

    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        # 加载路径映射
        path_mapping_env = os.getenv("SILENT_PICTURE_PATHS", "")
        if path_mapping_env:
            for mapping in path_mapping_env.split(";"):
                if "=" in mapping:
                    external, internal = mapping.split("=", 1)
                    external = external.strip()
                    internal = internal.strip()
                    if external and internal:
                        self._path_mapping[external] = internal

        # 合并默认映射
        self._path_mapping.update(self.DEFAULT_PATH_MAPPING)

        # 加载允许的路径前缀
        allowed_prefixes_env = os.getenv("SILENT_ALLOWED_PATH_PREFIXES", "")
        if allowed_prefixes_env:
            self._allowed_prefixes = [
                p.strip() for p in allowed_prefixes_env.split(",") if p.strip()
            ]
        else:
            self._allowed_prefixes = self.DEFAULT_ALLOWED_PREFIXES.copy()

        # 确保映射的内部路径也在允许列表中
        for internal_path in self._path_mapping.values():
            if internal_path not in self._allowed_prefixes:
                self._allowed_prefixes.append(internal_path)

    def resolve_path(self, picture_path: str) -> str:
        """
        解析路径 - 将外部路径映射到内部路径

        Args:
            picture_path: 客户端传入的图片路径

        Returns:
            解析后的实际路径（容器内路径）
        """
        # 检查是否匹配任何路径映射
        for external_prefix, internal_prefix in self._path_mapping.items():
            if picture_path.startswith(external_prefix):
                resolved_path = picture_path.replace(
                    external_prefix, internal_prefix, 1
                )
                return resolved_path

        # 无映射，返回原路径
        return picture_path

    def is_path_allowed(self, picture_path: str) -> bool:
        """
        检查路径是否在允许的前缀列表中

        Args:
            picture_path: 要检查的路径

        Returns:
            True=允许，False=拒绝
        """
        # 检查是否匹配任何允许的前缀
        for prefix in self._allowed_prefixes:
            if picture_path.startswith(prefix):
                return True
        return False

    @property
    def path_mapping(self) -> Dict[str, str]:
        """获取路径映射配置"""
        return self._path_mapping.copy()

    @property
    def allowed_prefixes(self) -> List[str]:
        """获取允许的路径前缀列表"""
        return self._allowed_prefixes.copy()


# 全局配置实例（单例）
_config: Optional[SilentConfig] = None


def get_config() -> SilentConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = SilentConfig()
    return _config
