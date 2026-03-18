#!/usr/bin/env python3
"""
SSH 配置管理器 - 从配置文件中读取 SSH 连接信息
"""

import configparser
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SSHConfig:
    """SSH 连接配置"""

    name: str
    host: str
    user: str
    pem_key: str
    port: int = 22
    remote_log: Optional[str] = None
    remote_video_dir: Optional[str] = None
    # Docker 日志配置
    docker_container: Optional[str] = None
    docker_log_output: Optional[str] = None


class SSHConfigManager:
    """SSH 配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认使用项目根目录的 ssh-config.txt
        """
        if config_path is None:
            # 默认使用项目根目录的配置文件
            project_root = Path(__file__).parent.parent
            config_path = str(project_root / "ssh-config.txt")

        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            return  # 配置文件不存在时不报错，返回空配置

        self.config.read(self.config_path, encoding="utf-8")

    def get_config(self, name: str) -> Optional[SSHConfig]:
        """
        获取指定名称的 SSH 配置

        Args:
            name: 配置名称（配置文件中的 section 名称）

        Returns:
            SSHConfig 对象，如果配置不存在则返回 None
        """
        if name not in self.config:
            return None

        section = self.config[name]

        return SSHConfig(
            name=name,
            host=section.get("host", ""),
            user=section.get("user", ""),
            pem_key=section.get("pem-key", ""),
            port=section.getint("port", 22),
            remote_log=section.get("remote-log"),
            remote_video_dir=section.get("remote-video-dir"),
            docker_container=section.get("docker-container"),
            docker_log_output=section.get("docker-log-output"),
        )

    def list_configs(self) -> list[str]:
        """列出所有可用的配置名称"""
        return list(self.config.sections())

    def get_all_configs(self) -> Dict[str, SSHConfig]:
        """获取所有配置"""
        result = {}
        for name in self.config.sections():
            config = self.get_config(name)
            if config:
                result[name] = config
        return result


def get_ssh_config(name: str, config_path: Optional[str] = None) -> Optional[SSHConfig]:
    """便捷函数：获取指定名称的 SSH 配置"""
    manager = SSHConfigManager(config_path)
    return manager.get_config(name)


if __name__ == "__main__":
    import sys

    # 测试配置管理
    manager = SSHConfigManager()

    configs = manager.list_configs()
    if not configs:
        print(f"配置文件中没有发现任何配置")
        print(f"配置文件路径：{manager.config_path}")
        sys.exit(0)

    print(f"可用的 SSH 配置 ({len(configs)} 个):")
    print("-" * 40)

    for name in configs:
        config = manager.get_config(name)
        print(f"\n[{name}]")
        print(f"  Host: {config.host}")
        print(f"  User: {config.user}")
        print(f"  Port: {config.port}")
        print(f"  PEM Key: {config.pem_key}")
        print(f"  Remote Log: {config.remote_log or 'N/A'}")
        print(f"  Remote Video Dir: {config.remote_video_dir or 'N/A'}")
        if config.docker_container:
            print(f"  Docker Container: {config.docker_container}")
            print(f"  Docker Log Output: {config.docker_log_output or 'N/A'}")
