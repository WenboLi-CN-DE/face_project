#!/usr/bin/env python3
"""
远程视频拉取器 - 通过 SSH 连接从远程服务器下载视频文件

支持通过 task_id 在远程日志中查找视频，并下载到本地用于诊断。
"""

import logging
from pathlib import Path
from typing import Optional

from paramiko import SSHClient, AutoAddPolicy, SFTPClient

from .models import FetchConfig

import sys

script_dir = Path(__file__).parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

from log_parser import LogParser, VideoEntry  # noqa: E402
from ssh_config import get_ssh_config  # noqa: E402

logger = logging.getLogger(__name__)


class RemoteVideoFetcher:
    """远程视频拉取器"""

    def __init__(self, config: FetchConfig):
        """
        初始化拉取器

        Args:
            config: 拉取配置
        """
        self.config = config
        self.ssh: Optional[SSHClient] = None
        self.sftp: Optional[SFTPClient] = None
        self.video_entry: Optional[VideoEntry] = None

    def connect(self) -> None:
        """
        建立 SSH 连接

        Raises:
            Exception: SSH 连接失败时抛出异常
        """
        logger.info(
            f"连接到 {self.config.username}@{self.config.host}:{self.config.port}"
        )

        self.ssh = SSHClient()
        self.ssh.set_missing_host_key_policy(AutoAddPolicy())

        try:
            self.ssh.connect(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.username,
                key_filename=self.config.key_filename,
                timeout=30,
                allow_agent=True,
                look_for_keys=True,
            )
            self.sftp = self.ssh.open_sftp()
            logger.info("✓ SSH 连接成功")
        except Exception as e:
            logger.error(f"SSH 连接失败：{e}")
            raise

    def disconnect(self) -> None:
        """断开 SSH 连接"""
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        logger.info("SSH 连接已断开")

    def find_video_by_task_id(self, task_id: str) -> Optional[VideoEntry]:
        """
        通过 task_id 在远程日志中查找视频

        Args:
            task_id: 任务 ID

        Returns:
            VideoEntry 对象，如果未找到则返回 None
        """
        logger.info(f"查找 task_id={task_id} 的视频")

        if not self.sftp:
            raise RuntimeError("SSH 未连接，请先调用 connect()")

        temp_log = Path(self.config.output_dir) / "temp_fetch.log"
        temp_log.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.sftp.get(self.config.remote_log_path, str(temp_log))
            logger.info(f"✓ 日志已下载到 {temp_log}")

            parser = LogParser()
            entries = parser.parse_file(str(temp_log))

            for entry in entries:
                if entry.task_id == task_id:
                    self.video_entry = entry
                    logger.info(f"✓ 找到视频：{entry.video_filename}")
                    return entry

            logger.warning(f"未找到 task_id={task_id} 的视频条目")
            return None

        except FileNotFoundError:
            logger.error(f"远程日志文件不存在：{self.config.remote_log_path}")
            raise
        except Exception as e:
            logger.error(f"查找视频失败：{e}")
            raise
        finally:
            if temp_log.exists():
                temp_log.unlink()
                logger.debug(f"已清理临时文件：{temp_log}")

    def download_video(self, remote_path: str, local_path: str) -> bool:
        """
        下载单个视频文件（支持断点续传）

        Args:
            remote_path: 远程文件路径
            local_path: 本地保存路径

        Returns:
            bool: 下载是否成功
        """
        if not self.sftp:
            raise RuntimeError("SSH 未连接，请先调用 connect()")

        try:
            self.sftp.stat(remote_path)

            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            remote_size = self.sftp.stat(remote_path).st_size

            if Path(local_path).exists():
                local_size = Path(local_path).stat().st_size
                if local_size == remote_size:
                    logger.info(f"跳过 (已存在): {Path(local_path).name}")
                    return True
                logger.info(f"文件不完整，继续下载：{Path(local_path).name}")

            logger.info(f"下载中：{Path(remote_path).name}")
            self.sftp.get(remote_path, local_path)

            logger.info(f"✓ 下载完成：{Path(local_path).name}")
            return True

        except FileNotFoundError:
            logger.warning(f"视频文件不存在：{remote_path}")
            return False
        except Exception as e:
            logger.error(f"下载失败 {remote_path}: {e}")
            return False

    def fetch_for_diagnosis(self, task_id: str) -> Optional[str]:
        """
        完整的拉取流程（连接→查找→下载→断开）

        Args:
            task_id: 任务 ID

        Returns:
            本地视频文件路径，如果失败则返回 None
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.connect()

            entry = self.find_video_by_task_id(task_id)
            if not entry:
                logger.warning(f"未找到 task_id={task_id} 的视频")
                return None

            local_path = output_dir / "videos" / entry.video_filename
            success = self.download_video(entry.video_path, str(local_path))

            if not success:
                logger.error(f"下载失败：{entry.video_filename}")
                return None

            logger.info(f"✓ 拉取完成：{local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"拉取失败：{e}")
            raise
        finally:
            self.disconnect()

    @classmethod
    def from_ssh_config(cls, config_name: str, task_id: str) -> "RemoteVideoFetcher":
        """
        类方法：从 SSH 配置创建拉取器

        Args:
            config_name: SSH 配置名称
            task_id: 任务 ID

        Returns:
            RemoteVideoFetcher 实例

        Raises:
            ValueError: SSH 配置不存在时抛出
        """
        ssh_config = get_ssh_config(config_name)
        if not ssh_config:
            raise ValueError(f"SSH 配置不存在：{config_name}")

        if not ssh_config.remote_log:
            raise ValueError(f"SSH 配置缺少 remote-log: {config_name}")

        fetch_config = FetchConfig(
            host=ssh_config.host,
            port=ssh_config.port,
            username=ssh_config.user,
            key_filename=ssh_config.pem_key,
            remote_log_path=ssh_config.remote_log,
            output_dir=str(
                Path(__file__).parent.parent.parent / "output" / "remote_fetch"
            ),
            task_id=task_id,
        )

        logger.info(f"从 SSH 配置创建拉取器：{config_name}")
        return cls(fetch_config)
