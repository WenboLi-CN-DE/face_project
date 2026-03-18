#!/usr/bin/env python3
"""
远程日志与视频拉取工具

从远程服务器通过 SSH+.pem 认证拉取日志和视频文件，并自动匹配它们用于本地测试分析。

用法:
    uv run python scripts/remote_fetch.py \\
        --host <server_ip> \\
        --user <username> \\
        --pem-key <path_to_pem> \\
        --remote-log <path_to_log> \\
        --output-dir <local_output_dir>
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paramiko import SSHClient, AutoAddPolicy, SFTPClient

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# 支持从项目根目录运行
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from log_parser import LogParser, VideoEntry  # noqa: E402
from ssh_config import SSHConfigManager, get_ssh_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FetchConfig:
    """拉取配置"""

    host: str
    port: int
    username: str
    key_filename: str
    remote_log_path: str
    output_dir: str
    download_videos: bool = True
    download_logs: bool = True
    max_videos: Optional[int] = None


class RemoteFetcher:
    """远程文件拉取器"""

    def __init__(self, config: FetchConfig):
        self.config = config
        self.ssh: Optional[SSHClient] = None
        self.sftp: Optional[SFTPClient] = None
        self.video_entries: List[VideoEntry] = []

    def connect(self):
        """建立 SSH 连接"""
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

    def disconnect(self):
        """断开连接"""
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        logger.info("SSH 连接已断开")

    def parse_remote_log(self) -> List[VideoEntry]:
        """解析远程日志文件"""
        logger.info(f"解析远程日志：{self.config.remote_log_path}")

        temp_log = Path(self.config.output_dir) / "temp_remote.log"
        temp_log.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.sftp.get(self.config.remote_log_path, str(temp_log))
            logger.info(f"✓ 日志已下载到 {temp_log}")

            parser = LogParser()
            self.video_entries = parser.parse_file(str(temp_log))

            final_log = Path(self.config.output_dir) / "remote_server.log"
            temp_log.rename(final_log)

            return self.video_entries

        except FileNotFoundError:
            logger.error(f"远程日志文件不存在：{self.config.remote_log_path}")
            raise
        except Exception as e:
            logger.error(f"解析日志失败：{e}")
            raise

    def download_video(self, remote_path: str, local_path: str) -> bool:
        """下载单个视频文件"""
        try:
            self.sftp.stat(remote_path)

            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            remote_size = self.sftp.stat(remote_path).st_size

            if Path(local_path).exists():
                local_size = Path(local_path).stat().st_size
                if local_size == remote_size:
                    logger.info(f"跳过 (已存在): {Path(local_path).name}")
                    return True

            if tqdm:
                with tqdm(
                    total=remote_size,
                    unit="B",
                    unit_scale=True,
                    desc=Path(remote_path).name,
                ) as pbar:

                    def callback(current, total):
                        pbar.update(current - pbar.n)

                    self.sftp.get(remote_path, local_path, callback=callback)
            else:
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

    def download_videos(self, entries: List[VideoEntry]) -> Dict[str, Any]:
        """批量下载视频文件"""
        results = {
            "total": len(entries),
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "not_found": 0,
            "entries": [],
        }

        for i, entry in enumerate(entries, 1):
            if self.config.max_videos and i > self.config.max_videos:
                logger.info(f"已达到最大下载数量：{self.config.max_videos}")
                break

            local_path = Path(self.config.output_dir) / "videos" / entry.video_filename

            logger.info(f"[{i}/{len(entries)}] 下载：{entry.video_filename}")

            success = self.download_video(entry.video_path, str(local_path))

            if success:
                results["downloaded"] += 1
                entry.local_path = str(local_path)
                results["entries"].append(entry)
            else:
                try:
                    self.sftp.stat(entry.video_path)
                    results["failed"] += 1
                except FileNotFoundError:
                    results["not_found"] += 1

        return results

    def generate_report(self, download_results: Dict[str, Any]):
        """生成匹配报告"""
        report_path = Path(self.config.output_dir) / "fetch_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 远程文件拉取报告\n\n")
            f.write(f"**服务器**: {self.config.host}\n")
            f.write(f"**日志文件**: {self.config.remote_log_path}\n")
            f.write(f"**拉取时间**: {datetime.now().isoformat()}\n\n")

            f.write("## 汇总统计\n\n")
            f.write(f"- 总视频数：{download_results['total']}\n")
            f.write(f"- 成功下载：{download_results['downloaded']}\n")
            f.write(f"- 文件不存在：{download_results['not_found']}\n")
            f.write(f"- 下载失败：{download_results['failed']}\n")
            f.write(f"- 已跳过：{download_results['skipped']}\n\n")

            f.write("## 视频清单\n\n")
            f.write("| 文件名 | 任务 ID | 动作 | 状态 |\n")
            f.write("|--------|---------|------|------|\n")

            for entry in download_results["entries"]:
                actions_str = ", ".join(entry.actions)
                status = "✓" if hasattr(entry, "local_path") else "✗"
                f.write(
                    f"| {entry.video_filename} | {entry.task_id} | {actions_str} | {status} |\n"
                )

            f.write("\n## 使用说明\n\n")
            f.write("### 本地测试单个视频\n\n")
            f.write("```bash\n")
            f.write(
                "uv run python diagnose_video.py output/remote_fetch/videos/<video_name>\n"
            )
            f.write("```\n\n")

            f.write("### 批量测试所有视频\n\n")
            f.write("修改 `15/动作对应.txt` 后运行:\n\n")
            f.write("```bash\n")
            f.write("uv run python batch_test_videos.py\n")
            f.write("```\n")

        logger.info(f"✓ 报告已生成：{report_path}")

    def fetch(self):
        """执行完整拉取流程"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.connect()

            self.parse_remote_log()

            if not self.video_entries:
                logger.warning("未找到任何视频条目")
                return

            if self.config.download_videos:
                download_results = self.download_videos(self.video_entries)
                self.generate_report(download_results)

                logger.info("\n" + "=" * 60)
                logger.info("拉取完成")
                logger.info("=" * 60)
                logger.info(f"总视频数：{download_results['total']}")
                logger.info(f"成功下载：{download_results['downloaded']}")
                logger.info(f"文件不存在：{download_results['not_found']}")
                logger.info(f"下载失败：{download_results['failed']}")
                logger.info(f"\n报告：{output_dir / 'fetch_report.md'}")
                logger.info(f"视频目录：{output_dir / 'videos'}")

        finally:
            self.disconnect()


def main():
    # 先解析 --config 参数，以便加载配置
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", help="SSH 配置名称（从 ssh-config.txt 读取）")
    temp_args, remaining = temp_parser.parse_known_args()

    # 如果指定了配置，从配置中读取连接信息
    ssh_config = None
    if temp_args.config:
        ssh_config = get_ssh_config(temp_args.config)
        if not ssh_config:
            print(f"❌ 找不到 SSH 配置：{temp_args.config}")
            print(f"\n可用的配置:")
            manager = SSHConfigManager()
            for name in manager.list_configs():
                print(f"  - {name}")
            sys.exit(1)

    parser = argparse.ArgumentParser(
        description="从远程服务器拉取日志和视频文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件（推荐）
  uv run python scripts/remote_fetch.py --config face-server

  # 列出所有可用配置
  uv run python scripts/ssh_config.py

  # 基本用法（手动指定参数）
  uv run python scripts/remote_fetch.py \\
    --host 192.168.1.100 \\
    --user deploy \\
    --pem-key ~/.ssh/id_rsa.pem \\
    --remote-log /opt/face_cls/logs/app.log \\
    --output-dir output/remote_fetch

  # 仅下载前 5 个视频
  uv run python scripts/remote_fetch.py ... --max-videos 5

  # 仅解析日志，不下载视频
  uv run python scripts/remote_fetch.py ... --no-download-videos
        """,
    )

    # 当使用 --config 时，这些参数变为可选
    required = not bool(ssh_config)

    parser.add_argument("--config", help="SSH 配置名称（从 ssh-config.txt 读取）")
    parser.add_argument("--host", required=required, help="远程服务器 IP")
    parser.add_argument("--port", type=int, default=22, help="SSH 端口 (默认：22)")
    parser.add_argument("--user", required=required, help="SSH 用户名")
    parser.add_argument("--pem-key", required=required, help="SSH 私钥文件路径 (.pem)")
    parser.add_argument("--remote-log", required=required, help="远程日志文件路径")
    parser.add_argument(
        "--output-dir", default="output/remote_fetch", help="本地输出目录"
    )
    parser.add_argument(
        "--no-download-videos", action="store_true", help="仅解析日志，不下载视频"
    )
    parser.add_argument("--max-videos", type=int, help="最大下载视频数量")

    args = parser.parse_args()

    # 如果使用了配置文件，用配置中的值覆盖命令行参数
    if ssh_config:
        args.host = ssh_config.host
        args.user = ssh_config.user
        args.pem_key = ssh_config.pem_key
        args.port = ssh_config.port
        if ssh_config.remote_log and not args.remote_log:
            args.remote_log = ssh_config.remote_log

    config = FetchConfig(
        host=args.host,
        port=args.port,
        username=args.user,
        key_filename=args.pem_key,
        remote_log_path=args.remote_log,
        output_dir=args.output_dir,
        download_videos=not args.no_download_videos,
        max_videos=args.max_videos,
    )

    fetcher = RemoteFetcher(config)
    fetcher.fetch()


if __name__ == "__main__":
    main()
