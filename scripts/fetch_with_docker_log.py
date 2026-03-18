#!/usr/bin/env python3
"""
一键导出 Docker 日志并拉取到本地

整合了 export_docker_log.py 和 remote_fetch.py 的功能
只需一条命令完成所有操作

用法:
    uv run python scripts/fetch_with_docker_log.py --config face-server
"""

import argparse
import sys
from pathlib import Path

# 添加 scripts 目录到路径
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from ssh_config import get_ssh_config, SSHConfigManager
from export_docker_log import export_docker_log, verify_log_exists, build_ssh_command


def check_docker_container(config, container_name: str) -> bool:
    """检查 Docker 容器是否存在"""
    ssh_cmd = build_ssh_command(
        config, f"docker ps -a --filter name={container_name} --format '{{{{.Names}}}}'"
    )

    import subprocess

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

        return container_name in result.stdout or len(result.stdout.strip()) > 0

    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="一键导出 Docker 日志并拉取到本地",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（导出 Docker 日志 + 拉取日志和视频）
  uv run python scripts/fetch_with_docker_log.py --config face-server

  # 仅导出 Docker 日志
  uv run python scripts/fetch_with_docker_log.py --config face-server --export-only

  # 仅拉取（日志已存在）
  uv run python scripts/fetch_with_docker_log.py --config face-server --fetch-only

  # 查看容器信息
  uv run python scripts/fetch_with_docker_log.py --config face-server --check-containers
        """,
    )

    parser.add_argument("--config", required=True, help="SSH 配置名称")
    parser.add_argument(
        "--export-only", action="store_true", help="仅导出 Docker 日志，不拉取文件"
    )
    parser.add_argument(
        "--fetch-only", action="store_true", help="仅拉取文件，不导出 Docker 日志"
    )
    parser.add_argument(
        "--check-containers", action="store_true", help="查看匹配的容器列表"
    )
    parser.add_argument("--max-videos", type=int, help="最大下载视频数量")
    parser.add_argument(
        "--output-dir", default="output/remote_fetch", help="本地输出目录"
    )

    args = parser.parse_args()

    # 加载配置
    config = get_ssh_config(args.config)
    if not config:
        print(f"❌ 找不到 SSH 配置：{args.config}")
        manager = SSHConfigManager()
        for name in manager.list_configs():
            print(f"  - {name}")
        sys.exit(1)

    print("=" * 60)
    print("一键导出 Docker 日志并拉取")
    print("=" * 60)
    print(f"\n使用配置：[{args.config}]")
    print(f"  主机：{config.host}:{config.port}")
    print(f"  用户：{config.user}")

    # 检查 Docker 配置
    if not config.docker_container:
        print("\n⚠️  配置中没有 Docker 容器信息")
        print("   请在 ssh-config.txt 中添加:")
        print("   docker-container = <容器名称>")
        print("   docker-log-output = <日志输出路径>")
        print("\n   或者使用 --fetch-only 直接拉取文件")
        sys.exit(1)

    print(f"  Docker 容器：{config.docker_container}")
    print(f"  日志输出：{config.docker_log_output}")

    # 查看容器
    if args.check_containers:
        print("\n检查 Docker 容器...")
        import subprocess

        ssh_cmd = build_ssh_command(
            config,
            f"docker ps -a --filter name={config.docker_container} --format 'table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.State}}}}'",
        )
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        print("\n匹配的容器:")
        print(result.stdout)
        return 0

    # 导出 Docker 日志
    if not args.fetch_only:
        print("\n" + "=" * 60)
        print("步骤 1: 导出 Docker 日志")
        print("=" * 60)

        success = export_docker_log(
            config, config.docker_container, config.docker_log_output
        )

        if not success:
            print("\n❌ Docker 日志导出失败")
            return 1

        # 验证日志文件
        print("\n验证日志文件...")
        if verify_log_exists(config, config.docker_log_output):
            print(f"✅ 日志文件已创建：{config.docker_log_output}")
        else:
            print(f"⚠️  无法验证日志文件：{config.docker_log_output}")

    # 拉取文件
    if not args.export_only:
        print("\n" + "=" * 60)
        print("步骤 2: 拉取日志和视频")
        print("=" * 60)

        # 构建 remote_fetch 命令
        fetch_cmd = [
            sys.executable,
            str(script_dir / "remote_fetch.py"),
            "--config",
            args.config,
            "--remote-log",
            config.docker_log_output,
            "--output-dir",
            args.output_dir,
        ]

        if args.max_videos:
            fetch_cmd.extend(["--max-videos", str(args.max_videos)])

        import subprocess

        result = subprocess.run(fetch_cmd)

        if result.returncode != 0:
            print("\n❌ 文件拉取失败")
            return 1

    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
