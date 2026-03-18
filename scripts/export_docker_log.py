#!/usr/bin/env python3
"""
Docker 日志导出工具 - 通过 SSH 在远程服务器上导出 Docker 容器日志

用法:
    uv run python scripts/export_docker_log.py \
        --config face-server \
        --container-name liveness \
        --output-path /opt/test2026/face_cls/face_project/data/vrlface.log
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ssh_config import get_ssh_config, SSHConfigManager


def build_ssh_command(config, command: str) -> list:
    """构建 SSH 命令"""
    pem_key = Path(config.pem_key).expanduser()

    return [
        "ssh",
        "-i",
        str(pem_key),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "PreferredAuthentications=publickey",
        "-p",
        str(config.port),
        f"{config.user}@{config.host}",
        command,
    ]


def export_docker_log(config, container_name: str, output_path: str) -> bool:
    """
    在远程服务器上导出 Docker 容器日志

    Args:
        config: SSH 配置
        container_name: 容器名称或过滤器
        output_path: 日志输出路径

    Returns:
        成功返回 True，失败返回 False
    """
    # 构建 docker logs 命令
    # 支持两种模式:
    # 1. 直接指定容器名称：docker logs <name>
    # 2. 使用过滤器：docker logs $(docker ps -a --filter name=<name> --format "{{.Names}}")

    # 检查是否是过滤器模式（包含特殊字符）
    if "{" in container_name or "$" in container_name:
        # 用户提供了完整的命令片段
        docker_cmd = f"docker logs {container_name} > {output_path} 2>&1"
    else:
        # 使用过滤器模式（推荐）
        docker_cmd = (
            f"docker logs $(docker ps -a --filter name={container_name} --format '{{{{.Names}}}}') "
            f"> {output_path} 2>&1"
        )

    print(f"在远程服务器执行：{docker_cmd}")

    # 构建 SSH 命令
    ssh_cmd = build_ssh_command(config, docker_cmd)

    print(f"SSH 命令：{' '.join(ssh_cmd)}")

    try:
        # 执行 SSH 命令
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(f"✅ 日志导出成功：{output_path}")
            return True
        else:
            print(f"❌ 日志导出失败:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ SSH 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行出错：{e}")
        return False


def verify_log_exists(config, log_path: str) -> bool:
    """验证日志文件是否在远程服务器上存在"""
    ssh_cmd = build_ssh_command(config, f"test -f {log_path} && echo 'exists'")

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

        if "exists" in result.stdout:
            return True
        return False

    except Exception:
        return False


def get_container_info(config, container_name: str) -> str:
    """获取容器信息"""
    ssh_cmd = build_ssh_command(
        config,
        f"docker ps -a --filter name={container_name} --format 'table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.State}}}}'",
    )

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

        return result.stdout

    except Exception as e:
        return f"获取容器信息失败：{e}"


def main():
    parser = argparse.ArgumentParser(
        description="通过 SSH 在远程服务器上导出 Docker 容器日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件
  uv run python scripts/export_docker_log.py \\
    --config face-server \\
    --container-name liveness \\
    --output-path /opt/test2026/face_cls/face_project/data/vrlface.log

  # 指定特定容器
  uv run python scripts/export_docker_log.py \\
    --config face-server \\
    --container-name face_cls-liveness-1 \\
    --output-path /tmp/liveness.log

  # 使用过滤器（高级用法）
  uv run python scripts/export_docker_log.py \\
    --config face-server \\
    --container-name 'name=liveness' \\
    --output-path /tmp/liveness.log
        """,
    )

    parser.add_argument(
        "--config", required=True, help="SSH 配置名称（从 ssh-config.txt 读取）"
    )
    parser.add_argument(
        "--container-name",
        required=True,
        help="Docker 容器名称或过滤器（支持 $(docker ps -a --filter name=xxx) 模式）",
    )
    parser.add_argument(
        "--output-path", required=True, help="远程服务器上的日志输出路径"
    )
    parser.add_argument(
        "--verify", action="store_true", help="导出后验证日志文件是否存在"
    )
    parser.add_argument(
        "--show-containers", action="store_true", help="显示匹配的容器列表并退出"
    )

    args = parser.parse_args()

    # 加载配置
    config = get_ssh_config(args.config)
    if not config:
        print(f"❌ 找不到 SSH 配置：{args.config}")
        print(f"\n可用的配置:")
        manager = SSHConfigManager()
        for name in manager.list_configs():
            print(f"  - {name}")
        sys.exit(1)

    print(f"使用配置：[{args.config}]")
    print(f"  主机：{config.host}:{config.port}")
    print(f"  用户：{config.user}")
    print()

    # 如果只需要显示容器列表
    if args.show_containers:
        print("匹配的容器:")
        print(get_container_info(config, args.container_name))
        return 0

    # 导出日志
    print(f"导出 Docker 容器 '{args.container_name}' 的日志...")
    print(f"输出路径：{args.output_path}")
    print()

    success = export_docker_log(config, args.container_name, args.output_path)

    if not success:
        print("\n❌ 日志导出失败")
        return 1

    # 验证日志文件
    if args.verify:
        print("\n验证日志文件...")
        if verify_log_exists(config, args.output_path):
            print(f"✅ 日志文件已创建：{args.output_path}")
        else:
            print(f"⚠️  无法验证日志文件是否存在：{args.output_path}")

    print("\n✅ 日志导出完成!")
    print(f"\n下一步：使用 remote_fetch.py 拉取日志和视频")
    print(
        f"  uv run python scripts/remote_fetch.py --config {args.config} --remote-log {args.output_path}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
