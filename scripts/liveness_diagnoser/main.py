#!/usr/bin/env python3
"""
活体诊断 CLI 主入口 - 命令行界面

支持本地视频诊断和远程任务诊断两种模式。

使用示例:
    # 本地视频诊断
    python -m scripts.liveness_diagnoser --video test.webm --actions blink nod

    # 远程任务诊断
    python -m scripts.liveness_diagnoser --config face-server --task-id abc123

    # 诊断最近 N 个失败视频
    python -m scripts.liveness_diagnoser --config face-server --recent-failures 5

    # 查看帮助
    python -m scripts.liveness_diagnoser --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .analyzer import VideoAnalyzer
from .reporter import DiagnosisReporter
from .fetcher import RemoteVideoFetcher
from .models import FetchConfig


def setup_logging(verbose: bool = False) -> None:
    """
    配置日志系统

    Args:
        verbose: 是否显示详细日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="活体检测诊断工具 - 分析视频并生成诊断报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 本地视频诊断
  python -m scripts.liveness_diagnoser --video test.webm --actions blink nod

  # 远程任务诊断
  python -m scripts.liveness_diagnoser --config face-server --task-id abc123

  # 诊断最近 N 个失败视频
  python -m scripts.liveness_diagnoser --config face-server --recent-failures 5

  # 指定输出格式
  python -m scripts.liveness_diagnoser --video test.webm --formats console json html

  # 显示详细日志
  python -m scripts.liveness_diagnoser --video test.webm -v
        """,
    )

    # SSH 配置（远程诊断必需）
    parser.add_argument(
        "--config",
        type=str,
        help="SSH 配置名称（用于远程诊断）",
    )

    # 任务 ID（远程诊断）
    parser.add_argument(
        "--task-id",
        type=str,
        help="要诊断的任务 ID（用于远程诊断）",
    )

    # 最近失败诊断
    parser.add_argument(
        "--recent-failures",
        type=int,
        metavar="N",
        help="诊断最近 N 个失败视频（可选功能）",
    )

    # 本地视频路径
    parser.add_argument(
        "--video",
        type=str,
        help="本地视频文件路径（用于本地诊断）",
    )

    # 期望的动作列表
    parser.add_argument(
        "--actions",
        type=str,
        nargs="+",
        default=["blink", "nod"],
        help="期望的动作列表，默认：blink nod",
    )

    # 输出目录
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/diagnosis",
        help="报告输出目录，默认：output/diagnosis",
    )

    # 报告格式
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["console", "json", "html"],
        choices=["console", "json", "html"],
        help="报告格式，默认：console json html",
    )

    # 详细日志
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示详细日志",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    验证命令行参数

    Args:
        args: 解析后的参数

    Raises:
        SystemExit: 参数验证失败时退出
    """
    # 必须指定 --config 或 --video
    if not args.config and not args.video:
        logging.error("必须指定 --config（远程诊断）或 --video（本地诊断）")
        logging.error("使用 --help 查看帮助")
        sys.exit(1)

    # 远程诊断模式：必须同时指定 --config 和 --task-id
    if args.config and not args.task_id and not args.recent_failures:
        logging.error("远程诊断模式必须指定 --task-id 或 --recent-failures")
        sys.exit(1)

    # 本地诊断模式：必须指定 --video
    if args.video and not Path(args.video).exists():
        logging.error(f"视频文件不存在：{args.video}")
        sys.exit(1)

    # --recent-failures 必须与 --config 一起使用
    if args.recent_failures and not args.config:
        logging.error("--recent-failures 必须与 --config 一起使用")
        sys.exit(1)


def diagnose_local_video(
    video_path: str,
    actions: list[str],
    output_dir: str,
    task_id: Optional[str] = None,
    formats: Optional[list[str]] = None,
) -> int:
    """
    诊断本地视频

    Args:
        video_path: 视频文件路径
        actions: 期望的动作列表
        output_dir: 报告输出目录
        task_id: 可选的任务 ID
        formats: 报告格式列表

    Returns:
        exit code: 0（成功）或 1（失败）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始诊断本地视频：{video_path}")

    try:
        # 创建分析器并分析视频
        analyzer = VideoAnalyzer()
        result = analyzer.analyze(video_path, actions=actions, task_id=task_id)
        analyzer.close()

        # 创建报告生成器并生成报告
        reporter = DiagnosisReporter()

        # 打印控制台报告
        if "console" in (formats or ["console"]):
            console_report = reporter.generate_console_report(result)
            print(console_report)

        # 保存其他格式报告
        save_formats = [f for f in (formats or ["console"]) if f != "console"]
        if save_formats:
            saved_files = reporter.save_report(result, output_dir, formats=save_formats)
            logger.info(f"报告已保存：{saved_files}")

        logger.info("诊断完成")
        return 0

    except Exception as e:
        logger.error(f"诊断失败：{e}")
        if logging.getLogger().level == logging.DEBUG:
            import traceback

            traceback.print_exc()
        return 1


def diagnose_remote_task(
    config_name: str,
    task_id: str,
    output_dir: str,
    actions: Optional[list[str]] = None,
    formats: Optional[list[str]] = None,
) -> int:
    """
    诊断远程任务

    Args:
        config_name: SSH 配置名称
        task_id: 任务 ID
        output_dir: 报告输出目录
        actions: 期望的动作列表
        formats: 报告格式列表

    Returns:
        exit code: 0（成功）或 1（失败）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始诊断远程任务：task_id={task_id}, config={config_name}")

    try:
        # 创建拉取器并获取视频
        fetcher = RemoteVideoFetcher.from_ssh_config(config_name, task_id)
        video_path = fetcher.fetch_for_diagnosis(task_id)

        if not video_path:
            logger.error(f"未能获取视频文件：task_id={task_id}")
            return 1

        logger.info(f"视频已获取：{video_path}")

        # 调用本地诊断
        return diagnose_local_video(
            video_path=video_path,
            actions=actions or ["blink", "nod"],
            output_dir=output_dir,
            task_id=task_id,
            formats=formats,
        )

    except ValueError as e:
        logger.error(f"配置错误：{e}")
        return 1
    except Exception as e:
        logger.error(f"远程诊断失败：{e}")
        if logging.getLogger().level == logging.DEBUG:
            import traceback

            traceback.print_exc()
        return 1


def diagnose_recent_failures(
    config_name: str,
    count: int,
    output_dir: str,
    actions: Optional[list[str]] = None,
    formats: Optional[list[str]] = None,
) -> int:
    """
    诊断最近 N 个失败视频（可选功能）

    Args:
        config_name: SSH 配置名称
        count: 失败视频数量
        output_dir: 报告输出目录
        actions: 期望的动作列表
        formats: 报告格式列表

    Returns:
        exit code: 0（成功）或 1（失败）
    """
    logger = logging.getLogger(__name__)
    logger.warning(
        "--recent-failures 功能尚未完全实现，将尝试诊断最近的任务（如果日志支持）"
    )

    # 注意：此功能需要日志解析器支持按失败状态筛选
    # 这里提供一个简化实现框架
    logger.info("该功能需要日志解析器支持失败状态筛选，当前版本可能无法正确工作")
    return 1


def main() -> None:
    """主函数 - 解析参数并执行诊断"""
    args = parse_arguments()

    # 设置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 验证参数
    validate_args(args)

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录：{output_path}")

    # 执行诊断
    exit_code = 0

    if args.video:
        # 本地视频诊断
        exit_code = diagnose_local_video(
            video_path=args.video,
            actions=args.actions,
            output_dir=args.output_dir,
            task_id=args.task_id,
            formats=args.formats,
        )

    elif args.config and args.task_id:
        # 远程任务诊断
        exit_code = diagnose_remote_task(
            config_name=args.config,
            task_id=args.task_id,
            output_dir=args.output_dir,
            actions=args.actions,
            formats=args.formats,
        )

    elif args.config and args.recent_failures:
        # 最近失败诊断（可选功能）
        exit_code = diagnose_recent_failures(
            config_name=args.config,
            count=args.recent_failures,
            output_dir=args.output_dir,
            actions=args.actions,
            formats=args.formats,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
