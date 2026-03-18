#!/usr/bin/env python3
"""
日志解析工具 - 从 vrlFace 日志中提取视频和动作信息

日志格式示例:
INFO:vrlFace.liveness.api:vrlMoveLiveness request_id=xxx task_id=xxx video=/data/videos/2026/03/15/xxx.webm actions=['nod', 'blink']
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VideoEntry:
    """视频条目"""

    video_path: str
    video_filename: str
    task_id: str
    actions: List[str]
    request_id: Optional[str] = None
    line_number: Optional[int] = None
    local_path: Optional[str] = field(default=None, init=False)


class LogParser:
    """日志解析器"""

    PATTERN = re.compile(
        r"vrlMoveLiveness\s+"
        r"request_id=(\S+)\s+"
        r"task_id=(\S+)\s+"
        r"video=(\S+)\s+"
        r"actions=(\[[^\]]+\])"
    )

    def __init__(self):
        self.entries: List[VideoEntry] = []

    def parse_line(self, line: str, line_number: Optional[int] = None) -> Optional[VideoEntry]:
        """解析单行日志"""
        match = self.PATTERN.search(line)
        if not match:
            return None

        request_id, task_id, video_path, actions_str = match.groups()

        actions = []
        for item in actions_str.strip("[]").split(","):
            action = item.strip().strip("'\"")
            if action:
                actions.append(action)

        return VideoEntry(
            video_path=video_path,
            video_filename=Path(video_path).name,
            task_id=task_id,
            actions=actions,
            request_id=request_id,
            line_number=line_number,
        )

    def parse_file(self, log_path: str) -> List[VideoEntry]:
        """解析日志文件"""
        self.entries = []
        log_file = Path(log_path)

        if not log_file.exists():
            raise FileNotFoundError(f"日志文件不存在：{log_path}")

        with open(log_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                entry = self.parse_line(line, line_number)
                if entry:
                    self.entries.append(entry)

        logger.info(f"解析完成：共 {len(self.entries)} 个视频条目")
        return self.entries

    def parse_string(self, log_content: str) -> List[VideoEntry]:
        """解析日志字符串"""
        self.entries = []

        for line_number, line in enumerate(log_content.splitlines(), 1):
            entry = self.parse_line(line, line_number)
            if entry:
                self.entries.append(entry)

        return self.entries

    def to_dict_list(self) -> List[dict]:
        """转换为字典列表"""
        return [
            {
                "video_path": e.video_path,
                "video_filename": e.video_filename,
                "task_id": e.task_id,
                "actions": e.actions,
                "request_id": e.request_id,
                "line_number": e.line_number,
            }
            for e in self.entries
        ]


def parse_log_file(log_path: str) -> List[dict]:
    """便捷函数：解析日志文件并返回字典列表"""
    parser = LogParser()
    parser.parse_file(log_path)
    return parser.to_dict_list()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python scripts/log_parser.py <log_file>")
        print("\n示例:")
        print("  python scripts/log_parser.py output/vrlface.log")
        sys.exit(1)

    log_path = sys.argv[1]
    entries = parse_log_file(log_path)

    print(f"找到 {len(entries)} 个视频条目:\n")
    for entry in entries:
        print(f"  {entry['video_filename']}")
        print(f"    任务 ID: {entry['task_id']}")
        print(f"    动作：{entry['actions']}")
        print()
