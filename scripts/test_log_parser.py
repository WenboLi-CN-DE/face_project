"""
测试日志解析工具
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from log_parser import LogParser


class TestLogParser:
    """日志解析器测试"""

    def test_parse_single_line(self):
        """测试单行日志解析"""
        parser = LogParser()
        line = "INFO:vrlFace.liveness.api:vrlMoveLiveness request_id=d8c876bd-f99b-4371-bde2-65d053fddef9 task_id=39365b73-cc50-49a0-8f03-17010f0872fa video=/data/videos/2026/03/15/39365b73-cc50-49a0-8f03-17010f0872fa_1773570711.webm actions=['nod_up', 'mouth_open']"

        entry = parser.parse_line(line, 1)

        assert entry is not None
        assert entry.task_id == "39365b73-cc50-49a0-8f03-17010f0872fa"
        assert (
            entry.video_path
            == "/data/videos/2026/03/15/39365b73-cc50-49a0-8f03-17010f0872fa_1773570711.webm"
        )
        assert (
            entry.video_filename
            == "39365b73-cc50-49a0-8f03-17010f0872fa_1773570711.webm"
        )
        assert "nod_up" in entry.actions
        assert "mouth_open" in entry.actions
        assert entry.line_number == 1

    def test_parse_no_match(self):
        """测试不匹配的日志行"""
        parser = LogParser()
        line = "INFO: Application startup complete."

        entry = parser.parse_line(line)

        assert entry is None

    def test_parse_file(self, tmp_path):
        """测试解析日志文件"""
        log_content = """
INFO: Application startup complete.
INFO:vrlFace.liveness.api:vrlMoveLiveness request_id=abc task_id=task1 video=/data/videos/test1.webm actions=['blink']
INFO: Some other log
INFO:vrlFace.liveness.api:vrlMoveLiveness request_id=def task_id=task2 video=/data/videos/test2.webm actions=['nod', 'shake_head']
"""
        log_file = tmp_path / "test.log"
        log_file.write_text(log_content)

        parser = LogParser()
        entries = parser.parse_file(str(log_file))

        assert len(entries) == 2
        assert entries[0].task_id == "task1"
        assert entries[1].actions == ["nod", "shake_head"]

    def test_parse_actions_variations(self):
        """测试不同动作格式解析"""
        parser = LogParser()

        line1 = "vrlMoveLiveness request_id=a task_id=t1 video=/data/v1.webm actions=['blink']"
        entry1 = parser.parse_line(line1)
        assert entry1.actions == ["blink"]

        line2 = "vrlMoveLiveness request_id=b task_id=t2 video=/data/v2.webm actions=['nod', 'blink', 'shake_head']"
        entry2 = parser.parse_line(line2)
        assert len(entry2.actions) == 3
        assert "nod" in entry2.actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
