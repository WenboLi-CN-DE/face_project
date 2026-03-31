#!/usr/bin/env python3
"""测试报告生成模块 - DiagnosisReporter."""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.liveness_diagnoser.models import (
    VideoInfo,
    FrameAnalysis,
    ActionDetail,
    DiagnosisResult,
)


def create_sample_result() -> DiagnosisResult:
    """创建示例诊断结果用于测试。"""
    video_info = VideoInfo(
        path="/tmp/test_video.mp4",
        filename="test_video.mp4",
        total_frames=300,
        fps=30.0,
        width=1920,
        height=1080,
        duration=10.0,
    )

    # 创建一些示例帧分析数据
    frame_analyses = []
    for i in range(50):  # 只创建 50 帧用于测试
        frame_analyses.append(
            FrameAnalysis(
                frame_idx=i,
                has_face=True,
                ear=0.18 if i % 10 == 0 else 0.25,
                mar=0.50 if i % 15 == 0 else 0.35,
                pitch=10.0 + (i % 20),
                yaw=5.0 + (i % 15),
                motion_score=0.3 + (i % 30) * 0.02,
                smoothed_score=0.35 + (i % 30) * 0.015,
                quality_score=0.7 + (i % 10) * 0.02,
                is_blink=i % 10 == 0,
                is_mouth_open=i % 15 == 0,
                head_action="nod" if i % 20 == 0 else None,
            )
        )

    # 创建动作详情
    action_details = [
        ActionDetail(
            name="blink",
            name_cn="眨眼",
            frames=[0, 10, 20, 30, 40],
            events=5,
            avg_score=1.2,
            confidence="high",
            passed=True,
            message="眨眼帧数：5/50 (10.0%)",
        ),
        ActionDetail(
            name="mouth_open",
            name_cn="张嘴",
            frames=[0, 15, 30, 45],
            events=4,
            avg_score=0.9,
            confidence="medium",
            passed=False,
            message="张嘴帧数：4/50 (8.0%)",
        ),
        ActionDetail(
            name="nod",
            name_cn="点头",
            frames=[0, 20, 40],
            events=3,
            avg_score=1.5,
            confidence="high",
            passed=True,
            message="Pitch 峰峰值：25.0° (阈值：15.0°)",
        ),
        ActionDetail(
            name="shake_head",
            name_cn="摇头",
            frames=[5, 25, 45],
            events=3,
            avg_score=1.3,
            confidence="medium",
            passed=True,
            message="Yaw 峰峰值：20.0° (阈值：15.0°)",
        ),
    ]

    return DiagnosisResult(
        video_info=video_info,
        task_id="test_task_001",
        total_frames=50,
        frames_with_face=45,
        face_detection_rate=0.9,
        blink_count=5,
        mouth_open_count=4,
        nod_count=3,
        shake_count=3,
        action_details=action_details,
        frame_analyses=frame_analyses,
        avg_quality_score=0.75,
        quality_rating="good",
        suggestions=[
            "⚠️  张嘴检测未通过：张嘴帧数：4/50 (8.0%)，建议降低 mar_threshold 到 0.450",
            "✅  总体判定：通过。最佳运动分数：0.890 (阈值：0.500)",
        ],
        overall_passed=True,
        overall_message="最佳运动分数：0.890 (阈值：0.500)",
        analysis_time=2.5,
        threshold=0.5,
        ear_threshold=0.20,
        mar_threshold=0.55,
        yaw_threshold=15.0,
        pitch_threshold=15.0,
    )


class TestDiagnosisReporter:
    """DiagnosisReporter 测试类。"""

    @pytest.fixture
    def reporter(self):
        """创建 Reporter 实例。"""
        from scripts.liveness_diagnoser.reporter import DiagnosisReporter

        template_dir = (
            Path(__file__).parent.parent.parent.parent
            / "scripts"
            / "liveness_diagnoser"
            / "templates"
        )
        return DiagnosisReporter(template_dir=str(template_dir))

    @pytest.fixture
    def sample_result(self):
        """创建示例诊断结果。"""
        return create_sample_result()

    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_generate_console_report(self, reporter, sample_result):
        """测试生成控制台文本报告。"""
        report = reporter.generate_console_report(sample_result)

        # 验证报告包含关键信息
        assert isinstance(report, str)
        assert len(report) > 0
        assert "test_video.mp4" in report
        assert "视频信息" in report or "Video" in report
        assert "人脸检测" in report or "Face" in report
        assert "眨眼" in report or "blink" in report
        assert "张嘴" in report or "mouth" in report

    def test_generate_json_report(self, reporter, sample_result):
        """测试生成 JSON 格式报告。"""
        json_str = reporter.generate_json_report(sample_result)

        # 验证 JSON 格式
        assert isinstance(json_str, str)
        data = json.loads(json_str)

        # 验证包含关键字段
        assert "video_info" in data
        assert "filename" in data["video_info"]
        assert data["video_info"]["filename"] == "test_video.mp4"
        assert "total_frames" in data
        assert data["total_frames"] == 50
        assert "action_details" in data
        assert len(data["action_details"]) == 4

    def test_generate_html_report(self, reporter, sample_result):
        """测试生成 HTML 格式报告。"""
        html = reporter.generate_html_report(sample_result)

        # 验证 HTML 结构
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "test_video.mp4" in html
        assert "Chart" in html or "chart" in html  # 包含图表

    def test_generate_html_report_no_template(self, sample_result, temp_output_dir):
        """测试无模板时生成简化 HTML。"""
        from scripts.liveness_diagnoser.reporter import DiagnosisReporter

        # 使用不存在的模板目录
        reporter = DiagnosisReporter(template_dir="/nonexistent/templates")
        html = reporter.generate_html_report(sample_result)

        # 验证生成了简化 HTML
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<!DOCTYPE html>" in html
        assert "test_video.mp4" in html

    def test_prepare_chart_data(self, reporter, sample_result):
        """测试准备图表数据。"""
        chart_data = reporter._prepare_chart_data(sample_result)

        # 验证图表数据结构
        assert isinstance(chart_data, dict)
        assert "labels" in chart_data
        assert "smoothed_scores" in chart_data
        assert "raw_scores" in chart_data
        assert "thresholds" in chart_data

        # 验证数据长度
        assert len(chart_data["labels"]) > 0
        assert len(chart_data["smoothed_scores"]) > 0
        assert len(chart_data["raw_scores"]) > 0
        assert len(chart_data["thresholds"]) > 0

    def test_save_report_console(self, reporter, sample_result, temp_output_dir):
        """测试保存控制台报告。"""
        saved_files = reporter.save_report(
            sample_result, temp_output_dir, formats=["console"]
        )

        # 验证文件已保存
        assert len(saved_files) == 1
        assert Path(saved_files[0]).exists()
        assert saved_files[0].endswith(".txt")

        # 验证文件内容
        with open(saved_files[0], "r", encoding="utf-8") as f:
            content = f.read()
            assert "test_video.mp4" in content

    def test_save_report_json(self, reporter, sample_result, temp_output_dir):
        """测试保存 JSON 报告。"""
        saved_files = reporter.save_report(
            sample_result, temp_output_dir, formats=["json"]
        )

        # 验证文件已保存
        assert len(saved_files) == 1
        assert Path(saved_files[0]).exists()
        assert saved_files[0].endswith(".json")

        # 验证 JSON 格式
        with open(saved_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["video_info"]["filename"] == "test_video.mp4"

    def test_save_report_html(self, reporter, sample_result, temp_output_dir):
        """测试保存 HTML 报告。"""
        saved_files = reporter.save_report(
            sample_result, temp_output_dir, formats=["html"]
        )

        # 验证文件已保存
        assert len(saved_files) == 1
        assert Path(saved_files[0]).exists()
        assert saved_files[0].endswith(".html")

        # 验证 HTML 内容
        with open(saved_files[0], "r", encoding="utf-8") as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "test_video.mp4" in content

    def test_save_report_multiple_formats(
        self, reporter, sample_result, temp_output_dir
    ):
        """测试保存多种格式报告。"""
        saved_files = reporter.save_report(
            sample_result, temp_output_dir, formats=["console", "json", "html"]
        )

        # 验证所有文件已保存
        assert len(saved_files) == 3

        extensions = [".txt", ".json", ".html"]
        for file_path in saved_files:
            assert Path(file_path).exists()
            assert any(file_path.endswith(ext) for ext in extensions)

    def test_save_report_creates_directory(
        self, reporter, sample_result, temp_output_dir
    ):
        """测试保存报告时自动创建目录。"""
        nested_dir = os.path.join(temp_output_dir, "subdir", "nested")
        saved_files = reporter.save_report(sample_result, nested_dir, formats=["json"])

        # 验证目录已创建
        assert Path(nested_dir).exists()
        assert len(saved_files) == 1
        assert Path(saved_files[0]).exists()

    def test_filename_format(self, reporter, sample_result, temp_output_dir):
        """测试文件名格式。"""
        saved_files = reporter.save_report(
            sample_result, temp_output_dir, formats=["json"]
        )

        # 验证文件名格式：diagnosis_<filename>_<timestamp>.<ext>
        filename = Path(saved_files[0]).name
        assert filename.startswith("diagnosis_")
        assert "test_video" in filename
        assert filename.endswith(".json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
