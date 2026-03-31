#!/usr/bin/env python3
"""
报告生成模块 - 活体诊断报告生成器

支持三种输出格式:
- 控制台文本报告
- JSON 格式报告
- HTML 格式报告 (带图表)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import DiagnosisResult, FrameAnalysis

logger = logging.getLogger(__name__)


class DiagnosisReporter:
    """诊断报告生成器。"""

    def __init__(self, template_dir: str | None = None):
        """
        初始化报告生成器

        Args:
            template_dir: HTML 模板目录，默认为 templates/
        """
        if template_dir is None:
            template_dir = str(Path(__file__).parent / "templates")
        self.template_dir = Path(template_dir)
        self._jinja2_available = self._check_jinja2()

    def _check_jinja2(self) -> bool:
        """检查 Jinja2 是否可用。"""
        try:
            import jinja2  # noqa: F401

            return True
        except ImportError:
            logger.warning("Jinja2 未安装，将使用简化 HTML 模板")
            return False

    def generate_console_report(self, result: DiagnosisResult) -> str:
        """
        生成控制台文本报告

        Args:
            result: 诊断结果

        Returns:
            格式化的文本报告
        """
        lines = []
        sep = "=" * 70

        # 标题
        lines.append(sep)
        lines.append("活体检测诊断报告")
        lines.append(sep)
        lines.append("")

        # 视频信息
        lines.append("📹 视频信息")
        lines.append("-" * 40)
        lines.append(f"  文件名：    {result.video_info.filename}")
        if result.task_id:
            lines.append(f"  任务 ID:    {result.task_id}")
        lines.append(f"  总帧数：    {result.total_frames}")
        lines.append(f"  FPS:        {result.video_info.fps:.1f}")
        lines.append(
            f"  分辨率：    {result.video_info.width} × {result.video_info.height}"
        )
        lines.append(f"  时长：      {result.video_info.duration:.2f}s")
        lines.append(f"  分析时间：  {result.analysis_time:.2f}s")
        lines.append(f"  质量评级：  {result.quality_rating.upper()}")
        lines.append("")

        # 人脸检测统计
        lines.append("👤 人脸检测统计")
        lines.append("-" * 40)
        lines.append(f"  检出帧数：  {result.frames_with_face} / {result.total_frames}")
        lines.append(f"  检出率：    {result.face_detection_rate * 100:.1f}%")
        lines.append("")

        # 活体判定
        lines.append("🔍 活体判定")
        lines.append("-" * 40)
        status = "通过" if result.overall_passed else "未通过"
        lines.append(f"  判定结果：  {status}")

        # 计算最高平滑分
        smoothed_scores = [
            f.smoothed_score
            for f in result.frame_analyses
            if f.smoothed_score is not None
        ]
        max_score = max(smoothed_scores) if smoothed_scores else 0.0
        lines.append(f"  最高平滑分：{max_score:.3f}")
        lines.append(f"  阈值：      {result.threshold:.3f}")
        lines.append(f"  判定说明：  {result.overall_message}")
        lines.append("")

        # 动作验证
        lines.append("🎯 动作验证")
        lines.append("-" * 40)
        lines.append(
            f"  {'动作':<12} {'触发帧':<8} {'事件':<6} {'平均分':<8} {'置信度':<8} {'状态':<6} 说明"
        )
        lines.append("-" * 90)

        for action in result.action_details:
            status_icon = "✅" if action.passed else "❌"
            conf_cn = {"high": "高", "medium": "中", "low": "低"}.get(
                action.confidence, "低"
            )
            lines.append(
                f"  {action.name_cn:<12} {len(action.frames):<8} {action.events:<6} "
                f"{action.avg_score:<8.2f} {conf_cn:<8} {status_icon:<6}  {action.message}"
            )
        lines.append("")

        # 诊断建议
        lines.append("💡 诊断建议")
        lines.append("-" * 40)
        for suggestion in result.suggestions:
            lines.append(f"  {suggestion}")
        lines.append("")

        lines.append(sep)
        lines.append(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(sep)

        return "\n".join(lines)

    def generate_json_report(self, result: DiagnosisResult) -> str:
        """
        生成 JSON 格式报告

        Args:
            result: 诊断结果

        Returns:
            JSON 字符串
        """
        # 计算最高平滑分
        smoothed_scores = [
            f.smoothed_score
            for f in result.frame_analyses
            if f.smoothed_score is not None
        ]
        max_score = max(smoothed_scores) if smoothed_scores else 0.0

        # 构建报告数据
        report_data = {
            "video_info": {
                "filename": result.video_info.filename,
                "path": result.video_info.path,
                "fps": result.video_info.fps,
                "width": result.video_info.width,
                "height": result.video_info.height,
                "duration": result.video_info.duration,
            },
            "task_id": result.task_id,
            "total_frames": result.total_frames,
            "frames_with_face": result.frames_with_face,
            "face_detection_rate": result.face_detection_rate,
            "blink_count": result.blink_count,
            "mouth_open_count": result.mouth_open_count,
            "nod_count": result.nod_count,
            "shake_count": result.shake_count,
            "action_details": [
                {
                    "name": action.name,
                    "name_cn": action.name_cn,
                    "frames": action.frames,
                    "events": action.events,
                    "avg_score": action.avg_score,
                    "confidence": action.confidence,
                    "passed": action.passed,
                    "message": action.message,
                }
                for action in result.action_details
            ],
            "quality": {
                "avg_quality_score": result.avg_quality_score,
                "quality_rating": result.quality_rating,
            },
            "liveness": {
                "overall_passed": result.overall_passed,
                "overall_message": result.overall_message,
                "max_smoothed_score": max_score,
                "threshold": result.threshold,
            },
            "thresholds": {
                "ear": result.ear_threshold,
                "mar": result.mar_threshold,
                "yaw": result.yaw_threshold,
                "pitch": result.pitch_threshold,
            },
            "suggestions": result.suggestions,
            "analysis_time": result.analysis_time,
            "generation_time": datetime.now().isoformat(),
        }

        return json.dumps(report_data, ensure_ascii=False, indent=2)

    def generate_html_report(self, result: DiagnosisResult) -> str:
        """
        生成 HTML 格式报告

        Args:
            result: 诊断结果

        Returns:
            HTML 字符串
        """
        # 准备图表数据
        chart_data = self._prepare_chart_data(result)

        # 尝试使用 Jinja2 模板
        if self._jinja2_available:
            try:
                from jinja2 import Environment, FileSystemLoader

                env = Environment(loader=FileSystemLoader(str(self.template_dir)))
                template = env.get_template("report.html")

                return template.render(
                    result=result,
                    chart_data=json.dumps(chart_data),
                    generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            except Exception as e:
                logger.warning(f"模板渲染失败：{e}，使用简化 HTML")

        # 使用简化 HTML
        return self._generate_simple_html(result, chart_data)

    def _generate_simple_html(
        self, result: DiagnosisResult, chart_data: dict[str, Any]
    ) -> str:
        """
        生成简化 HTML（无模板时）

        Args:
            result: 诊断结果
            chart_data: 图表数据

        Returns:
            HTML 字符串
        """
        status_class = "passed" if result.overall_passed else "failed"
        status_text = "✅ 通过" if result.overall_passed else "❌ 未通过"

        # 计算最高平滑分
        smoothed_scores = [
            f.smoothed_score
            for f in result.frame_analyses
            if f.smoothed_score is not None
        ]
        max_score = max(smoothed_scores) if smoothed_scores else 0.0

        # 动作表格行
        action_rows = ""
        for action in result.action_details:
            row_class = "passed" if action.passed else "failed"
            status_icon = "✅" if action.passed else "❌"
            conf_cn = {"high": "高", "medium": "中", "low": "低"}.get(
                action.confidence, "低"
            )
            action_rows += f"""
            <tr class="{row_class}">
                <td><strong>{action.name_cn}</strong> ({action.name})</td>
                <td>{len(action.frames)}</td>
                <td>{action.events}</td>
                <td>{action.avg_score:.2f}</td>
                <td>{conf_cn}</td>
                <td>{status_icon}</td>
                <td>{action.message}</td>
            </tr>
            """

        # 建议列表
        suggestions_html = ""
        for suggestion in result.suggestions:
            css_class = ""
            if "✅" in suggestion:
                css_class = "success"
            elif "❌" in suggestion:
                css_class = "error"
            suggestions_html += (
                f'<div class="suggestion-item {css_class}">{suggestion}</div>\n'
            )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>活体检测诊断报告 - {result.video_info.filename}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ color: #333; font-size: 24px; margin-bottom: 8px; }}
        .header .subtitle {{ color: #666; font-size: 14px; }}
        .status-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            margin-top: 12px;
        }}
        .status-badge.passed {{ background: linear-gradient(135deg, #11998e, #38ef7d); color: white; }}
        .status-badge.failed {{ background: linear-gradient(135deg, #eb3349, #f45c43); color: white; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .card h2 {{
            color: #333;
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .stat-item {{ background: #f8f9fa; padding: 16px; border-radius: 8px; }}
        .stat-item .label {{ color: #666; font-size: 12px; text-transform: uppercase; margin-bottom: 4px; }}
        .stat-item .value {{ color: #333; font-size: 20px; font-weight: 600; }}
        .stat-item .value.highlight {{ color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f8f9fa; color: #555; font-weight: 600; font-size: 13px; }}
        td {{ color: #333; font-size: 14px; }}
        tr:hover {{ background: #f8f9fa; }}
        .action-row.passed {{ border-left: 4px solid #38ef7d; }}
        .action-row.failed {{ border-left: 4px solid #f45c43; }}
        .suggestion-item {{
            padding: 12px 16px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            margin-bottom: 8px;
            border-radius: 4px;
            color: #856404;
        }}
        .suggestion-item.success {{ background: #d4edda; border-left-color: #28a745; color: #155724; }}
        .suggestion-item.error {{ background: #f8d7da; border-left-color: #dc3545; color: #721c24; }}
        .chart-container {{ position: relative; height: 300px; margin-top: 16px; }}
        .footer {{ text-align: center; color: rgba(255, 255, 255, 0.8); font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 活体检测诊断报告</h1>
            <p class="subtitle">视频文件：{result.video_info.filename}</p>
            {f'<p class="subtitle">任务 ID: {result.task_id}</p>' if result.task_id else ""}
            <span class="status-badge {status_class}">{status_text}</span>
        </div>

        <div class="card">
            <h2>📹 视频信息</h2>
            <div class="grid-3">
                <div class="stat-item">
                    <div class="label">总帧数</div>
                    <div class="value">{result.total_frames}</div>
                </div>
                <div class="stat-item">
                    <div class="label">FPS</div>
                    <div class="value">{result.video_info.fps:.1f}</div>
                </div>
                <div class="stat-item">
                    <div class="label">分辨率</div>
                    <div class="value">{result.video_info.width} × {result.video_info.height}</div>
                </div>
                <div class="stat-item">
                    <div class="label">时长</div>
                    <div class="value">{result.video_info.duration:.2f}s</div>
                </div>
                <div class="stat-item">
                    <div class="label">分析时间</div>
                    <div class="value">{result.analysis_time:.2f}s</div>
                </div>
                <div class="stat-item">
                    <div class="label">质量评级</div>
                    <div class="value highlight">{result.quality_rating.upper()}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>👤 人脸检测统计</h2>
            <div class="grid-2">
                <div class="stat-item">
                    <div class="label">检出帧数</div>
                    <div class="value">{result.frames_with_face} / {result.total_frames}</div>
                </div>
                <div class="stat-item">
                    <div class="label">检出率</div>
                    <div class="value highlight">{result.face_detection_rate * 100:.1f}%</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔍 活体判定</h2>
            <div class="grid-3">
                <div class="stat-item">
                    <div class="label">判定结果</div>
                    <div class="value {"highlight" if result.overall_passed else ""}">{"通过" if result.overall_passed else "未通过"}</div>
                </div>
                <div class="stat-item">
                    <div class="label">最高平滑分</div>
                    <div class="value">{max_score:.3f}</div>
                </div>
                <div class="stat-item">
                    <div class="label">阈值</div>
                    <div class="value">{result.threshold:.3f}</div>
                </div>
            </div>
            <p style="margin-top: 12px; color: #666;">{result.overall_message}</p>
        </div>

        <div class="card">
            <h2>🎯 动作验证</h2>
            <table>
                <thead>
                    <tr>
                        <th>动作</th>
                        <th>触发帧数</th>
                        <th>事件次数</th>
                        <th>平均分数</th>
                        <th>置信度</th>
                        <th>状态</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    {action_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>📈 分数趋势</h2>
            <div class="chart-container">
                <canvas id="scoreChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>💡 诊断建议</h2>
            {suggestions_html}
        </div>

        <div class="footer">
            <p>报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>vrlFace 活体诊断工具 v1.0</p>
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data)};
        const ctx = document.getElementById('scoreChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: chartData.labels,
                datasets: [
                    {{
                        label: '平滑分数',
                        data: chartData.smoothed_scores,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }},
                    {{
                        label: '原始分数',
                        data: chartData.raw_scores,
                        borderColor: '#764ba2',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4
                    }},
                    {{
                        label: '阈值线',
                        data: chartData.thresholds,
                        borderColor: '#dc3545',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        fill: false,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    x: {{ title: {{ display: true, text: '帧索引' }}, ticks: {{ maxTicksLimit: 20 }} }},
                    y: {{ title: {{ display: true, text: '分数' }}, min: 0, max: 1 }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

        return html

    def _prepare_chart_data(self, result: DiagnosisResult) -> dict[str, Any]:
        """
        准备图表数据

        Args:
            result: 诊断结果

        Returns:
            图表数据字典
        """
        labels = []
        smoothed_scores = []
        raw_scores = []
        thresholds = []

        for frame in result.frame_analyses:
            labels.append(str(frame.frame_idx))
            smoothed_scores.append(
                frame.smoothed_score if frame.smoothed_score is not None else 0.0
            )
            raw_scores.append(
                frame.motion_score if frame.motion_score is not None else 0.0
            )
            thresholds.append(result.threshold)

        return {
            "labels": labels,
            "smoothed_scores": smoothed_scores,
            "raw_scores": raw_scores,
            "thresholds": thresholds,
        }

    def save_report(
        self,
        result: DiagnosisResult,
        output_dir: str,
        formats: list[str] | None = None,
    ) -> list[str]:
        """
        保存报告到文件

        Args:
            result: 诊断结果
            output_dir: 输出目录
            formats: 报告格式列表，默认 ["console", "json", "html"]

        Returns:
            保存的文件路径列表
        """
        if formats is None:
            formats = ["console", "json", "html"]

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建输出目录：{output_path}")

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"diagnosis_{result.video_info.filename}_{timestamp}"

        saved_files = []

        for fmt in formats:
            try:
                if fmt == "console":
                    content = self.generate_console_report(result)
                    file_path = output_path / f"{base_filename}.txt"
                elif fmt == "json":
                    content = self.generate_json_report(result)
                    file_path = output_path / f"{base_filename}.json"
                elif fmt == "html":
                    content = self.generate_html_report(result)
                    file_path = output_path / f"{base_filename}.html"
                else:
                    logger.warning(f"未知格式：{fmt}")
                    continue

                # 保存文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                saved_files.append(str(file_path))
                logger.info(f"保存报告：{file_path}")

            except Exception as e:
                logger.error(f"保存 {fmt} 报告失败：{e}")
                raise

        return saved_files
