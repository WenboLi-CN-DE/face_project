#!/usr/bin/env python3
"""
日志视频分析器 - 从远程日志中提取所有测试相关信息

可提取的信息：
1. 视频基本信息：task_id, request_id, video_path, actions
2. 视频属性：FPS, 总帧数，分辨率，时长
3. 基准帧配置：基准帧数，动作帧数
4. 分析配置：threshold, max_width
5. 人脸检测：总帧数，检测到人脸数，人脸检出率
6. 分数统计：最高分，平均分
7. 活体判定：is_liveness, best_score, threshold, confidence
8. 动作详情：每个动作的 frames, events, avg_score
9. 回调状态：callback_url, status, attempts
10. 错误信息：视频帧数无效等警告
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class VideoAnalysisResult:
    """单个视频的分析结果"""
    
    # 基本信息
    task_id: str = ""
    request_id: str = ""
    video_filename: str = ""
    video_path: str = ""
    expected_actions: List[str] = field(default_factory=list)
    callback_url: str = ""
    
    # 视频属性
    fps: float = 0.0
    total_frames: int = 0
    resolution: str = ""
    duration: float = 0.0
    
    # 基准帧配置
    benchmark_frames: int = 0
    action_frames: int = 0
    frames_per_action: int = 0
    
    # 分析配置
    threshold: float = 0.95
    max_width: int = 640
    
    # 人脸检测
    face_detected_frames: int = 0
    face_detection_rate: float = 0.0
    
    # 分数统计
    max_score: float = 0.0
    avg_score: float = 0.0
    
    # 活体判定
    is_liveness: int = 0
    best_score: float = 0.0
    liveness_threshold: float = 0.95
    liveness_confidence: float = 0.0
    
    # 动作详情
    action_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # 回调状态
    callback_status: int = 0
    callback_attempts: int = 0
    
    # 错误/警告
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # 时间戳
    log_timestamp: str = ""


class LogVideoAnalyzer:
    """日志视频分析器"""
    
    # 正则表达式模式
    PATTERNS = {
        'vrlMoveLiveness': re.compile(
            r"vrlMoveLiveness\s+"
            r"request_id=([a-f0-9-]+)\s+"
            r"task_id=([a-f0-9-]+)\s+"
            r"video=(\S+)\s+"
            r"actions=(\[[^\]]+\])"
        ),
        'callback_url': re.compile(r"callback_url=(\S+)"),
        'video_attr': re.compile(
            r"视频属性：FPS=([\d.]+),\s+总帧数=(-?\d+),\s+分辨率=(\d+x\d+),\s+时长=([\d.]+) 秒"
        ),
        'benchmark': re.compile(r"基准段：(\d+)\s+帧"),
        'analysis_params': re.compile(
            r"分析参数：max_frames=(\d+),\s+benchmark_frames=(\d+),\s+action_frames=(\d+),\s+frames_per_action=(\d+)"
        ),
        'config': re.compile(r"配置：threshold=([\d.]+),\s+max_width=(\d+)"),
        'face_detection': re.compile(
            r"帧处理完成：总帧数=(\d+),\s+检测到人脸=(\d+),\s+人脸检出率=([\d.]+)%"
        ),
        'score_stats': re.compile(r"分数统计：最高分=([\d.]+),\s+平均分=([\d.]+)"),
        'liveness_result': re.compile(
            r"活体判定：is_liveness=(\d+),\s+best_score=([\d.]+),\s+threshold=([\d.]+),\s+confidence=([\d.]+)"
        ),
        'action_result': re.compile(
            r"动作\s+'(\w+)':\s+frames=(\d+),\s+events=(\d+),\s+avg_score=([\d.]+)"
        ),
        'liveness_done': re.compile(
            r"活体检测完成\s+task_id=([a-f0-9-]+)\s+is_liveness=(\d+)\s+confidence=([\d.]+)"
        ),
        'callback_success': re.compile(r"回调成功.*status=(\d+)"),
        'callback_attempt': re.compile(r"发送回调\s+attempt=(\d+)/(\d+)"),
        'warning': re.compile(r"WARNING.*"),
    }
    
    def __init__(self):
        self.results: Dict[str, VideoAnalysisResult] = {}
        self.current_task_id: Optional[str] = None
    
    def parse_line(self, line: str):
        """解析单行日志"""
        line = line.strip()
        if not line:
            return
        
        # 提取时间戳
        timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d+)", line)
        timestamp = timestamp_match.group(1) if timestamp_match else ""
        
        # 1. vrlMoveLiveness 请求
        match = self.PATTERNS['vrlMoveLiveness'].search(line)
        if match:
            request_id, task_id, video_path, actions_str = match.groups()
            self.current_task_id = task_id
            
            # 解析 actions 列表
            actions = [a.strip().strip("'\"") for a in actions_str.strip('[]').split(',')]
            
            result = VideoAnalysisResult(
                task_id=task_id,
                request_id=request_id,
                video_path=video_path,
                video_filename=Path(video_path).name,
                expected_actions=actions,
                log_timestamp=timestamp,
            )
            
            # 提取 callback_url
            callback_match = self.PATTERNS['callback_url'].search(line)
            if callback_match:
                result.callback_url = callback_match.group(1)
            
            self.results[task_id] = result
            return
        
        # 如果没有当前 task_id，跳过后续解析
        if not self.current_task_id or self.current_task_id not in self.results:
            return
        
        result = self.results[self.current_task_id]
        
        # 2. 视频属性
        match = self.PATTERNS['video_attr'].search(line)
        if match:
            fps, total_frames, resolution, duration = match.groups()
            result.fps = float(fps)
            result.total_frames = int(total_frames)
            result.resolution = resolution
            result.duration = float(duration)
        
        # 3. 基准帧配置
        match = self.PATTERNS['benchmark'].search(line)
        if match:
            result.benchmark_frames = int(match.group(1))
        
        # 4. 分析参数
        match = self.PATTERNS['analysis_params'].search(line)
        if match:
            _, benchmark_frames, action_frames, frames_per_action = match.groups()
            result.benchmark_frames = int(benchmark_frames)
            result.action_frames = int(action_frames)
            result.frames_per_action = int(frames_per_action)
        
        # 5. 分析配置
        match = self.PATTERNS['config'].search(line)
        if match:
            threshold, max_width = match.groups()
            result.threshold = float(threshold)
            result.max_width = int(max_width)
            result.liveness_threshold = float(threshold)
        
        # 6. 人脸检测
        match = self.PATTERNS['face_detection'].search(line)
        if match:
            total, detected, rate = match.groups()
            result.face_detected_frames = int(detected)
            result.face_detection_rate = float(rate) / 100.0
        
        # 7. 分数统计
        match = self.PATTERNS['score_stats'].search(line)
        if match:
            max_score, avg_score = match.groups()
            result.max_score = float(max_score)
            result.avg_score = float(avg_score)
        
        # 8. 活体判定
        match = self.PATTERNS['liveness_result'].search(line)
        if match:
            is_liveness, best_score, threshold, confidence = match.groups()
            result.is_liveness = int(is_liveness)
            result.best_score = float(best_score)
            result.liveness_threshold = float(threshold)
            result.liveness_confidence = float(confidence)
        
        # 9. 动作结果
        match = self.PATTERNS['action_result'].search(line)
        if match:
            action_name, frames, events, avg_score = match.groups()
            
            # 计算动作置信度（根据代码逻辑）
            frames_in_slot = int(frames)
            events_count = int(events)
            avg_slot_score = float(avg_score)
            
            expected_events = max(1, int(frames_in_slot / 90))
            event_rate = min(events_count / max(expected_events, 1), 1.0)
            
            if action_name in ["blink", "mouth_open"]:
                confidence = round(event_rate * 0.85 + avg_slot_score * 0.15, 4)
            else:
                confidence = round(event_rate * 0.82 + avg_slot_score * 0.18, 4)
            
            passed = confidence >= 0.75  # 默认 action_threshold
            
            # 查找是否已存在该动作
            existing = next((a for a in result.action_details if a['action'] == action_name), None)
            if existing:
                # 更新已有动作
                existing.update({
                    'frames': frames_in_slot,
                    'events': events_count,
                    'avg_score': avg_slot_score,
                    'event_rate': round(event_rate, 4),
                    'confidence': confidence,
                    'passed': passed,
                })
            else:
                # 添加新动作
                result.action_details.append({
                    'action': action_name,
                    'frames': frames_in_slot,
                    'events': events_count,
                    'avg_score': avg_slot_score,
                    'event_rate': round(event_rate, 4),
                    'confidence': confidence,
                    'passed': passed,
                })
        
        # 10. 活体检测完成
        match = self.PATTERNS['liveness_done'].search(line)
        if match:
            task_id, is_liveness, confidence = match.groups()
            if task_id == self.current_task_id:
                result.is_liveness = int(is_liveness)
                result.liveness_confidence = float(confidence)
        
        # 11. 回调成功
        match = self.PATTERNS['callback_success'].search(line)
        if match:
            status = int(match.group(1))
            result.callback_status = status
        
        # 12. 回调尝试
        match = self.PATTERNS['callback_attempt'].search(line)
        if match:
            attempt, total = match.groups()
            result.callback_attempts = int(attempt)
        
        # 13. 警告信息
        if 'WARNING' in line:
            result.warnings.append(line.strip())
        
        # 14. 错误信息
        if 'ERROR' in line or '失败' in line:
            result.errors.append(line.strip())
    
    def parse_file(self, log_path: str):
        """解析日志文件"""
        log_file = Path(log_path)
        if not log_file.exists():
            raise FileNotFoundError(f"日志文件不存在：{log_path}")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.parse_line(line)
        
        return self.results
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """转换为字典列表"""
        return [asdict(r) for r in self.results.values()]
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict_list(), indent=indent, ensure_ascii=False)
    
    def generate_test_report(self) -> str:
        """生成测试报告"""
        if not self.results:
            return "没有分析结果"
        
        report = []
        report.append("=" * 80)
        report.append("日志视频分析报告")
        report.append("=" * 80)
        report.append(f"生成时间：{datetime.now().isoformat()}")
        report.append(f"视频总数：{len(self.results)}")
        report.append("")
        
        # 汇总统计
        total_videos = len(self.results)
        passed_videos = sum(1 for r in self.results.values() if r.is_liveness == 1)
        total_actions = sum(len(r.action_details) for r in self.results.values())
        passed_actions = sum(
            sum(1 for a in r.action_details if a.get('passed', False))
            for r in self.results.values()
        )
        
        report.append("汇总统计:")
        report.append(f"  通过视频数：{passed_videos}/{total_videos} ({passed_videos/total_videos*100:.1f}%)")
        report.append(f"  通过动作数：{passed_actions}/{total_actions} ({passed_actions/total_actions*100:.1f}%)")
        report.append("")
        
        # 各动作统计
        action_stats = {}
        for result in self.results.values():
            for action in result.action_details:
                name = action['action']
                if name not in action_stats:
                    action_stats[name] = {'total': 0, 'passed': 0, 'confidences': []}
                action_stats[name]['total'] += 1
                if action.get('passed', False):
                    action_stats[name]['passed'] += 1
                action_stats[name]['confidences'].append(action.get('confidence', 0))
        
        report.append("各动作统计:")
        report.append("-" * 80)
        for action, stats in sorted(action_stats.items()):
            pass_rate = stats['passed'] / stats['total'] * 100
            avg_conf = sum(stats['confidences']) / len(stats['confidences'])
            report.append(
                f"  {action:15s}  通过率：{pass_rate:5.1f}%  ({stats['passed']}/{stats['total']})  "
                f"平均置信度：{avg_conf:.2%}"
            )
        report.append("")
        
        # 详细结果
        report.append("详细结果:")
        report.append("-" * 80)
        
        for idx, result in enumerate(self.results.values(), 1):
            report.append(f"\n[{idx}/{total_videos}] {result.video_filename}")
            report.append(f"  Task ID: {result.task_id}")
            report.append(f"  期望动作：{result.expected_actions}")
            report.append(f"  视频属性：{result.fps} FPS, {result.total_frames}帧，{result.resolution}, {result.duration}秒")
            report.append(f"  人脸检出率：{result.face_detection_rate:.1%}")
            report.append(f"  分数统计：最高={result.max_score:.4f}, 平均={result.avg_score:.4f}")
            report.append(f"  活体判定：{'✅' if result.is_liveness else '❌'} (confidence={result.liveness_confidence:.2%})")
            
            if result.action_details:
                report.append(f"  动作验证:")
                for action in result.action_details:
                    status = "✅" if action.get('passed', False) else "❌"
                    report.append(
                        f"    {status} {action['action']:12s}  "
                        f"frames={action['frames']}, events={action['events']}, "
                        f"avg_score={action['avg_score']:.4f}, "
                        f"confidence={action.get('confidence', 0):.2%}"
                    )
            
            if result.warnings:
                report.append(f"  警告：{len(result.warnings)} 条")
                for warning in result.warnings[:3]:  # 只显示前 3 条
                    report.append(f"    ⚠️  {warning[:100]}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法：python scripts/log_video_analyzer.py <log_file> [output_json]")
        print("\n示例:")
        print("  python scripts/log_video_analyzer.py output/remote_fetch/remote_server.log")
        print("  python scripts/log_video_analyzer.py output/remote_fetch/remote_server.log output/analysis.json")
        sys.exit(1)
    
    log_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = LogVideoAnalyzer()
    analyzer.parse_file(log_path)
    
    # 打印测试报告
    print(analyzer.generate_test_report())
    
    # 保存 JSON
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            f.write(analyzer.to_json())
        print(f"\n✅ JSON 已保存：{output_json}")


if __name__ == "__main__":
    main()
