# Liveness Diagnoser 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 开发一个远程活体检测失效诊断工具，支持 SSH 拉取失败视频、本地深度分析、生成诊断报告

**Architecture:** 模块化设计，包含 fetcher（远程拉取）、analyzer（视频分析）、reporter（报告生成）三个核心模块，通过 main.py 提供统一 CLI 接口

**Tech Stack:** Python, paramiko (SSH), OpenCV, MediaPipe, Jinja2 (HTML 模板)

---

## 文件结构

```
scripts/liveness_diagnoser/
├── __init__.py              # 包初始化，导出主要类
├── main.py                  # CLI 主入口
├── fetcher.py               # 远程拉取模块（基于 RemoteFetcher）
├── analyzer.py              # 视频分析模块（整合诊断逻辑）
├── reporter.py              # 报告生成模块
├── models.py                # 数据模型（dataclasses）
└── templates/
    └── report.html          # HTML 报告模板

tests/scripts/liveness_diagnoser/
├── __init__.py
├── test_fetcher.py          # fetcher 测试
├── test_analyzer.py         # analyzer 测试
└── test_reporter.py         # reporter 测试
```

---

## Task 1: 创建目录结构和基础文件

**Files:**
- Create: `scripts/liveness_diagnoser/__init__.py`
- Create: `scripts/liveness_diagnoser/models.py`
- Create: `tests/scripts/liveness_diagnoser/__init__.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p scripts/liveness_diagnoser/templates
mkdir -p tests/scripts/liveness_diagnoser
```

- [ ] **Step 2: 创建 models.py 定义数据模型**

```python
"""数据模型定义"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class VideoInfo:
    """视频基本信息"""
    path: str
    filename: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration: float = 0.0


@dataclass
class FrameAnalysis:
    """单帧分析结果"""
    frame_idx: int
    has_face: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    motion_score: float = 0.0
    smoothed_score: float = 0.0
    quality_score: float = 0.0
    is_blink: bool = False
    is_mouth_open: bool = False
    head_action: str = "none"


@dataclass
class ActionDetail:
    """动作检测详情"""
    name: str
    name_cn: str
    frames: int
    events: int
    avg_score: float
    confidence: float
    passed: bool
    message: str


@dataclass
class DiagnosisResult:
    """诊断结果"""
    video_info: VideoInfo
    task_id: Optional[str]
    expected_actions: List[str]
    
    # 人脸检测统计
    face_detected_frames: int
    face_detection_rate: float
    
    # 活体判定
    is_liveness: int
    best_score: float
    threshold: float
    
    # 动作验证
    action_passed: bool
    action_details: List[ActionDetail]
    
    # 逐帧数据
    frame_data: List[FrameAnalysis] = field(default_factory=list)
    
    # 诊断建议
    suggestions: List[str] = field(default_factory=list)
    
    # 时间戳
    analyzed_at: datetime = field(default_factory=datetime.now)


@dataclass
class FetchConfig:
    """远程拉取配置"""
    host: str
    port: int
    username: str
    key_filename: str
    remote_log_path: str
    output_dir: str
    task_id: Optional[str] = None
```

- [ ] **Step 3: 创建包初始化文件**

```python
# scripts/liveness_diagnoser/__init__.py
"""Liveness Diagnoser - 活体检测失效诊断工具"""

from .models import (
    VideoInfo,
    FrameAnalysis,
    ActionDetail,
    DiagnosisResult,
    FetchConfig,
)
from .fetcher import RemoteVideoFetcher
from .analyzer import VideoAnalyzer
from .reporter import DiagnosisReporter

__version__ = "1.0.0"
__all__ = [
    "VideoInfo",
    "FrameAnalysis",
    "ActionDetail",
    "DiagnosisResult",
    "FetchConfig",
    "RemoteVideoFetcher",
    "VideoAnalyzer",
    "DiagnosisReporter",
]
```

- [ ] **Step 4: Commit**

```bash
git add scripts/liveness_diagnoser/ tests/scripts/liveness_diagnoser/
git commit -m "feat: create liveness_diagnoser package structure and models"
```

---

## Task 2: 实现 fetcher.py 远程拉取模块

**Files:**
- Create: `scripts/liveness_diagnoser/fetcher.py`
- Modify: `scripts/remote_fetch.py` (可选，如果需要复用)

- [ ] **Step 1: 编写测试**

```python
# tests/scripts/liveness_diagnoser/test_fetcher.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from scripts.liveness_diagnoser.fetcher import RemoteVideoFetcher
from scripts.liveness_diagnoser.models import FetchConfig


class TestRemoteVideoFetcher:
    """测试远程视频拉取器"""
    
    def test_init(self):
        """测试初始化"""
        config = FetchConfig(
            host="192.168.1.1",
            port=22,
            username="test",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="output/test"
        )
        fetcher = RemoteVideoFetcher(config)
        assert fetcher.config == config
        assert fetcher.ssh is None
        assert fetcher.sftp is None
    
    @patch('scripts.liveness_diagnoser.fetcher.SSHClient')
    def test_connect_success(self, mock_ssh_class):
        """测试连接成功"""
        config = FetchConfig(
            host="192.168.1.1",
            port=22,
            username="test",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="output/test"
        )
        fetcher = RemoteVideoFetcher(config)
        
        mock_ssh = MagicMock()
        mock_ssh_class.return_value = mock_ssh
        
        fetcher.connect()
        
        mock_ssh.set_missing_host_key_policy.assert_called_once()
        mock_ssh.connect.assert_called_once_with(
            hostname="192.168.1.1",
            port=22,
            username="test",
            key_filename="/path/to/key.pem",
            timeout=30,
            allow_agent=True,
            look_for_keys=True,
        )
    
    @patch('scripts.liveness_diagnoser.fetcher.LogParser')
    @patch('scripts.liveness_diagnoser.fetcher.Path')
    def test_find_video_by_task_id(self, mock_path_class, mock_parser_class):
        """测试通过 task_id 查找视频"""
        config = FetchConfig(
            host="192.168.1.1",
            port=22,
            username="test",
            key_filename="/path/to/key.pem",
            remote_log_path="/var/log/app.log",
            output_dir="output/test",
            task_id="test-task-123"
        )
        fetcher = RemoteVideoFetcher(config)
        
        # Mock SFTP 和日志下载
        fetcher.sftp = MagicMock()
        mock_path = MagicMock()
        mock_path_class.return_value = mock_path
        
        # Mock 日志解析结果
        mock_entry = MagicMock()
        mock_entry.task_id = "test-task-123"
        mock_entry.video_path = "/data/videos/test.webm"
        mock_entry.video_filename = "test.webm"
        mock_entry.actions = ["blink", "nod"]
        
        mock_parser = MagicMock()
        mock_parser.parse_file.return_value = [mock_entry]
        mock_parser_class.return_value = mock_parser
        
        result = fetcher.find_video_by_task_id("test-task-123")
        
        assert result is not None
        assert result.task_id == "test-task-123"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/scripts/liveness_diagnoser/test_fetcher.py -v
```

Expected: FAIL - "No module named 'scripts.liveness_diagnoser.fetcher'"

- [ ] **Step 3: 实现 fetcher.py**

```python
# scripts/liveness_diagnoser/fetcher.py
"""远程视频拉取模块"""
import logging
import sys
from pathlib import Path
from typing import Optional, List

from paramiko import SSHClient, AutoAddPolicy

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.log_parser import LogParser, VideoEntry
from scripts.ssh_config import SSHConfigManager, get_ssh_config
from .models import FetchConfig

logger = logging.getLogger(__name__)


class RemoteVideoFetcher:
    """远程视频拉取器 - 专门用于诊断工具"""
    
    def __init__(self, config: FetchConfig):
        self.config = config
        self.ssh: Optional[SSHClient] = None
        self.sftp = None
        self.video_entry: Optional[VideoEntry] = None
    
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
    
    def find_video_by_task_id(self, task_id: str) -> Optional[VideoEntry]:
        """通过 task_id 在远程日志中查找视频"""
        logger.info(f"在远程日志中查找 task_id: {task_id}")
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
            temp_log_path = tmp.name
        
        try:
            # 下载日志文件
            self.sftp.get(self.config.remote_log_path, temp_log_path)
            logger.info(f"✓ 日志已下载到临时文件")
            
            # 解析日志
            parser = LogParser()
            entries = parser.parse_file(temp_log_path)
            
            # 查找匹配的 task_id
            for entry in entries:
                if entry.task_id == task_id:
                    logger.info(f"✓ 找到匹配视频: {entry.video_filename}")
                    self.video_entry = entry
                    return entry
            
            logger.warning(f"未找到 task_id 为 {task_id} 的视频")
            return None
            
        finally:
            # 清理临时文件
            Path(temp_log_path).unlink(missing_ok=True)
    
    def download_video(self, remote_path: str, local_path: str) -> bool:
        """下载视频文件"""
        try:
            self.sftp.stat(remote_path)
            
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 检查本地是否已存在
            if Path(local_path).exists():
                local_size = Path(local_path).stat().st_size
                remote_size = self.sftp.stat(remote_path).st_size
                if local_size == remote_size:
                    logger.info(f"视频已存在，跳过下载: {Path(local_path).name}")
                    return True
            
            logger.info(f"下载视频: {Path(remote_path).name}")
            self.sftp.get(remote_path, local_path)
            logger.info(f"✓ 下载完成: {Path(local_path).name}")
            return True
            
        except FileNotFoundError:
            logger.error(f"远程视频文件不存在: {remote_path}")
            return False
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def fetch_for_diagnosis(self, task_id: str) -> Optional[str]:
        """
        为诊断拉取视频
        
        Returns:
            本地视频路径，如果失败返回 None
        """
        try:
            self.connect()
            
            # 查找视频
            entry = self.find_video_by_task_id(task_id)
            if not entry:
                return None
            
            # 确定本地保存路径
            local_path = Path(self.config.output_dir) / "videos" / entry.video_filename
            
            # 下载视频
            if self.download_video(entry.video_path, str(local_path)):
                return str(local_path)
            return None
            
        finally:
            self.disconnect()
    
    @classmethod
    def from_ssh_config(cls, config_name: str, task_id: Optional[str] = None) -> "RemoteVideoFetcher":
        """从 SSH 配置创建 fetcher"""
        ssh_config = get_ssh_config(config_name)
        if not ssh_config:
            raise ValueError(f"找不到 SSH 配置: {config_name}")
        
        config = FetchConfig(
            host=ssh_config.host,
            port=ssh_config.port,
            username=ssh_config.user,
            key_filename=ssh_config.pem_key,
            remote_log_path=ssh_config.remote_log or "/var/log/app.log",
            output_dir="output/diagnosis",
            task_id=task_id
        )
        
        return cls(config)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/scripts/liveness_diagnoser/test_fetcher.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/liveness_diagnoser/fetcher.py tests/scripts/liveness_diagnoser/test_fetcher.py
git commit -m "feat: implement remote video fetcher module"
```

---

## Task 3: 实现 analyzer.py 视频分析模块

**Files:**
- Create: `scripts/liveness_diagnoser/analyzer.py`

- [ ] **Step 1: 编写测试**

```python
# tests/scripts/liveness_diagnoser/test_analyzer.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from scripts.liveness_diagnoser.analyzer import VideoAnalyzer
from scripts.liveness_diagnoser.models import DiagnosisResult, VideoInfo


class TestVideoAnalyzer:
    """测试视频分析器"""
    
    @patch('scripts.liveness_diagnoser.analyzer.cv2.VideoCapture')
    @patch('scripts.liveness_diagnoser.analyzer.LivenessFusionEngine')
    @patch('scripts.liveness_diagnoser.analyzer.FastLivenessDetector')
    def test_analyze_video_basic(self, mock_detector_class, mock_engine_class, mock_cap_class):
        """测试基本视频分析"""
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {
            7: 30.0,  # FPS
            5: 100,   # 宽度
            6: 200,   # 高度
            8: 10,    # 总帧数
        }.get(x, 0)
        mock_cap.read.side_effect = [
            (True, MagicMock()),  # 第一帧
            (True, MagicMock()),  # 第二帧
            (False, None),        # 结束
        ]
        mock_cap_class.return_value = mock_cap
        
        # Mock Engine
        mock_engine = MagicMock()
        mock_engine.mp_detector.extract_landmarks.return_value = {
            'landmarks': MagicMock(),
            'quality_score': 0.8,
            'transform_matrix': None,
            'aspect_ratio': 1.0,
        }
        mock_engine.mp_detector.calculate_ear.return_value = 0.2
        mock_engine.mp_detector.calculate_mar.return_value = 0.3
        mock_engine.mp_detector.calculate_head_pose.return_value = (5.0, 10.0, 0.0)
        mock_engine.score_history = []
        mock_engine_class.return_value = mock_engine
        
        # Mock Detector
        mock_detector = MagicMock()
        mock_detector.detect_liveness.return_value = {'score': 0.6}
        mock_detector_class.return_value = mock_detector
        
        # 创建分析器并测试
        analyzer = VideoAnalyzer()
        result = analyzer.analyze("test.webm", actions=["blink"])
        
        assert isinstance(result, DiagnosisResult)
        assert result.video_info.filename == "test.webm"
        assert result.expected_actions == ["blink"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/scripts/liveness_diagnoser/test_analyzer.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 analyzer.py**

```python
# scripts/liveness_diagnoser/analyzer.py
"""视频分析模块 - 深度分析活体检测失效原因"""
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2

from vrlFace.liveness.config import LivenessConfig
from vrlFace.liveness.fusion_engine import LivenessFusionEngine
from vrlFace.liveness.fast_detector import FastLivenessDetector
from vrlFace.liveness.head_action import HeadActionDetector, HeadActionConfig
from vrlFace.liveness.utils import build_fast_detector_config

from .models import (
    VideoInfo,
    FrameAnalysis,
    ActionDetail,
    DiagnosisResult,
)


class VideoAnalyzer:
    """视频分析器 - 诊断活体检测失效原因"""
    
    def __init__(self, config: Optional[LivenessConfig] = None):
        self.config = config or LivenessConfig.video_fast_config()
        self.engine: Optional[LivenessFusionEngine] = None
        self.fast_detector: Optional[FastLivenessDetector] = None
        self.head_detector: Optional[HeadActionDetector] = None
    
    def _init_detectors(self):
        """初始化检测器"""
        self.engine = LivenessFusionEngine(self.config)
        self.fast_detector = FastLivenessDetector(
            **build_fast_detector_config(self.config)
        )
        self.head_detector = HeadActionDetector(
            HeadActionConfig(
                yaw_threshold=self.config.yaw_threshold,
                pitch_threshold=self.config.pitch_threshold,
                window_size=self.config.window_size,
            )
        )
    
    def _analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> FrameAnalysis:
        """分析单帧"""
        analysis = FrameAnalysis(frame_idx=frame_idx, has_face=False)
        
        # 缩小帧以加速处理
        max_w = self.config.max_width
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (max_w, int(frame.shape[0] * scale)))
        
        # 提取 landmarks
        lm_data = self.engine.mp_detector.extract_landmarks(frame)
        
        if lm_data is None:
            # 未检测到人脸，重置头部检测器
            self.head_detector.reset()
            return analysis
        
        analysis.has_face = True
        landmarks = lm_data['landmarks']
        aspect_ratio = lm_data.get('aspect_ratio', 1.0)
        analysis.quality_score = lm_data.get('quality_score', 0.0)
        
        # 计算 EAR, MAR
        analysis.ear = self.engine.mp_detector.calculate_ear(landmarks, aspect_ratio)
        analysis.mar = self.engine.mp_detector.calculate_mar(landmarks, aspect_ratio)
        
        # 计算头部姿态
        analysis.pitch, analysis.yaw, _ = self.engine.mp_detector.calculate_head_pose(
            landmarks, frame.shape, lm_data.get('transform_matrix')
        )
        
        # 检测头部动作
        analysis.head_action = self.head_detector.detect(analysis.pitch, analysis.yaw)
        
        # 检测眨眼/张嘴
        analysis.is_blink = analysis.ear < self.config.ear_threshold
        analysis.is_mouth_open = analysis.mar > self.config.mar_threshold
        
        # 计算运动分数
        fd_result = self.fast_detector.detect_liveness(
            landmarks, lm_data.get('frame_shape', frame.shape)
        )
        analysis.motion_score = fd_result['score']
        
        # 更新 engine 的 score_history 并计算平滑分
        self.engine.score_history.append(analysis.motion_score)
        analysis.smoothed_score = float(
            sum(list(self.engine.score_history)[-self.config.smooth_window:])
            / min(len(self.engine.score_history), self.config.smooth_window)
        )
        
        return analysis
    
    def _calculate_action_details(
        self,
        frame_data: List[FrameAnalysis],
        expected_actions: List[str]
    ) -> List[ActionDetail]:
        """计算动作检测详情"""
        action_names = {
            'blink': ('blink', '眨眼'),
            'mouth_open': ('mouth_open', '张嘴'),
            'nod': ('nod', '点头'),
            'shake_head': ('shake_head', '摇头'),
        }
        
        details = []
        
        for action in expected_actions:
            if action not in action_names:
                continue
            
            name, name_cn = action_names[action]
            
            # 统计动作事件
            if action == 'blink':
                events = sum(1 for f in frame_data if f.is_blink)
                avg_score = np.mean([f.ear for f in frame_data if f.has_face]) if frame_data else 0
            elif action == 'mouth_open':
                events = sum(1 for f in frame_data if f.is_mouth_open)
                avg_score = np.mean([f.mar for f in frame_data if f.has_face]) if frame_data else 0
            elif action == 'nod':
                events = sum(1 for f in frame_data if f.head_action in ['nod', 'nod_up', 'nod_down'])
                pitch_values = [f.pitch for f in frame_data if f.has_face]
                avg_score = np.mean(pitch_values) if pitch_values else 0
            elif action == 'shake_head':
                events = sum(1 for f in frame_data if f.head_action in ['shake_head', 'turn_left', 'turn_right'])
                yaw_values = [f.yaw for f in frame_data if f.has_face]
                avg_score = np.mean(yaw_values) if yaw_values else 0
            else:
                events = 0
                avg_score = 0
            
            frames_with_face = sum(1 for f in frame_data if f.has_face)
            
            # 计算置信度（简化版）
            event_rate = min(events / max(frames_with_face * 0.1, 1), 1.0)  # 期望至少 10% 的帧触发
            confidence = event_rate * 0.85 + (avg_score / 100) * 0.15 if action in ['nod', 'shake_head'] else event_rate
            
            # 判断是否通过
            passed = confidence >= 0.75  # action_threshold
            
            # 生成消息
            if events == 0:
                msg = f"未检测到{name_cn}"
            elif not passed:
                msg = f"动作幅度过小或置信度不足（触发率 {event_rate:.1%}）"
            else:
                msg = f"检测通过（触发率 {event_rate:.1%}）"
            
            details.append(ActionDetail(
                name=name,
                name_cn=name_cn,
                frames=frames_with_face,
                events=events,
                avg_score=float(avg_score),
                confidence=float(confidence),
                passed=passed,
                message=msg
            ))
        
        return details
    
    def _generate_suggestions(
        self,
        result: DiagnosisResult
    ) -> List[str]:
        """生成诊断建议"""
        suggestions = []
        
        # 活体判定失败建议
        if result.is_liveness == 0:
            suggestions.append(
                f"活体判定失败：最高平滑分 ({result.best_score:.4f}) < 阈值 ({result.threshold})"
            )
            suggestions.append(
                f"建议：降低 threshold 到 {result.best_score * 0.9:.2f} 或提高动作幅度"
            )
        
        # 人脸检出率建议
        if result.face_detection_rate < 0.5:
            suggestions.append(
                f"人脸检出率过低 ({result.face_detection_rate:.1%})，建议检查视频质量或光照条件"
            )
        
        # 动作检测建议
        for detail in result.action_details:
            if not detail.passed:
                if detail.name == 'nod':
                    pitch_values = [f.pitch for f in result.frame_data if f.has_face]
                    if pitch_values:
                        pitch_range = max(pitch_values) - min(pitch_values)
                        suggestions.append(
                            f"点头检测失败：Pitch 峰峰值 ({pitch_range:.1f}°) 不足，建议降低 pitch_threshold"
                        )
                elif detail.name == 'shake_head':
                    yaw_values = [f.yaw for f in result.frame_data if f.has_face]
                    if yaw_values:
                        yaw_range = max(yaw_values) - min(yaw_values)
                        suggestions.append(
                            f"转头检测失败：Yaw 峰峰值 ({yaw_range:.1f}°) 不足，建议降低 yaw_threshold"
                        )
                elif detail.name == 'blink':
                    ear_values = [f.ear for f in result.frame_data if f.has_face]
                    if ear_values:
                        min_ear = min(ear_values)
                        suggestions.append(
                            f"眨眼检测失败：最小 EAR ({min_ear:.3f}) 过高，建议降低 ear_threshold 到 {min_ear * 1.1:.3f}"
                        )
        
        return suggestions
    
    def analyze(
        self,
        video_path: str,
        actions: Optional[List[str]] = None,
        task_id: Optional[str] = None
    ) -> DiagnosisResult:
        """
        分析视频，诊断活体检测失效原因
        
        Args:
            video_path: 视频文件路径
            actions: 期望的动作列表
            task_id: 可选的任务 ID
        
        Returns:
            DiagnosisResult 诊断结果
        """
        actions = actions or ['blink', 'nod']
        
        # 初始化检测器
        self._init_detectors()
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 修复：如果帧数为负数，通过实际读取计算
        if total_frames <= 0:
            temp_cap = cv2.VideoCapture(video_path)
            total_frames = 0
            while temp_cap.read()[0]:
                total_frames += 1
            temp_cap.release()
        
        video_info = VideoInfo(
            path=video_path,
            filename=Path(video_path).name,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            duration=total_frames / fps if fps > 0 else 0
        )
        
        # 逐帧分析
        frame_data: List[FrameAnalysis] = []
        frame_idx = 0
        face_detected_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            analysis = self._analyze_frame(frame, frame_idx)
            frame_data.append(analysis)
            
            if analysis.has_face:
                face_detected_frames += 1
            
            frame_idx += 1
        
        cap.release()
        self.engine.close()
        
        # 计算活体判定
        smoothed_scores = [f.smoothed_score for f in frame_data if f.has_face]
        best_score = max(smoothed_scores) if smoothed_scores else 0.0
        is_liveness = 1 if best_score >= self.config.threshold else 0
        
        # 计算动作详情
        action_details = self._calculate_action_details(frame_data, actions)
        action_passed = all(d.passed for d in action_details)
        
        # 创建结果
        result = DiagnosisResult(
            video_info=video_info,
            task_id=task_id,
            expected_actions=actions,
            face_detected_frames=face_detected_frames,
            face_detection_rate=face_detected_frames / max(frame_idx, 1),
            is_liveness=is_liveness,
            best_score=best_score,
            threshold=self.config.threshold,
            action_passed=action_passed,
            action_details=action_details,
            frame_data=frame_data,
        )
        
        # 生成建议
        result.suggestions = self._generate_suggestions(result)
        
        return result
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/scripts/liveness_diagnoser/test_analyzer.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/liveness_diagnoser/analyzer.py tests/scripts/liveness_diagnoser/test_analyzer.py
git commit -m "feat: implement video analyzer module with detailed diagnosis"
```

---

## Task 4: 实现 reporter.py 报告生成模块

**Files:**
- Create: `scripts/liveness_diagnoser/reporter.py`
- Create: `scripts/liveness_diagnoser/templates/report.html`

- [ ] **Step 1: 编写测试**

```python
# tests/scripts/liveness_diagnoser/test_reporter.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from scripts.liveness_diagnoser.reporter import DiagnosisReporter
from scripts.liveness_diagnoser.models import (
    DiagnosisResult,
    VideoInfo,
    ActionDetail,
)


class TestDiagnosisReporter:
    """测试诊断报告生成器"""
    
    def test_generate_console_report(self):
        """测试生成控制台报告"""
        # 创建模拟结果
        result = DiagnosisResult(
            video_info=VideoInfo(
                path="test.webm",
                filename="test.webm",
                total_frames=100,
                fps=30.0,
                width=640,
                height=480,
                duration=3.33
            ),
            task_id="test-task",
            expected_actions=["blink"],
            face_detected_frames=90,
            face_detection_rate=0.9,
            is_liveness=0,
            best_score=0.28,
            threshold=0.35,
            action_passed=False,
            action_details=[
                ActionDetail(
                    name="blink",
                    name_cn="眨眼",
                    frames=90,
                    events=5,
                    avg_score=0.25,
                    confidence=0.6,
                    passed=False,
                    message="眨眼不足"
                )
            ],
            frame_data=[],
            suggestions=["建议降低阈值"]
        )
        
        reporter = DiagnosisReporter()
        report = reporter.generate_console_report(result)
        
        assert "test.webm" in report
        assert "0.28" in report
        assert "建议降低阈值" in report
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/scripts/liveness_diagnoser/test_reporter.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 reporter.py**

```python
# scripts/liveness_diagnoser/reporter.py
"""诊断报告生成模块"""
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from jinja2 import Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

from .models import DiagnosisResult

logger = logging.getLogger(__name__)


class DiagnosisReporter:
    """诊断报告生成器"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
    
    def generate_console_report(self, result: DiagnosisResult) -> str:
        """生成控制台文本报告"""
        lines = []
        
        # 标题
        lines.append("=" * 80)
        lines.append("活体检测失效诊断报告")
        lines.append("=" * 80)
        
        # 视频信息
        lines.append(f"\n【视频信息】")
        lines.append(f"  文件名: {result.video_info.filename}")
        lines.append(f"  任务ID: {result.task_id or 'N/A'}")
        lines.append(f"  总帧数: {result.video_info.total_frames}")
        lines.append(f"  FPS: {result.video_info.fps:.2f}")
        lines.append(f"  分辨率: {result.video_info.width}x{result.video_info.height}")
        lines.append(f"  时长: {result.video_info.duration:.2f}s")
        
        # 人脸检测统计
        lines.append(f"\n【人脸检测统计】")
        lines.append(f"  检出帧数: {result.face_detected_frames}/{result.video_info.total_frames}")
        lines.append(f"  检出率: {result.face_detection_rate:.1%}")
        
        # 活体判定
        lines.append(f"\n【活体判定】")
        lines.append(f"  判定结果: {'通过' if result.is_liveness else '失败'}")
        lines.append(f"  最高平滑分: {result.best_score:.4f}")
        lines.append(f"  阈值: {result.threshold}")
        lines.append(f"  判定逻辑: is_liveness = 1 if {result.best_score:.4f} >= {result.threshold} else 0")
        
        # 动作验证
        lines.append(f"\n【动作验证】")
        lines.append(f"  期望动作: {', '.join(result.expected_actions)}")
        lines.append(f"  整体通过: {'是' if result.action_passed else '否'}")
        lines.append("")
        
        for detail in result.action_details:
            lines.append(f"  {detail.name_cn} ({detail.name}):")
            lines.append(f"    检测帧数: {detail.frames}")
            lines.append(f"    触发次数: {detail.events}")
            lines.append(f"    平均分数: {detail.avg_score:.4f}")
            lines.append(f"    置信度: {detail.confidence:.2%}")
            lines.append(f"    是否通过: {'✓' if detail.passed else '✗'}")
            lines.append(f"    消息: {detail.message}")
            lines.append("")
        
        # 诊断建议
        lines.append("【诊断建议】")
        if result.suggestions:
            for i, suggestion in enumerate(result.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
        else:
            lines.append("  无特殊建议")
        
        # 页脚
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"分析时间: {result.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_json_report(self, result: DiagnosisResult) -> str:
        """生成 JSON 格式报告"""
        data = {
            "video_info": {
                "filename": result.video_info.filename,
                "path": result.video_info.path,
                "total_frames": result.video_info.total_frames,
                "fps": result.video_info.fps,
                "width": result.video_info.width,
                "height": result.video_info.height,
                "duration": result.video_info.duration,
            },
            "task_id": result.task_id,
            "expected_actions": result.expected_actions,
            "face_detection": {
                "detected_frames": result.face_detected_frames,
                "detection_rate": result.face_detection_rate,
            },
            "liveness": {
                "is_liveness": result.is_liveness,
                "best_score": result.best_score,
                "threshold": result.threshold,
            },
            "action_verification": {
                "passed": result.action_passed,
                "details": [
                    {
                        "name": d.name,
                        "name_cn": d.name_cn,
                        "frames": d.frames,
                        "events": d.events,
                        "avg_score": d.avg_score,
                        "confidence": d.confidence,
                        "passed": d.passed,
                        "message": d.message,
                    }
                    for d in result.action_details
                ],
            },
            "suggestions": result.suggestions,
            "analyzed_at": result.analyzed_at.isoformat(),
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def generate_html_report(self, result: DiagnosisResult) -> str:
        """生成 HTML 格式报告"""
        if not HAS_JINJA2:
            logger.warning("jinja2 未安装，生成简化 HTML")
            return self._generate_simple_html(result)
        
        template_path = self.template_dir / "report.html"
        if not template_path.exists():
            logger.warning(f"模板文件不存在: {template_path}，生成简化 HTML")
            return self._generate_simple_html(result)
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template = Template(f.read())
        
        # 准备图表数据
        chart_data = self._prepare_chart_data(result)
        
        return template.render(
            result=result,
            chart_data=json.dumps(chart_data),
            analyzed_at=result.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _generate_simple_html(self, result: DiagnosisResult) -> str:
        """生成简化 HTML 报告（无模板时）"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>活体检测诊断报告 - {result.video_info.filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .info {{ background: #e3f2fd; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .warning {{ background: #fff3e0; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .error {{ background: #ffebee; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .passed {{ color: #4CAF50; font-weight: bold; }}
        .failed {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>活体检测失效诊断报告</h1>
        
        <h2>视频信息</h2>
        <div class="info">
            <p><strong>文件名:</strong> {result.video_info.filename}</p>
            <p><strong>任务ID:</strong> {result.task_id or 'N/A'}</p>
            <p><strong>总帧数:</strong> {result.video_info.total_frames}</p>
            <p><strong>FPS:</strong> {result.video_info.fps:.2f}</p>
            <p><strong>分辨率:</strong> {result.video_info.width}x{result.video_info.height}</p>
            <p><strong>时长:</strong> {result.video_info.duration:.2f}s</p>
        </div>
        
        <h2>人脸检测统计</h2>
        <div class="info">
            <p><strong>检出帧数:</strong> {result.face_detected_frames}/{result.video_info.total_frames}</p>
            <p><strong>检出率:</strong> {result.face_detection_rate:.1%}</p>
        </div>
        
        <h2>活体判定</h2>
        <div class="{'error' if not result.is_liveness else 'info'}">
            <p><strong>判定结果:</strong> <span class="{'failed' if not result.is_liveness else 'passed'}">{'失败' if not result.is_liveness else '通过'}</span></p>
            <p><strong>最高平滑分:</strong> {result.best_score:.4f}</p>
            <p><strong>阈值:</strong> {result.threshold}</p>
        </div>
        
        <h2>动作验证</h2>
        <table>
            <tr>
                <th>动作</th>
                <th>检测帧数</th>
                <th>触发次数</th>
                <th>置信度</th>
                <th>结果</th>
                <th>消息</th>
            </tr>
"""
        
        for detail in result.action_details:
            html += f"""
            <tr>
                <td>{detail.name_cn}</td>
                <td>{detail.frames}</td>
                <td>{detail.events}</td>
                <td>{detail.confidence:.2%}</td>
                <td class="{'passed' if detail.passed else 'failed'}">{'通过' if detail.passed else '失败'}</td>
                <td>{detail.message}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>诊断建议</h2>
        <div class="warning">
            <ol>
"""
        
        for suggestion in result.suggestions:
            html += f"                <li>{suggestion}</li>\n"
        
        if not result.suggestions:
            html += "                <li>无特殊建议</li>\n"
        
        html += f"""            </ol>
        </div>
        
        <p style="margin-top: 40px; color: #999; text-align: center;">
            分析时间: {result.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""
        
        return html
    
    def _prepare_chart_data(self, result: DiagnosisResult) -> dict:
        """准备图表数据"""
        return {
            "frames": [f.frame_idx for f in result.frame_data],
            "smoothed_scores": [f.smoothed_score for f in result.frame_data],
            "motion_scores": [f.motion_score for f in result.frame_data],
            "quality_scores": [f.quality_score for f in result.frame_data],
            "ear_values": [f.ear for f in result.frame_data],
            "mar_values": [f.mar for f in result.frame_data],
            "pitch_values": [f.pitch for f in result.frame_data],
            "yaw_values": [f.yaw for f in result.frame_data],
            "threshold": result.threshold,
        }
    
    def save_report(
        self,
        result: DiagnosisResult,
        output_dir: str,
        formats: Optional[list] = None
    ) -> dict:
        """
        保存报告到文件
        
        Args:
            result: 诊断结果
            output_dir: 输出目录
            formats: 报告格式列表 ['console', 'json', 'html']，默认全部
        
        Returns:
            保存的文件路径字典
        """
        formats = formats or ['console', 'json', 'html']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        base_name = f"diagnosis_{result.video_info.filename}_{result.analyzed_at.strftime('%Y%m%d_%H%M%S')}"
        
        if 'console' in formats:
            txt_path = output_path / f"{base_name}.txt"
            txt_path.write_text(self.generate_console_report(result), encoding='utf-8')
            saved_files['console'] = str(txt_path)
            logger.info(f"✓ 文本报告已保存: {txt_path}")
        
        if 'json' in formats:
            json_path = output_path / f"{base_name}.json"
            json_path.write_text(self.generate_json_report(result), encoding='utf-8')
            saved_files['json'] = str(json_path)
            logger.info(f"✓ JSON 报告已保存: {json_path}")
        
        if 'html' in formats:
            html_path = output_path / f"{base_name}.html"
            html_path.write_text(self.generate_html_report(result), encoding='utf-8')
            saved_files['html'] = str(html_path)
            logger.info(f"✓ HTML 报告已保存: {html_path}")
        
        return saved_files
```

- [ ] **Step 4: 创建 HTML 模板**

```html
<!-- scripts/liveness_diagnoser/templates/report.html -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>活体检测诊断报告 - {{ result.video_info.filename }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f0f2f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        header h1 {
            margin: 0 0 10px 0;
            font-size: 28px;
        }
        header .meta {
            opacity: 0.9;
            font-size: 14px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .card h2 {
            margin: 0 0 20px 0;
            color: #444;
            font-size: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .info-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }
        .info-item label {
            display: block;
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .info-item value {
            display: block;
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .status-passed {
            background: #d4edda;
            color: #155724;
        }
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        .suggestions {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            border-radius: 0 8px 8px 0;
        }
        .suggestions h3 {
            margin: 0 0 10px 0;
            color: #856404;
        }
        .suggestions ol {
            margin: 0;
            padding-left: 20px;
        }
        .suggestions li {
            margin: 8px 0;
            color: #856404;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>活体检测失效诊断报告</h1>
            <div class="meta">
                {{ result.video_info.filename }} | 分析时间: {{ analyzed_at }}
            </div>
        </header>

        <div class="card">
            <h2>视频信息</h2>
            <div class="info-grid">
                <div class="info-item">
                    <label>文件名</label>
                    <value>{{ result.video_info.filename }}</value>
                </div>
                <div class="info-item">
                    <label>任务ID</label>
                    <value>{{ result.task_id or 'N/A' }}</value>
                </div>
                <div class="info-item">
                    <label>总帧数</label>
                    <value>{{ result.video_info.total_frames }}</value>
                </div>
                <div class="info-item">
                    <label>FPS</label>
                    <value>{{ "%.2f"|format(result.video_info.fps) }}</value>
                </div>
                <div class="info-item">
                    <label>分辨率</label>
                    <value>{{ result.video_info.width }}x{{ result.video_info.height }}</value>
                </div>
                <div class="info-item">
                    <label>时长</label>
                    <value>{{ "%.2f"|format(result.video_info.duration) }}s</value>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>人脸检测统计</h2>
            <div class="info-grid">
                <div class="info-item">
                    <label>检出帧数</label>
                    <value>{{ result.face_detected_frames }} / {{ result.video_info.total_frames }}</value>
                </div>
                <div class="info-item">
                    <label>检出率</label>
                    <value>{{ "%.1f"|format(result.face_detection_rate * 100) }}%</value>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>活体判定</h2>
            <div class="info-grid">
                <div class="info-item">
                    <label>判定结果</label>
                    <value>
                        <span class="status-badge {{ 'status-passed' if result.is_liveness else 'status-failed' }}">
                            {{ '通过' if result.is_liveness else '失败' }}
                        </span>
                    </value>
                </div>
                <div class="info-item">
                    <label>最高平滑分</label>
                    <value>{{ "%.4f"|format(result.best_score) }}</value>
                </div>
                <div class="info-item">
                    <label>阈值</label>
                    <value>{{ result.threshold }}</value>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="scoreChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>动作验证</h2>
            <table>
                <thead>
                    <tr>
                        <th>动作</th>
                        <th>检测帧数</th>
                        <th>触发次数</th>
                        <th>平均分数</th>
                        <th>置信度</th>
                        <th>结果</th>
                        <th>消息</th>
                    </tr>
                </thead>
                <tbody>
                    {% for detail in result.action_details %}
                    <tr>
                        <td>{{ detail.name_cn }} ({{ detail.name }})</td>
                        <td>{{ detail.frames }}</td>
                        <td>{{ detail.events }}</td>
                        <td>{{ "%.4f"|format(detail.avg_score) }}</td>
                        <td>{{ "%.2f"|format(detail.confidence * 100) }}%</td>
                        <td>
                            <span class="status-badge {{ 'status-passed' if detail.passed else 'status-failed' }}">
                                {{ '通过' if detail.passed else '失败' }}
                            </span>
                        </td>
                        <td>{{ detail.message }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if result.suggestions %}
        <div class="card">
            <div class="suggestions">
                <h3>诊断建议</h3>
                <ol>
                    {% for suggestion in result.suggestions %}
                    <li>{{ suggestion }}</li>
                    {% endfor %}
                </ol>
            </div>
        </div>
        {% endif %}

        <footer>
            Generated by Liveness Diagnoser v1.0
        </footer>
    </div>

    <script>
        const chartData = {{ chart_data | safe }};
        
        // 分数趋势图
        const ctx = document.getElementById('scoreChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.frames,
                datasets: [
                    {
                        label: '平滑分数',
                        data: chartData.smoothed_scores,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: '原始分数',
                        data: chartData.motion_scores,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: '阈值',
                        data: chartData.frames.map(() => chartData.threshold),
                        borderColor: 'rgb(255, 159, 64)',
                        borderDash: [5, 5],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '活体检测分数趋势'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    </script>
</body>
</html>
```

- [ ] **Step 5: 运行测试确认通过**

```bash
uv run pytest tests/scripts/liveness_diagnoser/test_reporter.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/liveness_diagnoser/reporter.py scripts/liveness_diagnoser/templates/report.html tests/scripts/liveness_diagnoser/test_reporter.py
git commit -m "feat: implement diagnosis reporter with HTML template"
```

---

## Task 5: 实现 main.py CLI 主入口

**Files:**
- Create: `scripts/liveness_diagnoser/main.py`

- [ ] **Step 1: 实现 main.py**

```python
#!/usr/bin/env python3
"""
Liveness Diagnoser - 活体检测失效诊断工具

用法:
    # 诊断特定任务（远程拉取 + 本地分析）
    uv run python -m scripts.liveness_diagnoser --config face-server --task-id xxx

    # 诊断最近 N 个失败视频
    uv run python -m scripts.liveness_diagnoser --config face-server --recent-failures 10

    # 诊断本地视频
    uv run python -m scripts.liveness_diagnoser --video path/to/video.webm --actions blink nod

    # 指定输出目录
    uv run python -m scripts.liveness_diagnoser ... --output-dir output/my_diagnosis
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.liveness_diagnoser import RemoteVideoFetcher, VideoAnalyzer, DiagnosisReporter
from scripts.liveness_diagnoser.models import FetchConfig
from scripts.ssh_config import get_ssh_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_local_video(
    video_path: str,
    actions: List[str],
    output_dir: str,
    task_id: Optional[str] = None
) -> bool:
    """诊断本地视频"""
    logger.info(f"开始诊断本地视频: {video_path}")
    
    try:
        # 分析视频
        analyzer = VideoAnalyzer()
        result = analyzer.analyze(video_path, actions=actions, task_id=task_id)
        
        # 生成报告
        reporter = DiagnosisReporter()
        
        # 打印控制台报告
        print("\n" + reporter.generate_console_report(result))
        
        # 保存报告文件
        saved_files = reporter.save_report(result, output_dir)
        
        logger.info(f"\n诊断完成！报告已保存到:")
        for fmt, path in saved_files.items():
            logger.info(f"  [{fmt}] {path}")
        
        return True
        
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        return False


def diagnose_remote_task(
    config_name: str,
    task_id: str,
    output_dir: str
) -> bool:
    """诊断远程任务"""
    logger.info(f"开始诊断远程任务: {task_id}")
    
    try:
        # 创建 fetcher
        fetcher = RemoteVideoFetcher.from_ssh_config(config_name, task_id)
        
        # 拉取视频
        video_path = fetcher.fetch_for_diagnosis(task_id)
        if not video_path:
            logger.error(f"无法获取任务 {task_id} 的视频")
            return False
        
        logger.info(f"视频已下载到: {video_path}")
        
        # 获取期望动作
        actions = fetcher.video_entry.actions if fetcher.video_entry else ['blink', 'nod']
        
        # 诊断视频
        return diagnose_local_video(video_path, actions, output_dir, task_id)
        
    except Exception as e:
        logger.error(f"远程诊断失败: {e}")
        return False


def diagnose_recent_failures(
    config_name: str,
    count: int,
    output_dir: str
) -> bool:
    """诊断最近失败的视频"""
    logger.info(f"开始诊断最近 {count} 个失败视频")
    
    # TODO: 实现从日志中筛选失败视频的逻辑
    logger.warning("此功能尚未实现，请先使用 --task-id 诊断特定任务")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="活体检测失效诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 诊断远程任务（使用 SSH 配置）
  uv run python -m scripts.liveness_diagnoser --config face-server --task-id abc123

  # 诊断本地视频
  uv run python -m scripts.liveness_diagnoser --video ./test.webm --actions blink nod

  # 指定输出目录
  uv run python -m scripts.liveness_diagnoser --config face-server --task-id abc123 --output-dir ./my_reports
        """
    )
    
    # 远程配置
    parser.add_argument('--config', help='SSH 配置名称（从 ssh-config.txt 读取）')
    parser.add_argument('--task-id', help='要诊断的任务 ID')
    parser.add_argument('--recent-failures', type=int, help='诊断最近 N 个失败视频')
    
    # 本地视频
    parser.add_argument('--video', help='本地视频文件路径')
    parser.add_argument('--actions', nargs='+', default=['blink', 'nod'],
                        help='期望的动作列表（默认: blink nod）')
    
    # 输出选项
    parser.add_argument('--output-dir', default='output/diagnosis',
                        help='报告输出目录（默认: output/diagnosis）')
    parser.add_argument('--formats', nargs='+', default=['console', 'json', 'html'],
                        choices=['console', 'json', 'html'],
                        help='报告格式（默认: console json html）')
    
    # 其他选项
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证参数
    if not args.config and not args.video:
        parser.error('必须指定 --config（远程诊断）或 --video（本地诊断）')
    
    if args.config and not args.task_id and not args.recent_failures:
        parser.error('使用 --config 时必须指定 --task-id 或 --recent-failures')
    
    # 执行诊断
    success = False
    
    if args.video:
        # 本地视频诊断
        success = diagnose_local_video(
            video_path=args.video,
            actions=args.actions,
            output_dir=args.output_dir,
            task_id=None
        )
    elif args.recent_failures:
        # 最近失败诊断
        success = diagnose_recent_failures(
            config_name=args.config,
            count=args.recent_failures,
            output_dir=args.output_dir
        )
    else:
        # 特定任务诊断
        success = diagnose_remote_task(
            config_name=args.config,
            task_id=args.task_id,
            output_dir=args.output_dir
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 添加 __main__.py 使包可执行**

```python
# scripts/liveness_diagnoser/__main__.py
"""使包可直接运行: python -m scripts.liveness_diagnoser"""
from .main import main

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/liveness_diagnoser/main.py scripts/liveness_diagnoser/__main__.py
git commit -m "feat: implement CLI main entry point"
```

---

## Task 6: 验证完整功能

- [ ] **Step 1: 运行帮助命令**

```bash
uv run python -m scripts.liveness_diagnoser --help
```

Expected: 显示帮助信息

- [ ] **Step 2: 验证包结构**

```bash
ls -la scripts/liveness_diagnoser/
```

Expected: 显示所有模块文件

- [ ] **Step 3: 运行测试套件**

```bash
uv run pytest tests/scripts/liveness_diagnoser/ -v
```

Expected: 所有测试通过

- [ ] **Step 4: Commit 最终版本**

```bash
git add -A
git commit -m "feat: complete liveness diagnoser tool implementation"
```

---

## 使用示例

```bash
# 1. 诊断本地视频
uv run python -m scripts.liveness_diagnoser \
  --video output/remote_fetch/videos/test.webm \
  --actions blink nod \
  --output-dir output/diagnosis

# 2. 诊断远程任务
uv run python -m scripts.liveness_diagnoser \
  --config face-server \
  --task-id abc-123-xyz \
  --output-dir output/diagnosis

# 3. 查看帮助
uv run python -m scripts.liveness_diagnoser --help
```

---

## 输出文件

诊断完成后会在 `output/diagnosis/` 目录生成：
- `diagnosis_<filename>_<timestamp>.txt` - 控制台文本报告
- `diagnosis_<filename>_<timestamp>.json` - JSON 格式数据
- `diagnosis_<filename>_<timestamp>.html` - 可视化 HTML 报告
