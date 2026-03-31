"""数据模型定义 - 活体诊断工具的数据结构。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoInfo:
    """视频基本信息。"""

    path: str
    filename: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration: float  # 视频时长（秒）


@dataclass
class FrameAnalysis:
    """单帧分析结果。"""

    frame_idx: int
    has_face: bool
    ear: Optional[float] = None  # 眨眼指数
    mar: Optional[float] = None  # 张嘴指数
    pitch: Optional[float] = None  # 俯仰角
    yaw: Optional[float] = None  # 偏航角
    motion_score: Optional[float] = None  # 运动分数
    smoothed_score: Optional[float] = None  # 平滑后的分数
    quality_score: Optional[float] = None  # 质量分数
    is_blink: bool = False
    is_mouth_open: bool = False
    head_action: Optional[str] = None  # "nod", "shake", None


@dataclass
class ActionDetail:
    """动作检测详情。"""

    name: str  # 动作英文名
    name_cn: str  # 动作中文名
    frames: list[int] = field(default_factory=list)  # 触发动作的帧索引
    events: int = 0  # 事件次数
    avg_score: float = 0.0  # 平均分数
    confidence: str = "low"  # "low", "medium", "high"
    passed: bool = False  # 是否通过
    message: str = ""  # 详细说明


@dataclass
class DiagnosisResult:
    """诊断结果。"""

    # 视频信息
    video_info: VideoInfo
    task_id: str | None = None  # 可选的任务 ID

    # 统计信息
    total_frames: int = 0
    frames_with_face: int = 0
    face_detection_rate: float = 0.0  # 人脸检测率

    # 动作统计
    blink_count: int = 0
    mouth_open_count: int = 0
    nod_count: int = 0
    shake_count: int = 0

    # 动作详情
    action_details: list[ActionDetail] = field(default_factory=list)

    # 逐帧数据
    frame_analyses: list[FrameAnalysis] = field(default_factory=list)

    # 质量评估
    avg_quality_score: float = 0.0
    quality_rating: str = "unknown"  # "poor", "fair", "good", "excellent"

    # 建议
    suggestions: list[str] = field(default_factory=list)

    # 总体判定
    overall_passed: bool = False
    overall_message: str = ""

    # 分析时间
    analysis_time: float = 0.0  # 分析耗时（秒）

    # 配置信息（用于报告）
    threshold: float = 0.0  # 活体阈值
    ear_threshold: float = 0.20  # 眨眼阈值
    mar_threshold: float = 0.55  # 张嘴阈值
    yaw_threshold: float = 15.0  # 摇头阈值
    pitch_threshold: float = 15.0  # 点头阈值


@dataclass
class FetchConfig:
    """远程拉取配置。"""

    host: str
    port: int
    username: str
    key_filename: str
    remote_log_path: str
    output_dir: str
    task_id: Optional[str] = None  # 可选的任务 ID
