"""
视频活体检测分析器

对视频文件按动作时间段逐段分析：
- 全局活体判定（基于全程分数）
- 每个指定动作的逐段检测与置信度
- 人脸存在性与质量评估
"""

import cv2
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np

from .config import LivenessConfig
from .fusion_engine import LivenessFusionEngine
from .fast_detector import FastLivenessDetector
from .utils import build_fast_detector_config, resolve_current_action


# ---------------------------------------------------------------------------
# 动作名称映射（请求字段 → 内部 action 标签）
# ---------------------------------------------------------------------------

ACTION_ALIASES: Dict[str, List[str]] = {
    "blink":       ["blinking", "blink"],
    "mouth_open":  ["mouth_open"],
    "shake_head":  ["head_turn_left", "head_turn_right"],
    "nod":         ["head_nod_down", "head_nod_up"],
    "nod_down":    ["head_nod_down"],
    "nod_up":      ["head_nod_up"],
    "turn_left":   ["head_turn_left"],
    "turn_right":  ["head_turn_right"],
}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    action: str
    passed: bool
    confidence: float
    msg: str


@dataclass
class FaceInfo:
    confidence: float
    quality_score: float


@dataclass
class ActionVerifyResult:
    passed: bool
    required_actions: List[str]
    action_details: List[ActionResult]


@dataclass
class VideoLivenessResult:
    is_liveness: int                        # 1=通过, 0=不通过
    liveness_confidence: float
    is_face_exist: int                      # 1=有, 0=无
    face_info: Optional[FaceInfo]
    action_verify: ActionVerifyResult


# ---------------------------------------------------------------------------
# 分析器
# ---------------------------------------------------------------------------

class VideoLivenessAnalyzer:
    """
    视频活体分析器

    参数:
        liveness_config:  活体检测引擎配置（None 使用 realtime_config）
        liveness_threshold: 全局活体阈值（覆盖 config.threshold）
        action_threshold:   单动作通过阈值（事件触发率）
    """

    def __init__(
        self,
        liveness_config: Optional[LivenessConfig] = None,
        liveness_threshold: Optional[float] = None,
        action_threshold: float = 0.85,
    ):
        self.liveness_config = liveness_config or LivenessConfig.video_fast_config()
        if liveness_threshold is not None:
            self.liveness_config.threshold = liveness_threshold
        self.action_threshold = action_threshold

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(
        self,
        video_path: str,
        actions: List[str],
        max_video_duration: Optional[float] = None,
        per_action_timeout: Optional[float] = None,
    ) -> VideoLivenessResult:
        """
        分析视频，返回活体检测 + 每动作检测结果。

        Args:
            video_path:          视频文件路径
            actions:             要求的动作列表，e.g. ["blink","mouth_open","shake_head"]
            max_video_duration:  最长分析时长（秒），超过则截断
            per_action_timeout:  每动作时间窗口（秒），为 None 则平均分配
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._error_result(actions, "无法打开视频文件")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / source_fps if source_fps > 0 else 0

        # 计算实际分析帧数上限
        if max_video_duration and max_video_duration > 0:
            max_frames = min(total_frames, int(max_video_duration * source_fps))
        else:
            max_frames = total_frames

        # 计算每个动作的帧数窗口
        if per_action_timeout and per_action_timeout > 0 and actions:
            frames_per_action = int(per_action_timeout * source_fps)
        elif actions:
            frames_per_action = max_frames // len(actions) if len(actions) > 0 else max_frames
        else:
            frames_per_action = max_frames

        cap.release()

        # 执行分析
        return self._run_analysis(
            video_path=video_path,
            actions=actions,
            source_fps=source_fps,
            max_frames=max_frames,
            frames_per_action=frames_per_action,
        )

    # ------------------------------------------------------------------
    # 内部分析流程
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        video_path: str,
        actions: List[str],
        source_fps: float,
        max_frames: int,
        frames_per_action: int,
    ) -> VideoLivenessResult:
        """逐帧推理，同时收集全局活体分数和每动作事件。"""

        config = self.liveness_config

        # 初始化引擎（每次分析独立实例，避免状态污染）
        engine = LivenessFusionEngine(config)
        fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            engine.close()
            return self._error_result(actions, "无法打开视频文件")

        _max_w = config.max_width if config.max_width > 0 else 0

        # 解码线程
        frame_queue: queue.Queue = queue.Queue(maxsize=4)
        decode_stop = threading.Event()

        def _decode_worker():
            count = 0
            while not decode_stop.is_set() and count < max_frames:
                ret, frm = cap.read()
                if not ret:
                    break
                count += 1
                if _max_w > 0:
                    h0, w0 = frm.shape[:2]
                    if w0 > _max_w:
                        frm = cv2.resize(frm, (_max_w, int(h0 * _max_w / w0)))
                try:
                    frame_queue.put(frm, timeout=0.5)
                except queue.Full:
                    pass
            frame_queue.put(None)

        decode_thread = threading.Thread(target=_decode_worker, daemon=True)
        decode_thread.start()

        # 收集全局数据
        all_scores: List[float] = []
        all_quality: List[float] = []
        face_detected_count = 0
        total_count = 0

        # 每个动作的事件窗口：action_name → {"events": int, "frames": int, "scores": [float]}
        action_windows: List[Dict[str, Any]] = [
            {"name": a, "events": 0, "frames": 0, "scores": [], "quality": []}
            for a in actions
        ]

        frame_idx = 0
        while True:
            try:
                frame = frame_queue.get(timeout=2.0)
            except queue.Empty:
                break
            if frame is None:
                break

            frame_idx += 1
            total_count += 1

            # 决定当前帧属于哪个动作窗口
            action_slot = min(frame_idx // frames_per_action, len(actions) - 1) if actions else -1

            # 推理
            lm_data = engine.mp_detector.extract_landmarks(frame)

            if lm_data is None:
                all_scores.append(0.0)
                all_quality.append(0.0)
                if action_slot >= 0:
                    action_windows[action_slot]["frames"] += 1
                continue

            face_detected_count += 1
            landmarks = lm_data["landmarks"]
            quality_score = lm_data["quality_score"]

            fd_result = fast_detector.detect_liveness(
                landmarks, lm_data.get("frame_shape", frame.shape)
            )
            current_action = resolve_current_action(fd_result)

            motion_score = fd_result["score"]
            engine.score_history.append(motion_score)
            smoothed = float(
                sum(list(engine.score_history)[-config.smooth_window:])
                / min(len(engine.score_history), config.smooth_window)
            )

            all_scores.append(smoothed)
            all_quality.append(quality_score)

            if action_slot >= 0:
                slot = action_windows[action_slot]
                slot["frames"] += 1
                slot["scores"].append(smoothed)
                slot["quality"].append(quality_score)

                # 检查当前帧是否触发了该动作的事件
                expected_actions = ACTION_ALIASES.get(slot["name"], [slot["name"]])
                if current_action in expected_actions:
                    slot["events"] += 1

        decode_stop.set()
        decode_thread.join(timeout=2.0)
        cap.release()
        engine.close()

        # ------------------------------------------------------------------
        # 计算全局活体结果
        # ------------------------------------------------------------------
        is_face_exist = 1 if face_detected_count > 0 else 0
        avg_quality = float(np.mean(all_quality)) if all_quality else 0.0
        avg_face_conf = float(np.mean([q for q in all_quality if q > 0])) if face_detected_count > 0 else 0.0

        # 全局最高平滑分（用最高分而非末帧，避免末尾静止拉低）
        best_score = max(all_scores) if all_scores else 0.0
        is_liveness = 1 if best_score >= config.threshold else 0
        liveness_confidence = round(min(best_score / max(config.threshold, 1e-6), 1.0), 4)

        face_info = FaceInfo(
            confidence=round(avg_face_conf, 4),
            quality_score=round(avg_quality, 4),
        ) if is_face_exist else None

        # ------------------------------------------------------------------
        # 计算每动作结果
        # ------------------------------------------------------------------
        action_details = []
        for slot in action_windows:
            name = slot["name"]
            frames_in_slot = slot["frames"]
            events = slot["events"]
            slot_scores = slot["scores"]

            if frames_in_slot == 0:
                # 这段视频没有帧（视频太短）
                action_details.append(ActionResult(
                    action=name, passed=False, confidence=0.0,
                    msg="该动作时间段内无有效帧",
                ))
                continue

            # 动作置信度 = 事件触发帧率（事件数/该时间段总帧数）
            # 并用平均分辅助加权，避免事件极少但分数高被误判
            event_rate = events / max(frames_in_slot, 1)
            avg_slot_score = float(np.mean(slot_scores)) if slot_scores else 0.0
            # 综合置信度：事件率权重 0.7 + 平均分权重 0.3
            confidence = round(event_rate * 0.7 + avg_slot_score * 0.3, 4)
            passed = confidence >= self.action_threshold

            if passed:
                msg = f"检测到有效{_action_cn(name)}"
            elif events == 0:
                msg = f"未检测到{_action_cn(name)}"
            else:
                msg = f"动作幅度过小或置信度不足（触发率 {event_rate:.1%}）"

            action_details.append(ActionResult(
                action=name,
                passed=passed,
                confidence=confidence,
                msg=msg,
            ))

        all_actions_passed = all(d.passed for d in action_details) if action_details else True
        action_verify = ActionVerifyResult(
            passed=all_actions_passed,
            required_actions=actions,
            action_details=action_details,
        )

        return VideoLivenessResult(
            is_liveness=is_liveness,
            liveness_confidence=liveness_confidence,
            is_face_exist=is_face_exist,
            face_info=face_info,
            action_verify=action_verify,
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(actions: List[str], reason: str) -> "VideoLivenessResult":
        return VideoLivenessResult(
            is_liveness=0,
            liveness_confidence=0.0,
            is_face_exist=0,
            face_info=None,
            action_verify=ActionVerifyResult(
                passed=False,
                required_actions=actions,
                action_details=[
                    ActionResult(action=a, passed=False, confidence=0.0, msg=reason)
                    for a in actions
                ],
            ),
        )


def _action_cn(action: str) -> str:
    """动作英文 → 中文说明"""
    _MAP = {
        "blink": "眨眼",
        "mouth_open": "张嘴",
        "shake_head": "摇头",
        "nod": "点头",
        "nod_down": "低头",
        "nod_up": "抬头",
        "turn_left": "向左转头",
        "turn_right": "向右转头",
    }
    return _MAP.get(action, action)



