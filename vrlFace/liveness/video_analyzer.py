"""
视频活体检测分析器

对视频文件按动作时间段逐段分析：
- 全局活体判定（基于全程分数）
- 每个指定动作的逐段检测与置信度
- 人脸存在性与质量评估
- 自动处理旋转视频
"""

import cv2
import threading
import queue
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np

from .config import LivenessConfig
from .fusion_engine import LivenessFusionEngine
from .fast_detector import FastLivenessDetector
from .utils import build_fast_detector_config, resolve_current_action
from .video_rotation import RotationHandler
from .benchmark_calibrator import BenchmarkCalibrator, BenchmarkConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 动作名称映射（请求字段 → 内部 action 标签）
# ---------------------------------------------------------------------------

ACTION_ALIASES: Dict[str, List[str]] = {
    "blink": ["blinking", "blink"],
    "mouth_open": ["mouth_open"],
    "shake_head": ["head_turn_left", "head_turn_right"],
    "nod": ["head_nod_down", "head_nod_up"],
    "nod_down": ["head_nod_down"],
    "nod_up": ["head_nod_up"],
    "turn_left": ["head_turn_left"],
    "turn_right": ["head_turn_right"],
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
    is_liveness: int  # 1=通过，0=不通过
    liveness_confidence: float
    is_face_exist: int  # 1=有，0=无
    face_info: Optional[FaceInfo]
    action_verify: ActionVerifyResult
    benchmark_verified: Optional[int] = (
        None  # 基准帧验证结果（1=通过，0=失败，None=未启用）
    )
    benchmark_details: Optional[Dict[str, Any]] = None  # 基准帧详细信息
    silent_detection: Optional[Dict[str, Any]] = None  # 静默检测结果
    reject_reason: Optional[str] = None  # 拒绝原因


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
        auto_rotate:        是否自动检测和修正视频旋转（默认 True）
        force_rotation:     强制指定旋转角度（0/90/180/270），覆盖自动检测
        enable_benchmark:   是否启用基准帧校准（防替换攻击）
        benchmark_config:   基准帧配置（None 使用默认配置）
    """

    def __init__(
        self,
        liveness_config: Optional[LivenessConfig] = None,
        liveness_threshold: Optional[float] = None,
        action_threshold: float = 0.50,
        auto_rotate: bool = True,
        force_rotation: Optional[int] = None,
        enable_benchmark: bool = True,
        benchmark_config: Optional[BenchmarkConfig] = None,
    ):
        self.liveness_config = liveness_config or LivenessConfig.video_fast_config()
        if liveness_threshold is not None:
            self.liveness_config.threshold = liveness_threshold
        self.action_threshold = action_threshold
        self.auto_rotate = auto_rotate
        self.force_rotation = force_rotation
        self.enable_benchmark = enable_benchmark
        self.benchmark_config = benchmark_config

        self._silent_detector = None
        self._frame_sampler = None

        if self.liveness_config.enable_silent_detection:
            from vrlFace.silent_liveness import SilentLivenessDetector
            from .frame_sampler import FrameSampler

            self._silent_detector = SilentLivenessDetector.get_instance()
            self._frame_sampler = FrameSampler()
            logger.info(
                "静默检测已启用（模式：%s）", self.liveness_config.silent_detection_mode
            )

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
        logger.info(f"开始分析视频: {video_path}")
        logger.info(f"要求动作: {actions}")

        # 静默检测（如果启用）
        silent_result = None
        if self.liveness_config.enable_silent_detection:
            silent_result = self._run_silent_detection(video_path)

            # 严格模式：检测失败立即返回
            if (
                self.liveness_config.silent_detection_mode == "strict"
                and not silent_result["passed"]
            ):
                return self._build_reject_result(silent_result)

        # 初始化旋转处理器
        rotation_handler = None
        if self.force_rotation is not None:
            # 强制指定旋转角度
            logger.info(f"强制旋转角度: {self.force_rotation}度")
            rotation_handler = RotationHandler(video_path, auto_detect=False)
            rotation_handler.rotation = self.force_rotation
        elif self.auto_rotate:
            # 自动检测旋转（包括基于人脸的检测）
            rotation_handler = RotationHandler(video_path, auto_detect=True)
            if rotation_handler.needs_rotation():
                logger.info(
                    f"检测到视频旋转: {rotation_handler.get_rotation()}度，将自动修正"
                )

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return self._error_result(actions, "无法打开视频文件")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # WebM 文件可能返回无效的帧数，通过实际读取来计算
        if total_frames <= 0:
            logger.warning(
                f"视频帧数无效 ({total_frames})，通过实际读取计算真实帧数..."
            )
            # 快速扫描视频获取真实帧数
            frame_count = 0
            while True:
                ret = cap.grab()  # grab() 比 read() 快，只解码不返回图像
                if not ret:
                    break
                frame_count += 1
            total_frames = frame_count
            logger.info(f"实际帧数: {total_frames}")

            # 重新打开视频以便后续处理
            cap.release()
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logger.error(f"重新打开视频失败: {video_path}")
                return self._error_result(actions, "无法打开视频文件")

        video_duration = total_frames / source_fps if source_fps > 0 else 0

        logger.info(
            f"视频属性: FPS={source_fps:.2f}, 总帧数={total_frames}, 分辨率={width}x{height}, 时长={video_duration:.2f}秒"
        )

        # 计算实际分析帧数上限
        if max_video_duration and max_video_duration > 0:
            max_frames = min(total_frames, int(max_video_duration * source_fps))
        else:
            max_frames = total_frames

        # 计算基准段帧数（如果启用基准帧校准）
        benchmark_frames = 0
        if self.enable_benchmark:
            benchmark_config = self.benchmark_config or BenchmarkConfig()
            benchmark_frames = int(benchmark_config.benchmark_duration * source_fps)

            if benchmark_frames >= max_frames * 0.5:
                benchmark_frames = int(max_frames * 0.2)
                logger.warning(
                    f"基准段过长（{benchmark_config.benchmark_duration}秒），自动调整为视频的20%（{benchmark_frames}帧）"
                )

            logger.info(
                f"基准段: {benchmark_frames} 帧 ({benchmark_frames / source_fps:.2f}秒)"
            )

        action_frames = max_frames - benchmark_frames
        if per_action_timeout and per_action_timeout > 0 and actions:
            frames_per_action = int(per_action_timeout * source_fps)
        elif actions:
            frames_per_action = (
                action_frames // len(actions) if len(actions) > 0 else action_frames
            )
        else:
            frames_per_action = action_frames

        logger.info(
            f"分析参数: max_frames={max_frames}, benchmark_frames={benchmark_frames}, "
            f"action_frames={action_frames}, frames_per_action={frames_per_action}"
        )

        cap.release()

        # 执行分析
        return self._run_analysis(
            video_path=video_path,
            actions=actions,
            source_fps=source_fps,
            max_frames=max_frames,
            frames_per_action=frames_per_action,
            benchmark_frames=benchmark_frames,
            rotation_handler=rotation_handler,
            silent_result=silent_result,
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
        benchmark_frames: int,
        rotation_handler: Optional[RotationHandler] = None,
        silent_result: Optional[Dict[str, Any]] = None,
    ) -> VideoLivenessResult:
        """逐帧推理，同时收集全局活体分数和每动作事件。

        Args:
            video_path: 视频文件路径
            actions: 动作列表
            source_fps: 视频帧率
            max_frames: 最大分析帧数
            frames_per_action: 每个动作的帧数
            benchmark_frames: 基准段帧数（前N帧用于基准采集，不分配给动作）
            rotation_handler: 旋转处理器
        """

        config = self.liveness_config
        logger.info(
            f"开始逐帧分析，配置: threshold={config.threshold}, max_width={config.max_width}"
        )

        # 初始化引擎（每次分析独立实例，避免状态污染）
        engine = LivenessFusionEngine(config)
        fast_detector = FastLivenessDetector(**build_fast_detector_config(config))

        # 初始化基准帧校准器
        calibrator: Optional[BenchmarkCalibrator] = None
        if self.enable_benchmark and self.benchmark_config is not None:
            calibrator = BenchmarkCalibrator(self.benchmark_config)
        elif self.enable_benchmark:
            calibrator = BenchmarkCalibrator()

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            engine.close()
            logger.error("重新打开视频失败")
            return self._error_result(actions, "无法打开视频文件")

        _max_w = config.max_width if config.max_width > 0 else 0

        # 解码线程
        frame_queue: queue.Queue = queue.Queue(maxsize=4)
        decode_stop = threading.Event()

        def _decode_worker():
            count = 0
            decode_success = 0
            while not decode_stop.is_set() and count < max_frames:
                ret, frm = cap.read()
                if not ret:
                    logger.warning(f"解码失败，已解码 {decode_success}/{count + 1} 帧")
                    break
                count += 1
                decode_success += 1

                # 应用旋转修正
                if rotation_handler and rotation_handler.needs_rotation():
                    frm = rotation_handler.process_frame(frm)

                if _max_w > 0:
                    h0, w0 = frm.shape[:2]
                    if w0 > _max_w:
                        frm = cv2.resize(frm, (_max_w, int(h0 * _max_w / w0)))
                try:
                    frame_queue.put(frm, timeout=0.5)
                except queue.Full:
                    pass
            logger.info(f"解码线程完成，成功解码 {decode_success} 帧")
            frame_queue.put(None)

        decode_thread = threading.Thread(target=_decode_worker, daemon=True)
        decode_thread.start()

        # 收集全局数据
        all_scores: List[float] = []  # 平滑分数（用于动作段统计）
        all_raw_scores: List[float] = []  # 原始分数（用于活体判定）
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

            # 决定当前帧属于哪个动作窗口（跳过基准段）
            # 使用交叉时间槽：允许 20% 重叠
            OVERLAP_RATIO = 0.2
            action_slot = -1
            in_cross_zone = False

            if frame_idx > benchmark_frames and actions:
                # 动作段内，计算相对于动作段起始的帧索引
                action_frame_idx = frame_idx - benchmark_frames
                base_slot = action_frame_idx // frames_per_action

                # 检查是否在交叉区域
                slot_end = (base_slot + 1) * frames_per_action
                overlap = int(frames_per_action * OVERLAP_RATIO)

                # 非最后一个动作：有向后重叠
                if base_slot < len(actions) - 1:
                    in_cross_zone = action_frame_idx >= slot_end - overlap

                action_slot = min(base_slot, len(actions) - 1)

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
                sum(list(engine.score_history)[-config.smooth_window :])
                / min(len(engine.score_history), config.smooth_window)
            )

            all_scores.append(smoothed)
            all_raw_scores.append(motion_score)  # 保存原始分数用于活体判定
            all_quality.append(quality_score)

            # 基准帧校准逻辑
            if calibrator is not None:
                embedding = lm_data.get("embedding")
                pitch = fd_result.get("pitch", 0.0)
                yaw = fd_result.get("yaw", 0.0)
                face_bbox = lm_data.get("face_bbox")

                if embedding is not None and face_bbox is not None:
                    if calibrator.is_collecting_benchmark():
                        calibrator.add_candidate_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            quality_score=quality_score,
                            face_bbox=face_bbox,
                            pitch=pitch,
                            yaw=yaw,
                            frame_index=frame_idx,
                        )
                    else:
                        calibrator.verify_frame(
                            embedding=embedding,
                            landmarks=np.array([[lm.x, lm.y] for lm in landmarks]),
                            pitch=pitch,
                            yaw=yaw,
                        )

            if action_slot >= 0:
                slot = action_windows[action_slot]
                slot["frames"] += 1
                slot["scores"].append(smoothed)
                slot["quality"].append(quality_score)

                # 检查当前帧是否触发了该动作的事件
                # 使用跨槽检测：根据检测到的动作类型匹配对应的时间槽
                expected_actions = ACTION_ALIASES.get(slot["name"], [slot["name"]])

                # 方案 1: 当前槽的动作匹配
                if current_action in expected_actions:
                    # 交叉区域内权重 0.8，严格区域内权重 1.0
                    weight = 0.8 if in_cross_zone else 1.0
                    slot["events"] += weight

                # 方案 2: 跨槽匹配 - 如果检测到的动作属于其他槽，也计入
                if current_action != "none" and current_action not in expected_actions:
                    for other_slot in action_windows:
                        other_actions = ACTION_ALIASES.get(
                            other_slot["name"], [other_slot["name"]]
                        )
                        if current_action in other_actions:
                            # 跨槽计入，权重 0.6
                            other_slot["events"] += 0.6
                            break

        decode_stop.set()
        decode_thread.join(timeout=2.0)
        cap.release()
        engine.close()

        logger.info(
            f"帧处理完成: 总帧数={total_count}, 检测到人脸={face_detected_count}, 人脸检出率={face_detected_count / max(total_count, 1):.2%}"
        )
        logger.info(
            f"分数统计: 最高分={max(all_scores) if all_scores else 0:.4f}, 平均分={np.mean(all_scores) if all_scores else 0:.4f}"
        )

        # ------------------------------------------------------------------
        # 计算全局活体结果
        # ------------------------------------------------------------------
        is_face_exist = 1 if face_detected_count > 0 else 0
        avg_quality = float(np.mean(all_quality)) if all_quality else 0.0
        avg_face_conf = (
            float(np.mean([q for q in all_quality if q > 0]))
            if face_detected_count > 0
            else 0.0
        )

        # 全局最高原始分（使用原始分数而非平滑分，避免平滑窗口拉低峰值）
        best_raw_score = max(all_raw_scores) if all_raw_scores else 0.0
        is_liveness = 1 if best_raw_score >= config.threshold else 0
        liveness_confidence = round(
            min(best_raw_score / max(config.threshold, 1e-6), 1.0), 4
        )

        logger.info(
            f"活体判定：is_liveness={is_liveness}, best_raw_score={best_raw_score:.4f}, threshold={config.threshold}, confidence={liveness_confidence}"
        )

        face_info = (
            FaceInfo(
                confidence=round(avg_face_conf, 4),
                quality_score=round(avg_quality, 4),
            )
            if is_face_exist
            else None
        )

        # ------------------------------------------------------------------
        # 计算每动作结果
        # ------------------------------------------------------------------
        action_details = []
        for slot in action_windows:
            name = slot["name"]
            frames_in_slot = slot["frames"]
            events = slot["events"]
            slot_scores = slot["scores"]

            logger.info(
                f"动作 '{name}': frames={frames_in_slot}, events={events}, avg_score={np.mean(slot_scores) if slot_scores else 0:.4f}"
            )

            if frames_in_slot == 0:
                # 这段视频没有帧（视频太短）
                action_details.append(
                    ActionResult(
                        action=name,
                        passed=False,
                        confidence=0.0,
                        msg="该动作时间段内无有效帧",
                    )
                )
                continue

            avg_slot_score = float(np.mean(slot_scores)) if slot_scores else 0.0

            expected_events = max(1, int(frames_in_slot / 90))
            event_rate = min(events / max(expected_events, 1), 1.0)

            if name in ["blink", "mouth_open"]:
                confidence = round(event_rate * 0.85 + avg_slot_score * 0.15, 4)
            else:
                confidence = round(event_rate * 0.82 + avg_slot_score * 0.18, 4)
            passed = confidence >= self.action_threshold

            if passed:
                msg = f"检测到有效{_action_cn(name)}"
            elif events == 0:
                msg = f"未检测到{_action_cn(name)}"
            else:
                msg = f"动作幅度过小或置信度不足（触发率 {event_rate:.1%}）"

            action_details.append(
                ActionResult(
                    action=name,
                    passed=passed,
                    confidence=confidence,
                    msg=msg,
                )
            )

        all_actions_passed = (
            all(d.passed for d in action_details) if action_details else True
        )
        action_verify = ActionVerifyResult(
            passed=all_actions_passed,
            required_actions=actions,
            action_details=action_details,
        )

        # 基准帧校准结果
        benchmark_verified: Optional[bool] = None
        benchmark_details: Optional[Dict[str, Any]] = None
        if calibrator is not None:
            if calibrator.is_ready():
                # 检查验证历史，确保全程没有换人
                verification_history = list(calibrator.verification_history)
                if verification_history:
                    all_verified = all(
                        v.get("is_same_person", False) for v in verification_history
                    )
                    benchmark_verified = all_verified
                    benchmark_details = {
                        "status": "verified",
                        "frames_verified": len(verification_history),
                        "all_same_person": all_verified,
                        "benchmark_quality": calibrator.benchmark_quality,
                    }
                else:
                    benchmark_verified = True
                    benchmark_details = {
                        "status": "no_verification_needed",
                        "reason": "视频太短，无验证历史",
                    }
            elif calibrator.is_collecting_benchmark():
                benchmark_verified = False
                benchmark_details = {
                    "status": "still_collecting",
                    "frames_collected": len(calibrator.benchmark_frames),
                }
            else:
                benchmark_verified = False
                benchmark_details = {
                    "status": "benchmark_failed",
                    "reason": "未能采集到足够的基准帧",
                }

        # 宽松模式：降低置信度
        final_confidence = liveness_confidence
        if (
            self.liveness_config.silent_detection_mode == "loose"
            and silent_result
            and not silent_result["passed"]
        ):
            final_confidence *= 0.8

        return VideoLivenessResult(
            is_liveness=is_liveness,
            liveness_confidence=final_confidence,
            is_face_exist=is_face_exist,
            face_info=face_info,
            action_verify=action_verify,
            benchmark_verified=1
            if benchmark_verified
            else (0 if benchmark_verified is False else None),
            benchmark_details=benchmark_details,
            silent_detection={
                "enabled": self.liveness_config.enable_silent_detection,
                "passed": silent_result["passed"] if silent_result else None,
                "confidence": silent_result["confidence"] if silent_result else None,
                "details": silent_result["details"] if silent_result else None,
            }
            if silent_result
            else None,
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

    def _run_silent_detection(self, video_path: str) -> Dict[str, Any]:
        """执行静默检测"""
        import tempfile
        import os

        logger.info("开始静默检测：采样关键帧...")

        keyframes = self._frame_sampler.sample_keyframes(
            video_path,
            num_frames=self.liveness_config.silent_sample_frames,
            min_quality=self.liveness_config.silent_min_quality,
            max_angle=self.liveness_config.silent_max_angle,
        )

        if not keyframes:
            return {
                "passed": False,
                "confidence": 0.0,
                "reject_reason": "no_quality_frames",
                "details": {},
            }

        results = []
        for i, frame in enumerate(keyframes):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, frame)

            try:
                result = self._silent_detector.detect(temp_path)
                results.append(result)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        passed = all(r["is_liveness"] == 1 for r in results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)

        reject_reason = None
        if not passed:
            for r in results:
                if r["reject_reason"]:
                    reject_reason = r["reject_reason"]
                    break

        return {
            "passed": passed,
            "confidence": avg_confidence,
            "reject_reason": reject_reason,
            "details": {"sampled_frames": len(keyframes), "frame_results": results},
        }

    def _build_reject_result(
        self, silent_result: Dict[str, Any]
    ) -> VideoLivenessResult:
        """构建静默检测拒绝结果"""
        return VideoLivenessResult(
            is_liveness=0,
            liveness_confidence=0.0,
            reject_reason=silent_result["reject_reason"],
            is_face_exist=1,
            face_info=FaceInfo(confidence=0.0, quality_score=0.0),
            action_verify=ActionVerifyResult(
                passed=False, required_actions=[], action_details=[]
            ),
            silent_detection={
                "enabled": True,
                "passed": False,
                "confidence": silent_result["confidence"],
                "details": silent_result["details"],
            },
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
