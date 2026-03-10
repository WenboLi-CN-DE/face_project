"""
快速活体检测器 - 固定阈值状态机方案

检测逻辑：
- 眨眼：EAR 骤降至 ear_threshold 以下并迅速恢复（状态机）
- 张嘴：MAR 突增至 mar_threshold 以上并复位（状态机）
- 摇头：Yaw 在滑动窗口内发生正负交替大角度变化
- 点头/抬头：Pitch 在滑动窗口内发生显著单向或往复变化
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional
import time

from .head_action import HeadActionConfig, HeadActionDetector


class FastLivenessDetector:
    """快速活体检测器 - 固定阈值 + 滑动窗口，无基线，启动即可用"""

    # 眼睛关键点索引 (MediaPipe Face Mesh 478-point)
    # 顺序：[内眼角 (0), 上外 (1), 上中 (2), 外眼角 (3), 下中 (4), 下外 (5)]
    # 配对：上外 (1)-下外 (5), 上中 (2)-下中 (4)
    LEFT_EYE_INDICES = [133, 160, 158, 33, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # 嘴巴关键点索引
    # 顺序：[左角(0), 右角(1), 上左(2), 上右(3), 上中(4), 下中(5)]
    OUTER_LIPS_INDICES = [61, 291, 39, 269, 0, 17]

    # 头部姿态关键点
    HEAD_POSE_INDICES = {
        "nose_tip": 1,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
        "chin": 152,
    }

    # 3D 模型点 (用于 PnP 姿态估计)
    FACE_MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (-2.5, -1.5, -4.0),
            (2.5, -1.5, -4.0),
            (-3.0, 2.0, -2.0),
            (3.0, 2.0, -2.0),
            (0.0, 4.0, -5.0),
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        ear_threshold: float = 0.20,
        mar_threshold: float = 0.60,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0,
        window_size: int = 30,
        action_confirm_frames: int = 3,
        eye_open_threshold: float | None = None,
        eye_close_threshold: float | None = None,
    ):
        """
        初始化快速检测器

        Args:
            ear_threshold:       EAR 闭眼阈值，低于此值判定为闭眼（典型 0.18~0.22）
            mar_threshold:       MAR 张嘴阈值，高于此值判定为张嘴（实测：闭嘴<0.33，张嘴=0.93）
            yaw_threshold:       摇头判定的 Yaw 角峰峰值阈值（度），滑动窗口内
            pitch_threshold:     点头判定的 Pitch 角变化阈值（度），滑动窗口内
            window_size:         Yaw/Pitch 滑动窗口帧数
            action_confirm_frames: 动作需持续的最少帧数（防瞬时噪声）
        """
        # Eye thresholds with hysteresis:
        # - close_threshold:       ear < close_threshold -> closed
        # - open_threshold:        ear > open_threshold  -> open
        # If None: keep compatibility with the single ear_threshold.
        self.EYE_CLOSE_THRESHOLD = float(
            ear_threshold if eye_close_threshold is None else eye_close_threshold
        )
        self.EYE_OPEN_THRESHOLD = float(
            ear_threshold if eye_open_threshold is None else eye_open_threshold
        )

        # Backward-compatible alias used by older code / configs
        self.EAR_THRESHOLD = self.EYE_CLOSE_THRESHOLD
        self.MAR_THRESHOLD = mar_threshold
        self.YAW_THRESHOLD = yaw_threshold
        self.PITCH_THRESHOLD = pitch_threshold
        self.action_confirm_frames = action_confirm_frames

        # Yaw / Pitch sliding window is no longer used for action classification,
        # but we keep histories for motion_score.
        self.yaw_history: deque = deque(maxlen=window_size)
        self.pitch_history: deque = deque(maxlen=window_size)

        # New: event-based head action detector
        self._head_action_detector = HeadActionDetector(
            HeadActionConfig(
                yaw_threshold=float(yaw_threshold),
                pitch_threshold=float(pitch_threshold),
                confirm_frames=max(1, int(action_confirm_frames)),
            )
        )

        # EAR / MAR 历史（用于 motion score 计算）
        self.ear_history: deque = deque(maxlen=window_size)
        self.mar_history: deque = deque(maxlen=window_size)

        # ── 眨眼状态机 ──────────────────────────────────────────────
        # open → closed（EAR 低于 threshold） → open（EAR 恢复）计一次眨眼
        self.eye_state: str = "open"  # "open" | "closed"
        self._eye_closed_frames: int = 0
        self._blink_hold_frames: int = 0  # 眨眼事件 UI 保持帧数

        # ── 张嘴状态机 ──────────────────────────────────────────────
        # closed → open（MAR 高于 threshold） → closed 计一次张嘴
        self.mouth_state: str = "closed"  # "open" | "closed"
        self._mouth_open_frames: int = 0
        self._mouth_hold_frames: int = 0  # 张嘴事件 UI 保持帧数

        # ── 动作确认计数 ─────────────────────────────────────────────
        self.last_action: str = "none"
        self._action_frame_count: int = 0
        self._action_hold_frames: int = 0  # 动作确认后的保持帧计数
        self._head_cooldown: int = 0  # 头部动作触发后的 cooldown 帧计数

        # ── 事件分数提升（event score boost） ────────────────────────
        # 眨眼/张嘴事件触发后，向 score_history 注入高分，持续 _BOOST_FRAMES 帧，
        # 使 motion_score 在事件窗口内能超过活体判定阈值。
        _BOOST_FRAMES = 20   # 保持约 0.67s@30fps
        self._event_history: deque = deque(maxlen=_BOOST_FRAMES)
        self._BOOST_FRAMES = _BOOST_FRAMES
        self._BLINK_BOOST = 0.85   # 眨眼事件注入分
        self._MOUTH_BOOST = 0.85   # 张嘴事件注入分

        # ── 性能统计 ──────────────────────────────────────────────────
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._last_fps_time: float = time.time()

        # ── Pitch/Yaw EMA平滑 ─────────────────────────────────────────
        # 滤除因关键点抖动造成的逐帧噪声
        # 降低 alpha 值以减少跳动：0.35 -> 0.15
        self._pose_alpha: float = 0.15  # 越小越平滑
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_yaw: Optional[float] = None

        # 添加异常值过滤：记录上一帧姿态，过滤突变
        self._last_raw_pitch: Optional[float] = None
        self._last_raw_yaw: Optional[float] = None
        self._max_frame_delta: float = 10.0  # 单帧最大变化角度（度）

    # ------------------------------------------------------------------
    # 特征计算
    # ------------------------------------------------------------------

    def calculate_ear(self, landmarks: list, aspect_ratio: float = 1.0) -> float:
        """计算双眼平均 EAR (Eye Aspect Ratio)
        aspect_ratio = w/h（调用方传入图像宽高比）。
        MediaPipe 归一化坐标中 Δy 对应 Δy×h 像素、Δx 对应 Δx×w 像素，
        直接做欧氏距离时 y 方向被低估 w/h 倍（因为 h < w）。
        修正方法：将 y 乘以 aspect_ratio=w/h，使 x/y 回到等比像素空间。
        """
        left = self._single_ear(landmarks, self.LEFT_EYE_INDICES, aspect_ratio)
        right = self._single_ear(landmarks, self.RIGHT_EYE_INDICES, aspect_ratio)
        return (left + right) / 2.0

    def _single_ear(
        self, landmarks: list, indices: list, aspect_ratio: float = 1.0
    ) -> float:
        """计算单眼 EAR (Eye Aspect Ratio)
        EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        索引配对：[内眼角(0), 上外(1), 上中(2), 外眼角(3), 下中(4), 下外(5)]
        垂直距离：上外(1)-下外(5)、上中(2)-下中(4)
          左眼：160(上外)-144(下外)，158(上中)-153(下中)
          右眼：385(上外)-380(下外)，387(上中)-373(下中)
        宽高比修正：y *= aspect_ratio(w/h)，还原垂直像素比例。
        典型横屏 640×480：aspect_ratio=1.333，睁眼 EAR ≈ 0.28~0.35。
        """
        # y 方向乘以 w/h，放大垂直分量使其与水平分量等比
        pts = np.array(
            [[landmarks[i].x, landmarks[i].y * aspect_ratio] for i in indices]
        )
        v1 = np.linalg.norm(pts[1] - pts[5])  # 上外 - 下外
        v2 = np.linalg.norm(pts[2] - pts[4])  # 上中 - 下中
        h = np.linalg.norm(pts[0] - pts[3])  # 内眼角 - 外眼角（水平，不受影响）
        return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0

    def calculate_mar(self, landmarks: list, aspect_ratio: float = 1.0) -> float:
        """计算 MAR (Mouth Aspect Ratio)
        MAR = |上中 - 下中| / |左角 - 右角|
        同样需要 y 方向宽高比修正（y *= aspect_ratio=w/h）。
        """
        pts = np.array(
            [
                [landmarks[i].x, landmarks[i].y * aspect_ratio]
                for i in self.OUTER_LIPS_INDICES
            ]
        )
        v = np.linalg.norm(pts[4] - pts[5])  # 上中 - 下中（垂直）
        h = np.linalg.norm(pts[0] - pts[1])  # 左角 - 右角（水平）
        return float(v / h) if h > 1e-6 else 0.0

    @staticmethod
    def _rotation_matrix_to_euler_yxz(R: np.ndarray) -> Tuple[float, float, float]:
        """YXZ 欧拉角提取，返回 (pitch_deg, yaw_deg, roll_deg)。

        使用 YXZ 旋转顺序（先 Y-yaw, 再 X-pitch, 最后 Z-roll）避免 yaw-pitch 串扰：
          - yaw (Y轴): 左右转头
          - pitch (X轴): 上下点头
          - roll (Z轴): 左右歪头

        此顺序下，纯 yaw 旋转不会影响 pitch 分量，解决了左右摇头导致 pitch ±30° 的问题。
        """
        # 极分解：去除 scale/shear，仅保留纯旋转
        U, S, Vt = np.linalg.svd(R)
        R_pure = U @ Vt

        # 确保正旋转（det=1）
        if np.linalg.det(R_pure) < 0:
            R_pure[:, 2] *= -1

        # 强制正交性（处理数值误差）
        R_pure[0, :] /= np.linalg.norm(R_pure[0, :])
        R_pure[1, :] /= np.linalg.norm(R_pure[1, :])
        R_pure[2, :] = np.cross(R_pure[0, :], R_pure[1, :])

        # YXZ 欧拉角分解：R = Ry(yaw) * Rx(pitch) * Rz(roll)
        # R[1,2] = -sin(pitch)
        sin_pitch = -R_pure[1, 2]
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)

        if abs(sin_pitch) < 0.99999:  # 非 gimbal lock
            pitch = np.arcsin(sin_pitch)
            yaw = np.arctan2(R_pure[0, 2], R_pure[2, 2])
            roll = np.arctan2(R_pure[1, 0], R_pure[1, 1])
        else:  # gimbal lock (pitch ≈ ±90°)
            pitch = np.copysign(np.pi / 2, sin_pitch)
            yaw = np.arctan2(-R_pure[0, 1], R_pure[0, 0])
            roll = 0.0

        return (
            float(np.degrees(pitch)),
            float(np.degrees(yaw)),
            float(np.degrees(roll)),
        )

    def calculate_head_pose(
        self, landmarks: list, img_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """计算头部姿态，返回 (pitch, yaw, roll) 度数
        pitch > 0 = 低头；yaw > 0 = 向左转（相机视角）

        稳健性说明：
        - 口角点在说话/张嘴时会发生明显形变，若用于 PnP 容易把面部表情当成头部转动。
        - 这里改用更"刚性"的点：鼻尖 + 双眼外侧 + 下巴 + 双眼内侧，降低 pitch 抖动。
        """
        h, w = img_shape[:2]

        # Prefer rigid points over mouth corners
        idx = self.HEAD_POSE_INDICES
        rigid_image_points = np.array(
            [
                [landmarks[idx["nose_tip"]].x * w, landmarks[idx["nose_tip"]].y * h],
                [landmarks[idx["left_eye"]].x * w, landmarks[idx["left_eye"]].y * h],
                [landmarks[idx["right_eye"]].x * w, landmarks[idx["right_eye"]].y * h],
                # eye inner corners (more stable)
                [landmarks[133].x * w, landmarks[133].y * h],
                [landmarks[362].x * w, landmarks[362].y * h],
                [landmarks[idx["chin"]].x * w, landmarks[idx["chin"]].y * h],
            ],
            dtype=np.float64,
        )

        # 3D model points aligned with rigid_image_points above
        rigid_model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (-2.5, -1.5, -4.0),
                (2.5, -1.5, -4.0),
                (-2.0, -1.5, -4.0),
                (2.0, -1.5, -4.0),
                (0.0, 4.0, -5.0),
            ],
            dtype=np.float64,
        )

        focal_length = float(w)
        camera_matrix = np.array(
            [[focal_length, 0, w / 2.0], [0, focal_length, h / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))
        try:
            _, rvec, _ = cv2.solvePnP(
                rigid_model_points,
                rigid_image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
            R, _ = cv2.Rodrigues(rvec)

            # 使用 YXZ 欧拉角分解，避免 yaw-pitch 串扰
            return self._rotation_matrix_to_euler_yxz(R)
        except Exception:
            return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # 动作检测（状态机 / 滑动窗口）
    # ------------------------------------------------------------------

    def detect_blink(self, ear: float) -> bool:
        """眨眼检测：EAR 从“睁眼”掉到“闭眼”再恢复为“睁眼”计为一次眨眼。

        使用双阈值迟滞（hysteresis）避免在临界值附近抖动：
        - ear < EYE_CLOSE_THRESHOLD -> closed
        - ear > EYE_OPEN_THRESHOLD  -> open

        这也支持按经验将 >0.2 视为睁眼、<0.1 视为闭眼。
        """
        self.ear_history.append(ear)
        blinked = False

        if self.eye_state == "open":
            if ear < self.EYE_CLOSE_THRESHOLD:
                self.eye_state = "closed"
                self._eye_closed_frames = 1
        else:  # closed
            self._eye_closed_frames += 1
            if ear > self.EYE_OPEN_THRESHOLD:
                # 上限从 15 放宽到 60（约 2s@30fps），兼容缓慢闭眼或低帧率视频
                if 1 <= self._eye_closed_frames <= 60:
                    blinked = True
                self.eye_state = "open"
                self._eye_closed_frames = 0

        return blinked

    def detect_mouth(self, mar: float) -> bool:
        """张嘴检测：MAR 突增至阈值以上并复位，计为一次张嘴。
        状态机：closed → open → closed = mouth_open
        参考眨眼检测逻辑，确保状态转换正确。
        """
        self.mar_history.append(mar)
        moved = False

        if self.mouth_state == "closed":
            if mar > self.MAR_THRESHOLD:
                self.mouth_state = "open"
                self._mouth_open_frames = 1  # 初始化为1
        else:  # open
            self._mouth_open_frames += 1  # 持续张嘴时累加
            if mar <= self.MAR_THRESHOLD:
                # 上限从 30 放宽到 150（约 5s@30fps），兼容夸张张嘴或低帧率视频
                if 1 <= self._mouth_open_frames <= 150:
                    moved = True
                self.mouth_state = "closed"
                self._mouth_open_frames = 0

        return moved

    def detect_head_action(self, pitch: float, yaw: float) -> str:
        """头部动作检测（升级策略）。

        这保持了稳定的公共 API，同时将内部策略切换为基于事件的检测器
        （基线 + 确认 + 冷却）。
        """
        # Keep histories for scoring/debug.
        self.yaw_history.append(float(yaw))
        self.pitch_history.append(float(pitch))
        return self._head_action_detector.detect(pitch=float(pitch), yaw=float(yaw))

    # ------------------------------------------------------------------
    # 运动评分
    # ------------------------------------------------------------------

    def _calculate_motion_score(self) -> float:
        """综合活体运动分 = 窗口标准差分 + 事件提升分（取较大值融合）。

        窗口标准差分（std-based）：
          - EAR std × 10（眼部运动）
          - MAR std × 8 （嘴部运动）
          - (Yaw_range + Pitch_range) / 30（头部运动）
          三项均值，适合连续运动但对短暂事件不敏感。

        事件提升分（event-based）：
          - 眨眼/张嘴事件触发后向 _event_history 注入高分（0.85）
          - 取最近 _BOOST_FRAMES 帧事件分的均值
          - 与 std-based 分取 max，确保即使姿势静止时事件也能拉高得分
        """
        # ── std-based 部分 ────────────────────────────────────────────
        parts = []
        if len(self.ear_history) >= 5:
            parts.append(min(float(np.std(list(self.ear_history))) * 10.0, 1.0))
        else:
            parts.append(0.0)

        if len(self.mar_history) >= 5:
            parts.append(min(float(np.std(list(self.mar_history))) * 8.0, 1.0))
        else:
            parts.append(0.0)

        if len(self.yaw_history) >= 5:
            yaw_arr = np.array(list(self.yaw_history), dtype=np.float64)
            pitch_arr = np.array(list(self.pitch_history), dtype=np.float64)
            yaw_range = float(yaw_arr.max() - yaw_arr.min())
            pitch_range = float(pitch_arr.max() - pitch_arr.min())
            parts.append(min((yaw_range + pitch_range) / 30.0, 1.0))
        else:
            parts.append(0.0)

        std_score = float(np.mean(parts))

        # ── event-based 部分 ──────────────────────────────────────────
        if len(self._event_history) > 0:
            event_score = float(np.mean(list(self._event_history)))
        else:
            event_score = 0.0

        # 融合：取较大值，确保事件能单独把分数拉高
        return float(np.clip(max(std_score, event_score), 0.0, 1.0))

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self._last_fps_time = now

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def _normalize_hw(self, img_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Normalize an image shape into (h, w).

        Callers sometimes pass (w, h) or a raw cv2 shape (h, w, c).
        EAR/MAR in normalized landmark space needs a correct aspect_ratio=w/h.
        """
        if not img_shape or len(img_shape) < 2:
            return 1, 1

        a0 = int(img_shape[0])
        a1 = int(img_shape[1])

        # Common cv2 case: (h, w[, c])
        h_img, w_img = a0, a1

        # Heuristic fix: if it looks like (w, h) swap it.
        # Typical video: w>=h (landscape). If first dim is much larger, it's likely width.
        if a0 > a1:
            # Swap to (h,w)
            h_img, w_img = a1, a0

        h_img = max(h_img, 1)
        w_img = max(w_img, 1)
        return h_img, w_img

    def detect_liveness(
        self, landmarks: list, img_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        单帧活体检测。

        Args:
            landmarks:  MediaPipe face landmarks (478 点)
            img_shape:  图像形状 (h, w) 或 (h, w, c)

        Returns:
            结果字典，包含 ear / mar / yaw / pitch / 各动作标志 / score
        """
        self._update_fps()

        # 1. 计算宽高比（修正 MediaPipe 归一化坐标压缩）
        h_img, w_img = self._normalize_hw(img_shape)
        aspect_ratio = w_img / h_img if h_img > 0 else 1.0

        # 2. 计算原始特征（传入 aspect_ratio）
        ear = self.calculate_ear(landmarks, aspect_ratio)
        mar = self.calculate_mar(landmarks, aspect_ratio)

        # Use normalized (h,w) for head pose too
        pitch, yaw, roll = self.calculate_head_pose(landmarks, (h_img, w_img))

        # 异常值过滤：限制单帧最大变化，防止关键点误检导致的跳动
        if self._last_raw_pitch is not None:
            pitch_delta = pitch - self._last_raw_pitch
            yaw_delta = yaw - self._last_raw_yaw

            # 如果变化超过阈值，则限制到最大允许变化
            if abs(pitch_delta) > self._max_frame_delta:
                pitch = self._last_raw_pitch + np.sign(pitch_delta) * self._max_frame_delta
            if abs(yaw_delta) > self._max_frame_delta:
                yaw = self._last_raw_yaw + np.sign(yaw_delta) * self._max_frame_delta

        self._last_raw_pitch = pitch
        self._last_raw_yaw = yaw

        # EMA平滑：滤除关键点抖动造成的逐帧噪声
        a = self._pose_alpha
        if self._smoothed_pitch is None:
            self._smoothed_pitch = pitch
            self._smoothed_yaw = yaw
        else:
            self._smoothed_pitch = a * pitch + (1.0 - a) * float(self._smoothed_pitch)
            self._smoothed_yaw = a * yaw + (1.0 - a) * float(self._smoothed_yaw)
        pitch = float(self._smoothed_pitch)
        yaw = float(self._smoothed_yaw)

        # 3. 动作检测
        blink_detected = self.detect_blink(ear)
        mouth_detected = self.detect_mouth(mar)
        head_action_raw = self.detect_head_action(pitch, yaw)

        # 4. 眨眼 / 张嘴事件 hold（瞬时事件保持显示 _HOLD 帧，避免一闪而过）
        _HOLD = 12
        if blink_detected:
            self._blink_hold_frames = _HOLD
            # 注入事件提升分：连续 _BOOST_FRAMES 帧写入高分
            for _ in range(self._BOOST_FRAMES):
                self._event_history.append(self._BLINK_BOOST)
        elif self._blink_hold_frames > 0:
            self._blink_hold_frames -= 1

        if mouth_detected:
            self._mouth_hold_frames = _HOLD
            # 注入事件提升分
            for _ in range(self._BOOST_FRAMES):
                self._event_history.append(self._MOUTH_BOOST)
        elif self._mouth_hold_frames > 0:
            self._mouth_hold_frames -= 1

        # 事件历史衰减：无事件时每帧追加 0，让均值随时间下降
        if not blink_detected and not mouth_detected:
            self._event_history.append(0.0)

        # 5. 头部动作确认（锁定 + hold）
        # head_action_raw 在 cooldown 机制下通常是稀疏事件，
        # 直接确认并通过 hold 保持显示，避免"永远达不到连续确认帧数"。
        if head_action_raw != "none":
            self.last_action = head_action_raw
            self._action_frame_count = 0
            self._action_hold_frames = _HOLD
        else:
            if self.last_action != "none":
                if self._action_hold_frames > 0:
                    self._action_hold_frames -= 1
                else:
                    self.last_action = "none"
                    self._action_frame_count = 0
            else:
                self._action_frame_count = 0

        # 6. 综合运动评分
        score = self._calculate_motion_score()

        return {
            "score": score,
            "ear": ear,
            "mar": mar,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "blink_detected": blink_detected,  # 仅本帧触发
            "blink_active": self._blink_hold_frames > 0,  # hold 期间持续为 True
            "mouth_open": mouth_detected,  # 仅本帧触发
            "mouth_active": self._mouth_hold_frames > 0,  # hold 期间持续为 True
            "head_action": self.last_action,
            "is_blinking": self.eye_state == "closed",
            "is_mouth_open": self.mouth_state == "open",
            "fps": self.fps,
        }

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self):
        """重置所有状态"""
        self.ear_history.clear()
        self.mar_history.clear()
        self.yaw_history.clear()
        self.pitch_history.clear()
        self._event_history.clear()
        self.eye_state = "open"
        self._eye_closed_frames = 0
        self._blink_hold_frames = 0
        self.mouth_state = "closed"
        self._mouth_open_frames = 0
        self._mouth_hold_frames = 0
        self.last_action = "none"
        self._action_frame_count = 0
        self._action_hold_frames = 0
        self._head_cooldown = 0

        # New head action detector state
        if (
            hasattr(self, "_head_action_detector")
            and self._head_action_detector is not None
        ):
            self._head_action_detector.reset()

        # Reset EMA smoothed values
        self._smoothed_pitch = None
        self._smoothed_yaw = None

        # Reset outlier filter
        self._last_raw_pitch = None
        self._last_raw_yaw = None

        self.frame_count = 0
        self.fps = 0.0
        self._last_fps_time = time.time()
