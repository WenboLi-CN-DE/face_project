"""
MediaPipe 动作活体检测器

检测逻辑：
- 眨眼：EAR 骤降至阈值以下并迅速恢复（固定阈值状态机）
- 张嘴：MAR 突增至阈值以上并复位（固定阈值状态机）
- 摇头：Yaw 在滑动窗口内峰峰值超过阈值
- 点头/抬头：Pitch 在滑动窗口内峰峰值超过阈值（Yaw 不显著时）

兼容 MediaPipe 0.10.x+ (Tasks API)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time

from .head_action import HeadActionConfig, HeadActionDetector

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# InsightFace 用于 embedding 提取（基准帧校准需要）
try:
    import insightface
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None

_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"
)


class MediaPipeLivenessDetector:
    """MediaPipe 动作活体检测器 — 固定阈值 + 滑动窗口，无动态基线"""

    # 眼睛关键点索引 (MediaPipe Face Mesh 478-point)
    # 顺序：[内眼角 (0), 上外 (1), 上中 (2), 外眼角 (3), 下中 (4), 下外 (5)]
    # 配对：上外 (1)-下外 (5), 上中 (2)-下中 (4)
    LEFT_EYE_INDICES = [133, 160, 158, 33, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # 嘴巴关键点索引
    # 顺序：[左角(0), 右角(1), 上左(2), 上右(3), 上中(4), 下中(5)]
    OUTER_LIPS_INDICES = [61, 291, 39, 269, 0, 17]

    # 头部姿态估计关键点
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
            (0.0, 0.0, 0.0),  # nose tip
            (-2.5, -1.5, -4.0),  # left eye
            (2.5, -1.5, -4.0),  # right eye
            (-3.0, 2.0, -2.0),  # left mouth
            (3.0, 2.0, -2.0),  # right mouth
            (0.0, 4.0, -5.0),  # chin
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        ear_threshold: float = 0.20,
        mar_threshold: float = 0.60,
        head_movement_threshold: float = 0.02,  # 保留参数名兼容调用方，内部未使用
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0,
        window_size: int = 30,
        max_faces: int = 1,
        model_path: Optional[str] = None,
        action_confirm_frames: int = 2,
    ):
        """
        Args:
            ear_threshold:          EAR 闭眼阈值（低于此值 → 闭眼）
            mar_threshold:          MAR 张嘴阈值（高于此值 → 张嘴，实测：闭嘴<0.33，张嘴=0.93）
            head_movement_threshold:保留参数，兼容旧调用方（未使用）
            yaw_threshold:          摇头：滑动窗口内 Yaw 峰峰值阈值（度）
            pitch_threshold:        点头：滑动窗口内 Pitch 峰峰值阈值（度）
            window_size:            Yaw/Pitch 滑动窗口帧数
            max_faces:              最大检测人脸数
            model_path:             face_landmarker.task 路径（None = 默认）
            action_confirm_frames:  头部动作需连续确认的帧数
        """
        model_path = model_path or _DEFAULT_MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker 模型未找到：{model_path}\n"
                "请下载：\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                f'-OutFile "{model_path}"'
            )

        # ── MediaPipe FaceLandmarker (VIDEO 跟踪模式) ─────────────────
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time_ms: int = int(time.time() * 1000)

        # ── 阈值 ──────────────────────────────────────────────────────
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.action_confirm_frames = action_confirm_frames

        # ── 滑动窗口 ──────────────────────────────────────────────────
        self.ear_history: deque = deque(maxlen=window_size)
        self.mar_history: deque = deque(maxlen=window_size)
        self.head_pose_history: deque = deque(maxlen=window_size)
        self.face_position_history: deque = deque(maxlen=window_size)

        # ── 眨眼状态机 ────────────────────────────────────────────────
        self.last_ear: Optional[float] = None
        self.eye_closed_frames: int = 0
        self.is_blinking: bool = False

        # ── 张嘴状态机 ────────────────────────────────────────────────
        self.last_mar: Optional[float] = None
        self._mouth_open_frames: int = 0
        self.is_mouth_open: bool = False

        # ── 头部姿态 ──────────────────────────────────────────────────
        self.current_yaw: float = 0.0
        self.current_pitch: float = 0.0

        # EMA 平滑系数（越小越平滑）：滤除逐帧数值噪声
        # 降低 alpha 值以减少跳动：0.35 -> 0.15
        self._pose_alpha: float = 0.15
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_yaw: Optional[float] = None

        # 添加异常值过滤：记录上一帧姿态，过滤突变

        # ── InsightFace (用于 embedding 提取) ─────────────────────────
        self._face_analyzer: Optional[Any] = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self._face_analyzer = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._face_analyzer.prepare(ctx_id=-1, det_size=(320, 320))
                print("✅ InsightFace 已初始化（支持基准帧校准）")
            except Exception as e:
                print(f"⚠️  InsightFace 初始化失败：{e}，基准帧校准功能将不可用")
        self._last_raw_pitch: Optional[float] = None
        self._last_raw_yaw: Optional[float] = None
        self._max_frame_delta: float = 10.0  # 单帧最大变化角度（度）

        # ── 动作确认 ──────────────────────────────────────────────────
        self.current_action: str = "none"
        self.confirmed_action: str = "none"
        self.action_frame_count: int = 0
        self._action_hold_frames: int = 0  # 动作确认后的保持帧计数
        self._head_action_cooldown: int = 0  # 头部动作触发后的 cooldown 帧计数

        # ── 帧计数 / FPS ──────────────────────────────────────────────
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.last_fps_time: float = time.time()

        # ── 上一帧推理结果缓存（跳帧复用） ───────────────────────────
        self._last_result: Optional[Dict[str, Any]] = None

        # New: shared event-based head action detector (unified behavior with FastLivenessDetector)
        self._head_action_detector = HeadActionDetector(
            HeadActionConfig(
                yaw_threshold=float(yaw_threshold),
                pitch_threshold=float(pitch_threshold),
                confirm_frames=max(1, int(action_confirm_frames)),
            )
        )

    # ------------------------------------------------------------------
    # 特征计算
    # ------------------------------------------------------------------

    def calculate_ear(self, landmarks: list, aspect_ratio: float = 1.0) -> float:
        """双眼平均 EAR，aspect_ratio=w/h。
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
        """单眼 EAR = (|p1-p5| + |p2-p4|) / (2*|p0-p3|)
        索引配对：[内眼角(0), 上外(1), 上中(2), 外眼角(3), 下中(4), 下外(5)]
        垂直距离：上外(1)-下外(5)、上中(2)-下中(4)
          左眼：160(上外)-144(下外)，158(上中)-153(下中)
          右眼：385(上外)-380(下外)，387(上中)-373(下中)
        宽高比修正：y *= aspect_ratio(w/h)，还原垂直像素比例。
        典型横屏 640×480：aspect_ratio=1.333，睁眼 EAR ≈ 0.28~0.35。
        """
        pts = np.array(
            [[landmarks[i].x, landmarks[i].y * aspect_ratio] for i in indices]
        )
        v1 = np.linalg.norm(pts[1] - pts[5])  # 上外 - 下外
        v2 = np.linalg.norm(pts[2] - pts[4])  # 上中 - 下中
        h = np.linalg.norm(pts[0] - pts[3])  # 内眼角 - 外眼角（水平，不受影响）
        return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0

    def calculate_mar(self, landmarks: list, aspect_ratio: float = 1.0) -> float:
        """MAR = |上中 - 下中| / |左角 - 右角|
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

    # 降级 EPnP 的 3D 模型点（鼻尖/前额/左颧/右颧/下巴/鼻梁），
    # 空间分布均匀，条件数远优于之前的眼部聚集方案。
    _EPnP_MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),  # 1   鼻尖
            (0.0, -4.5, -1.5),  # 10  前额中心（鼻梁上方）
            (-4.5, -1.5, -3.5),  # 234 左颧（脸颊外侧）
            (4.5, -1.5, -3.5),  # 454 右颧
            (-2.5, 2.5, -2.5),  # 61  左嘴角（表情影响小）
            (2.5, 2.5, -2.5),  # 291 右嘴角
            (0.0, 5.5, -4.5),  # 152 下巴
        ],
        dtype=np.float64,
    )
    _EPnP_IMAGE_INDICES = [1, 10, 234, 454, 61, 291, 152]

    @staticmethod
    def _rotation_matrix_to_euler_yxz(R: np.ndarray) -> Tuple[float, float, float]:
        """YXZ 欧拉角提取，返回 (pitch_deg, yaw_deg, roll_deg)。

        MediaPipe 使用标准相机坐标系（OpenCV 风格）：
          X 向右，Y 向下，Z 向前（远离相机）

        YXZ 旋转顺序（先 Y-yaw, 再 X-pitch, 最后 Z-roll）能最大程度减少 yaw-pitch 串扰：
          - yaw (Y轴): 左右转头
          - pitch (X轴): 上下点头
          - roll (Z轴): 左右歪头

        此顺序下，纯 yaw 旋转不会影响 pitch 分量（反之亦然），
        解决了 ZYX 顺序中左右摇头导致 pitch 变化 ±30° 的问题。
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
        # R[1,2] = -sin(pitch), 检测 gimbal lock
        sin_pitch = -R_pure[1, 2]
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)

        if abs(sin_pitch) < 0.99999:  # 非 gimbal lock
            pitch = np.arcsin(sin_pitch)
            yaw = np.arctan2(R_pure[0, 2], R_pure[2, 2])
            roll = np.arctan2(R_pure[1, 0], R_pure[1, 1])
        else:  # gimbal lock (pitch ≈ ±90°)
            pitch = np.copysign(np.pi / 2, sin_pitch)
            # 当 pitch=±90° 时，yaw 和 roll 退化到同一自由度
            yaw = np.arctan2(-R_pure[0, 1], R_pure[0, 0])
            roll = 0.0

        return (
            float(np.degrees(pitch)),
            float(np.degrees(yaw)),
            float(np.degrees(roll)),
        )

    def calculate_head_pose(
        self,
        landmarks: list,
        img_shape: Tuple[int, int],
        transformation_matrix=None,
    ) -> Tuple[float, float, float]:
        """返回 (pitch, yaw, roll) 度数。
        优先使用 MediaPipe 提供的 facial_transformation_matrixes（极分解去 scale）。
        否则降级到 SOLVEPNP_EPNP（使用空间分布均匀的7点模型，条件数更好）。
        pitch > 0 = 低头；yaw > 0 = 向右转（MediaPipe 坐标系）
        """
        # ── 优先路径：MediaPipe 变换矩阵（极分解去 scale） ──────────
        if transformation_matrix is not None:
            try:
                R_raw = np.array(transformation_matrix, dtype=np.float64)[:3, :3]
                return self._rotation_matrix_to_euler_yxz(R_raw)
            except Exception:
                pass

        # ── 降级路径：EPnP（7点均匀分布模型） ───────────────────────
        h_img, w_img = img_shape[:2]

        image_points = np.array(
            [
                [landmarks[idx].x * w_img, landmarks[idx].y * h_img]
                for idx in self._EPnP_IMAGE_INDICES
            ],
            dtype=np.float64,
        )

        focal = float(w_img)
        camera_matrix = np.array(
            [[focal, 0, w_img / 2.0], [0, focal, h_img / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))
        try:
            _, rvec, _ = cv2.solvePnP(
                self._EPnP_MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
            R, _ = cv2.Rodrigues(rvec)
            return self._rotation_matrix_to_euler_yxz(R)
        except Exception:
            return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # 动作检测
    # ------------------------------------------------------------------

    def detect_blink(self, ear: float) -> bool:
        """眨眼：EAR 骤降至阈值以下并迅速恢复（状态机）。
        open → closed（EAR < threshold） → open = blink
        闭合帧数限制在 1~8 帧（约 0.03~0.27s @ 30fps），排除长时间闭眼/遮挡误报。
        """
        self.ear_history.append(ear)
        blinked = False

        if ear < self.ear_threshold:
            self.is_blinking = True
            self.eye_closed_frames += 1
        else:
            if self.is_blinking:
                # 上限从 15 放宽到 60（约 2s@30fps），兼容缓慢闭眼或低帧率视频
                if 1 <= self.eye_closed_frames <= 60:
                    blinked = True
                self.eye_closed_frames = 0
            self.is_blinking = False

        self.last_ear = ear
        return blinked

    def detect_mouth_movement(self, mar: float) -> bool:
        """张嘴：MAR 突增至阈值以上并复位（状态机）。
        closed → open（MAR > threshold） → closed = mouth_open_event
        张开帧数限制在 1~30 帧，排除长时间张嘴不动。
        参考眨眼检测逻辑，确保状态转换正确。
        """
        self.mar_history.append(mar)
        moved = False

        if not self.is_mouth_open:  # closed 状态
            if mar > self.mar_threshold:
                self.is_mouth_open = True
                self._mouth_open_frames = 1  # 初始化为1
        else:  # open 状态
            self._mouth_open_frames += 1  # 持续张嘴时累加
            if mar <= self.mar_threshold:
                # 上限从 30 放宽到 150（约 5s@30fps），兼容夸张张嘴或低帧率视频
                if 1 <= self._mouth_open_frames <= 150:
                    moved = True
                self._mouth_open_frames = 0
                self.is_mouth_open = False

        self.last_mar = mar
        return moved

    def detect_head_action(self, pitch: float, yaw: float) -> str:
        """Head action detection (unified strategy).

        Uses the shared event-based detector (baseline + confirmation + cooldown)
        to ensure the same behavior across all pipelines.
        """
        # keep history for motion_score/debug
        self.head_pose_history.append((float(pitch), float(yaw)))
        return self._head_action_detector.detect(pitch=float(pitch), yaw=float(yaw))

    def update_current_action(self, head_action_raw: str) -> str:
        """对头部动作做帧计数确认，防止瞬时噪声误报。
        眨眼 / 张嘴直接由状态机边缘触发，不走此确认。

        修复：
        - 动作一旦确认后锁定 confirmed_action；
        - raw 持续非 none 时续期 hold（重置为 _HOLD_FRAMES），保持标签稳定；
        - 只有当 raw 回到 "none" 后才开始消耗 hold，hold 归零再清除标签。
        """
        _HOLD_FRAMES = 8  # 动作确认后保持显示的最少帧数

        if head_action_raw != "none":
            if self.confirmed_action == "none":
                # 尚未确认，累积计数
                self.action_frame_count += 1
                if self.action_frame_count >= self.action_confirm_frames:
                    self.confirmed_action = head_action_raw
                    self._action_hold_frames = _HOLD_FRAMES
            else:
                # ★ 修复1：已确认，raw 仍有效 → 续期，不倒计
                self._action_hold_frames = _HOLD_FRAMES
        else:
            if self.confirmed_action != "none":
                # raw 已回到 none，开始消耗 hold
                if self._action_hold_frames > 0:
                    self._action_hold_frames -= 1
                else:
                    self.confirmed_action = "none"
                    self.action_frame_count = 0
            else:
                self.action_frame_count = 0

        self.current_action = self.confirmed_action
        return self.current_action

    # ------------------------------------------------------------------
    # 运动评分
    # ------------------------------------------------------------------

    def _calculate_motion_score(self) -> float:
        """综合 EAR / MAR / Yaw+Pitch 标准差/范围计算活体运动分 [0, 1]"""
        scores = []

        if len(self.ear_history) >= 5:
            scores.append(min(float(np.std(self.ear_history)) * 10.0, 1.0))
        else:
            scores.append(0.0)

        if len(self.mar_history) >= 5:
            scores.append(min(float(np.std(self.mar_history)) * 8.0, 1.0))
        else:
            scores.append(0.0)

        if len(self.head_pose_history) >= 5:
            yaw_arr = np.array([h[1] for h in self.head_pose_history])
            pitch_arr = np.array([h[0] for h in self.head_pose_history])
            yaw_range = float(yaw_arr.max() - yaw_arr.min())
            pitch_range = float(pitch_arr.max() - pitch_arr.min())
            scores.append(min((yaw_range + pitch_range) / 30.0, 1.0))
        else:
            scores.append(0.0)

        return float(np.clip(float(np.mean(scores)), 0.0, 1.0))

    def _calculate_blur(self, frame: np.ndarray) -> float:
        """
        计算模糊度 (Laplacian 方差)

        Args:
            frame: BGR 图像

        Returns:
            模糊度分数 (0-1, 越大越清晰)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 归一化：variance < 50 很模糊，variance > 200 很清晰
        blur_score = min(variance / 200.0, 1.0)
        return blur_score

    def _calculate_face_angle(self, landmarks: list) -> float:
        """基于 MediaPipe 关键点计算人脸偏转角度分数"""
        try:
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]

            # 安全检查
            if not all(
                hasattr(lm, "x") and hasattr(lm, "y")
                for lm in [left_eye, right_eye, nose_tip]
            ):
                return 0.5

            # 计算眼睛连线的水平角度
            dx = float(right_eye.x - left_eye.x)
            dy = float(right_eye.y - left_eye.y)
            roll_angle = abs(np.degrees(np.arctan2(dy, dx)))

            # 计算鼻子相对于眼睛的垂直距离（估算 pitch）
            eye_mid_y = (float(left_eye.y) + float(right_eye.y)) / 2
            pitch_estimate = abs(float(nose_tip.y) - eye_mid_y) / max(abs(dx), 0.001)

            # 综合角度：roll + pitch 的近似
            total_angle = roll_angle + (pitch_estimate * 30)

            # 0-15 度：1.0, 15-30 度：线性下降，>45 度：0.3
            if total_angle <= 15:
                return 1.0
            elif total_angle <= 30:
                return 1.0 - (total_angle - 15) / 15 * 0.4
            elif total_angle <= 45:
                return 0.6 - (total_angle - 30) / 15 * 0.3
            else:
                return 0.3
        except (AttributeError, IndexError, ZeroDivisionError):
            return 0.5  # 默认中等分数

    def _calculate_quality_score(self, frame: np.ndarray, landmarks: list) -> float:
        """
        基于 4 个维度计算人脸质量分数 [0, 1]

        维度与权重：
        - 人脸尺寸 (40%): 人脸占画面比例
        - 亮度 (25%): 光照适中程度
        - 模糊度 (20%): Laplacian 方差
        - 角度 (15%): 人脸偏转角度
        """
        h_img, w_img = frame.shape[:2]

        # 1. 人脸尺寸分数 (40%)
        face_width = abs(landmarks[454].x - landmarks[234].x) * w_img
        face_height = abs(landmarks[152].y - landmarks[10].y) * h_img
        face_area = face_width * face_height
        frame_area = h_img * w_img
        size_ratio = face_area / frame_area if frame_area > 0 else 0

        if size_ratio < 0.01:
            size_score = 0.3
        elif size_ratio < 0.05:
            size_score = 0.6
        elif size_ratio < 0.25:
            size_score = 1.0
        else:
            size_score = 0.7

        # 2. 亮度分数 (25%)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        if 0.25 <= brightness <= 0.75:
            brightness_score = 1.0
        elif brightness < 0.25:
            brightness_score = brightness / 0.25
        else:
            brightness_score = (1.0 - brightness) / 0.25

        # 3. 模糊度分数 (20%)
        blur_score = self._calculate_blur(frame)

        # 4. 角度分数 (15%)
        angle_score = self._calculate_face_angle(landmarks)

        # 加权平均
        weights = [0.40, 0.25, 0.20, 0.15]
        scores = [size_score, brightness_score, blur_score, angle_score]
        quality_score = sum(s * w for s, w in zip(scores, weights))

        return float(np.clip(quality_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------

    def _update_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------

    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        仅提取 MediaPipe face landmarks，不运行任何动作状态机。

        供 FastLivenessDetector 独占使用：mp_detector 只负责图像推理，
        fast_detector 负责全部动作判断，彻底消除双状态机竞争。

        Args:
            frame: BGR 图像

        Returns:
            包含 landmarks / transform_matrix / quality_score / frame_shape 的字典，
            若未检测到人脸则返回 None。
        """
        self.frame_count += 1
        self._update_fps()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._start_time_ms:
            timestamp_ms = self._start_time_ms + 1
        self._start_time_ms = timestamp_ms

        detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.face_landmarks:
            return None

        landmarks = detection_result.face_landmarks[0]
        transform_matrix = None
        if (
            detection_result.facial_transformation_matrixes
            and len(detection_result.facial_transformation_matrixes) > 0
        ):
            transform_matrix = detection_result.facial_transformation_matrixes[0]

        quality_score = self._calculate_quality_score(frame, landmarks)

        return {
            "landmarks": landmarks,
            "transform_matrix": transform_matrix,
            "quality_score": quality_score,
            "frame_shape": frame.shape,
            "fps": self.fps,
        }

    def detect_liveness(self, frame: np.ndarray, skip: bool = False) -> Dict[str, Any]:
        """
        单帧活体检测。

        Args:
            frame: BGR 图像
            skip:  True = 跳帧，复用上次结果

        Returns:
            结果字典
        """
        self.frame_count += 1
        self._update_fps()

        _empty: Dict[str, Any] = {
            "score": 0.0,
            "ear": 0.0,
            "mar": 0.0,
            "head_pose": (0.0, 0.0, 0.0),
            "face_detected": False,
            "detection_confidence": 0.0,
            "quality_score": 0.0,
            "blink_detected": False,
            "mouth_open": False,
            "mouth_moved": False,
            "head_moved": False,
            "current_action": "none",
            "is_blinking": False,
            "is_mouth_open": False,
            "yaw": 0.0,
            "pitch": 0.0,
            "landmarks": None,
            "embedding": None,
            "face_bbox": None,
            "fps": self.fps,
        }

        if skip:
            # 跳帧：复用上次完整推理结果，但仍向历史窗口补充上次的 EAR/MAR/pose 值，
            # 避免窗口因跳帧而数据稀疏导致峰峰值失真
            if self._last_result is not None:
                self.ear_history.append(self._last_result.get("ear", 0.0))
                self.mar_history.append(self._last_result.get("mar", 0.0))
                last_pose = self._last_result.get("head_pose", (0.0, 0.0, 0.0))
                self.head_pose_history.append((last_pose[0], last_pose[1]))
            return self._last_result if self._last_result is not None else _empty

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._start_time_ms:
            timestamp_ms = self._start_time_ms + 1
        self._start_time_ms = timestamp_ms

        detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.face_landmarks:
            return _empty

        landmarks = detection_result.face_landmarks[0]

        transform_matrix = None
        if (
            detection_result.facial_transformation_matrixes
            and len(detection_result.facial_transformation_matrixes) > 0
        ):
            transform_matrix = detection_result.facial_transformation_matrixes[0]

        # 质量评分：基于人脸尺寸和关键点稳定性
        quality_score = self._calculate_quality_score(frame, landmarks)

        # 提取 InsightFace embedding（用于基准帧校准）
        embedding = None
        face_bbox = None
        if self._face_analyzer is not None:
            try:
                faces = self._face_analyzer.get(frame)
                if faces:
                    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                    embedding = face.embedding
                    face_bbox = tuple(face.bbox.astype(int))
            except Exception:
                pass  # InsightFace 失败不影响主流程

        # 检测置信度（基于质量评分）
        detection_confidence = quality_score

        # 宽高比修正：w/h，MediaPipe 归一化坐标的 y 轴被压缩了这个比例
        h_img, w_img = frame.shape[:2]
        aspect_ratio = w_img / h_img if h_img > 0 else 1.0

        mar = self.calculate_mar(landmarks, aspect_ratio)
        ear = self.calculate_ear(landmarks, aspect_ratio)
        head_pose = self.calculate_head_pose(landmarks, frame.shape, transform_matrix)
        pitch, yaw, roll = head_pose

        # 异常值过滤：限制单帧最大变化，防止关键点误检导致的跳动
        if self._last_raw_pitch is not None:
            pitch_delta = pitch - self._last_raw_pitch
            yaw_delta = yaw - self._last_raw_yaw

            # 如果变化超过阈值，则限制到最大允许变化
            if abs(pitch_delta) > self._max_frame_delta:
                pitch = (
                    self._last_raw_pitch + np.sign(pitch_delta) * self._max_frame_delta
                )
            if abs(yaw_delta) > self._max_frame_delta:
                yaw = self._last_raw_yaw + np.sign(yaw_delta) * self._max_frame_delta

        self._last_raw_pitch = pitch
        self._last_raw_yaw = yaw

        # EMA 平滑：滤除因 scale 残差和关键点抖动造成的逐帧噪声
        a = self._pose_alpha
        if self._smoothed_pitch is None:
            self._smoothed_pitch = pitch
            self._smoothed_yaw = yaw
        else:
            self._smoothed_pitch = a * pitch + (1.0 - a) * self._smoothed_pitch
            self._smoothed_yaw = a * yaw + (1.0 - a) * self._smoothed_yaw
        pitch = self._smoothed_pitch
        yaw = self._smoothed_yaw

        self.current_yaw = yaw
        self.current_pitch = pitch

        mouth_moved = self.detect_mouth_movement(mar)
        blink_detected = self.detect_blink(ear)
        head_action_raw = self.detect_head_action(pitch, yaw)
        current_action = self.update_current_action(head_action_raw)

        head_moved = current_action in (
            "head_turn_left",
            "head_turn_right",
            "head_nod_down",
            "head_nod_up",
        )

        face_center = np.array([landmarks[1].x, landmarks[1].y])
        self.face_position_history.append(face_center)

        motion_score = self._calculate_motion_score()

        result: Dict[str, Any] = {
            "score": motion_score,
            "ear": ear,
            "mar": mar,
            "head_pose": head_pose,
            "face_detected": True,
            "detection_confidence": detection_confidence,
            "quality_score": quality_score,
            "blink_detected": blink_detected,
            "mouth_open": self.is_mouth_open,
            "mouth_moved": mouth_moved,
            "head_moved": head_moved,
            "current_action": current_action,
            "is_blinking": self.is_blinking,
            "is_mouth_open": self.is_mouth_open,
            "yaw": yaw,
            "pitch": pitch,
            "landmarks": landmarks,
            "embedding": embedding,
            "face_bbox": face_bbox,
            "fps": self.fps,
        }
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # 重置 / 关闭
    # ------------------------------------------------------------------

    def reset(self):
        """重置所有检测状态"""
        self.ear_history.clear()
        self.mar_history.clear()
        self.head_pose_history.clear()
        self.face_position_history.clear()

        self.last_ear = None
        self.last_mar = None
        self.eye_closed_frames = 0
        self._mouth_open_frames = 0
        self.is_blinking = False
        self.is_mouth_open = False

        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self._smoothed_pitch = None
        self._smoothed_yaw = None
        self._last_raw_pitch = None
        self._last_raw_yaw = None
        self.current_action = "none"
        self.confirmed_action = "none"
        self.action_frame_count = 0
        self._action_hold_frames = 0
        self._head_action_cooldown = 0

        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self._last_result = None

        if (
            hasattr(self, "_head_action_detector")
            and self._head_action_detector is not None
        ):
            self._head_action_detector.reset()

    def close(self):
        """释放 MediaPipe 资源"""
        self.face_landmarker.close()
