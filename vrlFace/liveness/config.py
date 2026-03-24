"""
活体检测配置模块

纯 MediaPipe 方案配置
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LivenessConfig:
    """活体检测配置类"""

    threshold: float = 0.5

    # 眨眼阈值：降低从 0.18 到 0.15，提高眨眼检出率
    ear_threshold: float = 0.15
    # 张嘴阈值：0.50→0.35→0.28，实测失败视频 avg_score=0.254-0.330
    mar_threshold: float = 0.28
    # 转头阈值：保持 8.0°，配合 nod_yaw_gate_ratio=0.75 使用
    yaw_threshold: float = 8.0
    # 点头阈值：保持 8.0°（峰峰值检测）
    pitch_threshold: float = 8.0
    min_blinks: int = 1

    window_size: int = 30
    smooth_window: int = 10

    max_faces: int = 1
    skip_frames: int = 0
    max_width: int = 640
    target_fps: float = 25.0

    challenge_mode: bool = False
    challenge_timeout: int = 10
    required_actions: Tuple[str, ...] = ("blink", "mouth_open", "head_turn")

    action_confirm_frames: int = 2

    # 基准帧校准配置
    enable_benchmark: bool = True  # 是否启用基准帧校准
    benchmark_duration: float = 2.0  # 基准采集时长（秒）
    benchmark_min_frames: int = 3  # 最少基准帧数
    benchmark_max_frames: int = 10  # 最多基准帧数
    benchmark_min_quality: float = 0.6  # 基准帧最低质量
    benchmark_max_angle: float = 15.0  # 基准帧最大角度
    embedding_threshold: float = 0.55  # embedding 相似度阈值
    enable_threshold_calibration: bool = False  # 是否启用阈值校准

    # 静默检测配置
    enable_silent_detection: bool = False
    silent_detection_mode: str = "strict"  # "strict" 或 "loose"
    silent_sample_frames: int = 5
    silent_min_quality: float = 0.6
    silent_max_angle: float = 15.0
    silent_detection_timeout: float = 5.0

    def validate(self) -> bool:
        errors = []
        if not (0.0 <= self.threshold <= 1.0):
            errors.append(f"阈值必须在 [0, 1] 范围内：{self.threshold}")
        if self.window_size <= 0:
            errors.append(f"时间窗口必须为正数：{self.window_size}")
        if self.skip_frames < 0:
            errors.append(f"跳帧数必须非负：{self.skip_frames}")
        if errors:
            for error in errors:
                print(f"配置错误：{error}")
            return False
        return True

    def display(self):
        print("=" * 60)
        print("活体检测配置")
        print("=" * 60)
        print(f"活体阈值：{self.threshold}")
        print("-" * 60)
        print(f"EAR 阈值：{self.ear_threshold}")
        print(f"MAR 阈值：{self.mar_threshold}")
        print(f"Yaw 阈值：{self.yaw_threshold}")
        print(f"Pitch 阈值：{self.pitch_threshold}")
        print("-" * 60)
        print(f"时间窗口：{self.window_size} 帧")
        print(f"平滑窗口：{self.smooth_window} 帧")
        print("-" * 60)
        print(f"跳帧设置：每 {self.skip_frames + 1} 帧检测 1 次")
        print(f"最大人脸：{self.max_faces}")
        print(f"最大宽度：{self.max_width}")
        print("-" * 60)
        print(f"挑战模式：{self.challenge_mode}")
        if self.challenge_mode:
            print(f"挑战超时：{self.challenge_timeout} 秒")
            print(f"要求动作：{self.required_actions}")
        print("=" * 60)

    @classmethod
    def cpu_fast_config(cls) -> "LivenessConfig":
        return cls(
            skip_frames=0,
            max_width=640,
            target_fps=25.0,
            window_size=15,
            smooth_window=5,
            threshold=0.45,
            yaw_threshold=8.0,
            pitch_threshold=8.0,
            ear_threshold=0.18,
            mar_threshold=0.50,
            action_confirm_frames=3,
        )

    @classmethod
    def cpu_accurate_config(cls) -> "LivenessConfig":
        return cls(
            skip_frames=1,
            window_size=60,
            threshold=0.55,
        )

    @classmethod
    def video_anti_spoofing_config(cls) -> "LivenessConfig":
        return cls(
            window_size=30,
            min_blinks=2,
            threshold=0.50,
            skip_frames=0,
            max_width=640,
            target_fps=0.0,
            yaw_threshold=8.0,
            pitch_threshold=8.0,
            ear_threshold=0.22,
            mar_threshold=0.50,
            action_confirm_frames=3,
        )

    @classmethod
    def realtime_config(cls) -> "LivenessConfig":
        return cls(
            skip_frames=0,
            max_width=480,
            target_fps=0.0,
            window_size=20,
            smooth_window=5,
            threshold=0.35,
            ear_threshold=0.18,
            mar_threshold=0.50,
            yaw_threshold=8.0,
            pitch_threshold=8.0,
            action_confirm_frames=1,
        )

    @classmethod
    def video_fast_config(cls) -> "LivenessConfig":
        return cls(
            skip_frames=0,
            max_width=640,
            target_fps=0.0,
            window_size=20,
            threshold=0.35,
            yaw_threshold=8.0,
            pitch_threshold=8.0,
            # ear_threshold: 0.18→0.15 提高眨眼检出率
            ear_threshold=0.15,
            # mar_threshold: 0.50→0.28 提高张嘴检出率（实测失败视频 avg_score=0.254-0.330）
            mar_threshold=0.28,
            action_confirm_frames=1,
        )

    @classmethod
    def fast_detector_config(cls) -> dict:
        return {
            "ear_threshold": 0.15,
            "eye_open_threshold": 0.18,
            "eye_close_threshold": 0.18,
            # mar_threshold: 0.50→0.28 提高张嘴检出率
            "mar_threshold": 0.28,
            "yaw_threshold": 8.0,
            "pitch_threshold": 8.0,
            "window_size": 15,
            "action_confirm_frames": 3,
        }

    @classmethod
    def video_anti_spoofing_with_silent_config(cls) -> "LivenessConfig":
        """视频防伪模式（含静默检测）"""
        config = cls.video_anti_spoofing_config()
        config.enable_silent_detection = True
        config.silent_detection_mode = "strict"
        config.silent_sample_frames = 5
        config.silent_min_quality = 0.6
        config.silent_max_angle = 15.0
        config.silent_detection_timeout = 5.0
        return config

    @classmethod
    def from_env(cls) -> "LivenessConfig":
        """从环境变量加载配置"""
        import os

        config = cls.realtime_config()

        if os.getenv("LIVENESS_ENABLE_SILENT", "").lower() == "true":
            config.enable_silent_detection = True
        config.silent_detection_mode = os.getenv("LIVENESS_SILENT_MODE", "strict")
        config.silent_sample_frames = int(
            os.getenv("LIVENESS_SILENT_SAMPLE_FRAMES", "5")
        )
        config.silent_min_quality = float(
            os.getenv("LIVENESS_SILENT_MIN_QUALITY", "0.6")
        )
        config.silent_max_angle = float(os.getenv("LIVENESS_SILENT_MAX_ANGLE", "15.0"))
        config.silent_detection_timeout = float(
            os.getenv("LIVENESS_SILENT_TIMEOUT", "5.0")
        )

        return config


@dataclass
class CallbackConfig:
    """回调配置"""

    default_url: str = "http://172.17.0.1:8092/api/v1/callbacks/liveness/action"
    secret_key: str = "kyc-service-secret-key-2024"
    timeout: int = 10  # 回调超时（秒）
    max_retries: int = 3  # 最大重试次数
    retry_delay: int = 2  # 重试间隔（秒）

    @classmethod
    def from_env(cls) -> "CallbackConfig":
        """从环境变量加载配置"""
        import os

        # 尝试自动检测 Docker 网关 IP
        default_url = cls.default_url
        if not os.getenv("LIVENESS_CALLBACK_URL"):
            try:
                # 读取默认网关（Docker 宿主机 IP）
                import subprocess

                result = subprocess.run(
                    ["ip", "route", "show", "default"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    # 输出格式: default via 172.17.0.1 dev eth0
                    parts = result.stdout.split()
                    if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
                        gateway_ip = parts[2]
                        default_url = (
                            f"http://{gateway_ip}:8092/api/v1/callbacks/liveness/action"
                        )
            except Exception:
                pass  # 失败则使用默认值

        return cls(
            default_url=os.getenv("LIVENESS_CALLBACK_URL", default_url),
            secret_key=os.getenv("LIVENESS_CALLBACK_SECRET", cls.secret_key),
            timeout=int(os.getenv("LIVENESS_CALLBACK_TIMEOUT", str(cls.timeout))),
            max_retries=int(
                os.getenv("LIVENESS_CALLBACK_MAX_RETRIES", str(cls.max_retries))
            ),
            retry_delay=int(
                os.getenv("LIVENESS_CALLBACK_RETRY_DELAY", str(cls.retry_delay))
            ),
        )

    secret_key: str = "kyc-service-secret-key-2024"
    timeout: int = 10  # 回调超时（秒）
    max_retries: int = 3  # 最大重试次数
    retry_delay: int = 2  # 重试间隔（秒）

    @classmethod
    def from_env(cls) -> "CallbackConfig":
        """从环境变量加载配置"""
        import os

        return cls(
            default_url=os.getenv("LIVENESS_CALLBACK_URL", cls.default_url),
            secret_key=os.getenv("LIVENESS_CALLBACK_SECRET", cls.secret_key),
            timeout=int(os.getenv("LIVENESS_CALLBACK_TIMEOUT", str(cls.timeout))),
            max_retries=int(
                os.getenv("LIVENESS_CALLBACK_MAX_RETRIES", str(cls.max_retries))
            ),
            retry_delay=int(
                os.getenv("LIVENESS_CALLBACK_RETRY_DELAY", str(cls.retry_delay))
            ),
        )


config = LivenessConfig.cpu_fast_config()


if __name__ == "__main__":
    print("默认配置:")
    config.display()
    print("\nCPU 快速配置:")
    LivenessConfig.cpu_fast_config().display()
    print("\n视频防伪配置:")
    LivenessConfig.video_anti_spoofing_config().display()
