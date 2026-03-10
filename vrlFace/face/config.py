"""
人脸识别配置模块

集中管理人脸识别系统的所有配置参数
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List


@dataclass
class FaceConfig:
    """人脸识别系统配置类"""

    # ==================== 模型配置 ====================
    model_name: str = "buffalo_l"
    model_path: str | None = None  # None 表示使用默认路径 ~/.insightface/models/

    # ==================== 检测配置 ====================
    det_size: Tuple[int, int] = (640, 640)  # 检测尺寸，越大越精确但越慢
    ctx_id: int = -1  # GPU ID，-1 表示使用 CPU，0+ 表示 GPU 编号

    # ==================== 相似度阈值 ====================
    similarity_threshold: float = 0.55       # 默认匹配阈值
    high_confidence_threshold: float = 0.70  # 高置信度阈值
    low_confidence_threshold: float = 0.40   # 低置信度阈值（宽松模式）
    id_comparison_threshold: float = 0.60    # 人证比对专用阈值（安全性要求更高）

    # ==================== 质量控制 ====================
    min_face_size: int = 80  # 最小人脸尺寸（像素）
    max_face_angle: float = 30.0  # 最大人脸偏转角度（度）
    min_quality_score: float = 0.5  # 最小质量分数

    # ==================== 性能配置 ====================
    batch_size: int = 32  # 批量处理大小
    use_gpu: bool = False  # 是否使用 GPU
    num_workers: int = 4  # 并行处理线程数

    # ==================== 数据路径 ====================
    images_base: str = str(Path("/app/data/dataset"))  # 图片库基础路径（用于人脸搜索）

    # ==================== 执行提供者 ====================
    @property
    def providers(self) -> List[str]:
        """根据配置返回 ONNX Runtime 执行提供者"""
        if self.use_gpu and self.ctx_id >= 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ==================== 从环境变量加载 ====================
    @classmethod
    def from_env(cls) -> "FaceConfig":
        """从环境变量加载配置，环境变量优先级高于默认值"""
        return cls(
            model_name=os.getenv("FACE_MODEL_NAME", "buffalo_l"),
            model_path=os.getenv("FACE_MODEL_PATH", None),
            det_size=cls._parse_det_size(os.getenv("FACE_DET_SIZE", "640,640")),
            ctx_id=int(os.getenv("FACE_GPU_ID", "-1")),
            similarity_threshold=float(os.getenv("FACE_THRESHOLD", "0.55")),
            high_confidence_threshold=float(os.getenv("FACE_HIGH_THRESHOLD", "0.70")),
            low_confidence_threshold=float(os.getenv("FACE_LOW_THRESHOLD", "0.40")),
            id_comparison_threshold=float(os.getenv("FACE_ID_THRESHOLD", "0.60")),
            min_face_size=int(os.getenv("FACE_MIN_SIZE", "80")),
            max_face_angle=float(os.getenv("FACE_MAX_ANGLE", "30.0")),
            min_quality_score=float(os.getenv("FACE_MIN_QUALITY", "0.5")),
            batch_size=int(os.getenv("FACE_BATCH_SIZE", "32")),
            use_gpu=os.getenv("FACE_USE_GPU", "false").lower() in ("true", "1", "yes"),
            num_workers=int(os.getenv("FACE_NUM_WORKERS", "4")),
            images_base=os.getenv(
                "FACE_IMAGES_BASE",
                str(Path("/app/data/dataset")),
            ),
        )

    @staticmethod
    def _parse_det_size(det_size_str: str) -> Tuple[int, int]:
        """解析检测尺寸字符串"""
        try:
            w, h = det_size_str.split(",")
            return (int(w.strip()), int(h.strip()))
        except Exception:
            return (640, 640)

    # ==================== 验证配置 ====================
    def validate(self) -> bool:
        """验证配置是否合法"""
        errors = []

        if self.det_size[0] <= 0 or self.det_size[1] <= 0:
            errors.append(f"检测尺寸必须为正数: {self.det_size}")

        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append(
                f"相似度阈值必须在 [0, 1] 范围内: {self.similarity_threshold}"
            )

        if self.min_face_size <= 0:
            errors.append(f"最小人脸尺寸必须为正数: {self.min_face_size}")

        if self.batch_size <= 0:
            errors.append(f"批量大小必须为正数: {self.batch_size}")

        if errors:
            for error in errors:
                print(f"配置错误: {error}")
            return False

        return True

    # ==================== 显示配置 ====================
    def display(self):
        """打印当前配置"""
        print("=" * 60)
        print("人脸识别系统配置")
        print("=" * 60)
        print(f"模型名称: {self.model_name}")
        print(f"模型路径: {self.model_path or '默认路径'}")
        print(f"检测尺寸: {self.det_size}")
        print(f"设备: {'GPU ' + str(self.ctx_id) if self.use_gpu else 'CPU'}")
        print(f"执行提供者: {', '.join(self.providers)}")
        print("-" * 60)
        print(f"相似度阈值: {self.similarity_threshold}")
        print(f"高置信度阈值: {self.high_confidence_threshold}")
        print(f"低置信度阈值: {self.low_confidence_threshold}")
        print(f"人证比对阈值: {self.id_comparison_threshold}")
        print("-" * 60)
        print(f"最小人脸尺寸: {self.min_face_size} 像素")
        print(f"最大人脸角度: {self.max_face_angle}°")
        print(f"最小质量分数: {self.min_quality_score}")
        print("-" * 60)
        print(f"批量大小: {self.batch_size}")
        print(f"并行线程数: {self.num_workers}")
        print(f"图片库路径: {self.images_base}")
        print("=" * 60)


# ==================== 预定义配置 ====================

# 默认配置（平衡模式）
DEFAULT_CONFIG = FaceConfig()

# CPU 优化配置（速度优先）
CPU_FAST_CONFIG = FaceConfig(
    det_size=(320, 320), similarity_threshold=0.50, batch_size=16
)

# GPU 高精度配置
GPU_HIGH_ACCURACY_CONFIG = FaceConfig(
    det_size=(1024, 1024),
    ctx_id=0,
    use_gpu=True,
    similarity_threshold=0.60,
    batch_size=64,
)

# 严格模式配置（低误识率）
STRICT_CONFIG = FaceConfig(
    similarity_threshold=0.70,
    high_confidence_threshold=0.85,
    min_face_size=100,
    min_quality_score=0.7,
)

# 宽松模式配置（高召回率）
LOOSE_CONFIG = FaceConfig(
    similarity_threshold=0.40,
    low_confidence_threshold=0.30,
    min_face_size=60,
    min_quality_score=0.3,
)


# ==================== 全局配置实例 ====================
# 默认使用环境变量配置，如果没有环境变量则使用默认配置
config = FaceConfig.from_env()


if __name__ == "__main__":
    print("\n1. 默认配置:")
    DEFAULT_CONFIG.display()

    print("\n2. CPU 快速配置:")
    CPU_FAST_CONFIG.display()

    print("\n3. 从环境变量加载的配置:")
    config.display()

    print("\n4. 配置验证:")
    is_valid = config.validate()
    print(f"配置{'有效' if is_valid else '无效'}")

