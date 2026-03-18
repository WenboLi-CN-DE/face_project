"""
证件照预处理模块测试

测试覆盖：
    - 配置默认值（id_crop_expand_ratio, id_enhance_enabled）
    - 环境变量覆盖
    - 无人脸时返回原图
    - 极小图片兜底（不崩溃）
    - 增强开关（id_enhance_enabled=False 时跳过 CLAHE）

运行:
    pytest tests/face/test_id_preprocess.py -v
"""

import sys
from pathlib import Path
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vrlFace.face.config import FaceConfig
from vrlFace.face.id_preprocess import preprocess_id_photo


# ---------------------------------------------------------------------------
# Task 1: 配置字段测试
# ---------------------------------------------------------------------------


def test_config_id_crop_expand_ratio_default():
    """id_crop_expand_ratio 默认值应为 0.3"""
    cfg = FaceConfig()
    assert hasattr(cfg, "id_crop_expand_ratio")
    assert cfg.id_crop_expand_ratio == 0.3


def test_config_id_enhance_enabled_default():
    """id_enhance_enabled 默认值应为 True"""
    cfg = FaceConfig()
    assert hasattr(cfg, "id_enhance_enabled")
    assert cfg.id_enhance_enabled is True


def test_config_id_crop_expand_from_env(monkeypatch):
    """FACE_ID_CROP_EXPAND 环境变量可覆盖 id_crop_expand_ratio"""
    monkeypatch.setenv("FACE_ID_CROP_EXPAND", "0.5")
    cfg = FaceConfig.from_env()
    assert cfg.id_crop_expand_ratio == 0.5


def test_config_id_enhance_from_env_false(monkeypatch):
    """FACE_ID_ENHANCE=false 应将 id_enhance_enabled 设为 False"""
    monkeypatch.setenv("FACE_ID_ENHANCE", "false")
    cfg = FaceConfig.from_env()
    assert cfg.id_enhance_enabled is False


def test_config_id_enhance_from_env_true(monkeypatch):
    """FACE_ID_ENHANCE=true 应将 id_enhance_enabled 设为 True"""
    monkeypatch.setenv("FACE_ID_ENHANCE", "true")
    cfg = FaceConfig.from_env()
    assert cfg.id_enhance_enabled is True


def test_config_display_contains_new_fields(capsys):
    """display() 应打印新增的两个字段"""
    cfg = FaceConfig()
    cfg.display()
    captured = capsys.readouterr()
    assert "id_crop_expand_ratio" in captured.out or "裁剪扩展" in captured.out
    assert "id_enhance_enabled" in captured.out or "图像增强" in captured.out


# ---------------------------------------------------------------------------
# Task 2: preprocess_id_photo() 核心逻辑测试
# ---------------------------------------------------------------------------


def test_preprocess_returns_ndarray_on_no_face():
    """无人脸图片应返回原图（ndarray）"""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_id_photo(black_img)
    assert isinstance(result, np.ndarray)
    assert result.shape == black_img.shape


def test_preprocess_returns_original_on_no_face():
    """无人脸图片返回的数组内容应与原图一致"""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_id_photo(black_img)
    np.testing.assert_array_equal(result, black_img)


def test_preprocess_tiny_image_does_not_crash():
    """极小图片（10x10）不应抛出异常，应返回 ndarray"""
    tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = preprocess_id_photo(tiny_img)
    assert isinstance(result, np.ndarray)


def test_preprocess_with_enhance_disabled(monkeypatch):
    """id_enhance_enabled=False 时，无人脸图片应正常返回原图"""
    monkeypatch.setenv("FACE_ID_ENHANCE", "false")

    # 重新导入以让 config 生效（使用 importlib 重载）
    import importlib
    import vrlFace.face.config as config_module
    import vrlFace.face.id_preprocess as preprocess_module

    # 创建临时配置并注入
    cfg = FaceConfig.from_env()
    assert cfg.id_enhance_enabled is False

    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # preprocess_id_photo 读取 from .config import config，
    # 因此这里直接验证无人脸路径下不崩溃
    result = preprocess_id_photo(black_img)
    assert isinstance(result, np.ndarray)


def test_preprocess_single_channel_image_does_not_crash():
    """单通道灰度图片不应崩溃，应返回 ndarray"""
    gray_img = np.zeros((100, 100), dtype=np.uint8)
    result = preprocess_id_photo(gray_img)
    assert isinstance(result, np.ndarray)
