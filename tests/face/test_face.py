"""
人脸识别模块测试

运行:
    pytest tests/face/test_face.py -v
"""

import sys
from pathlib import Path
import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vrlFace.face import face_detection, gen_verify_res, face_search
from vrlFace.face.config import FaceConfig, config


def test_config_defaults():
    """测试默认配置"""
    cfg = FaceConfig()
    assert cfg.model_name == "buffalo_l"
    assert cfg.similarity_threshold == 0.55
    assert cfg.images_base is not None  # 修复：images_base 必须存在
    assert cfg.validate()


def test_config_from_env():
    """测试环境变量加载"""
    cfg = FaceConfig.from_env()
    assert cfg.validate()
    assert hasattr(cfg, "images_base")


def test_face_detection_no_face():
    """测试空图片人脸检测"""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = face_detection(black_img)
    assert result["is_face_exist"] == 0
    assert result["face_num"] == 0
    assert isinstance(result["faces_detected"], list)


def test_gen_verify_res_no_face():
    """测试无人脸图片的比对结果"""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = gen_verify_res(black_img, black_img)
    assert result["is_face_exist"] == 0
    assert result["is_same_face"] == -1


def test_face_search_missing_db():
    """测试数据库目录不存在时的搜索"""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = face_search(black_img, db_path="/nonexistent/path")
    assert result["has_similar_picture"] == 0
    assert isinstance(result["searched_similar_pictures"], list)


# ---------------------------------------------------------------------------
# 人证比对（/vrlFaceIdComparison）相关测试
# ---------------------------------------------------------------------------

def test_id_comparison_threshold_in_config():
    """测试 FaceConfig 包含人证比对专用阈值"""
    cfg = FaceConfig()
    assert hasattr(cfg, "id_comparison_threshold")
    # 人证比对阈值应高于普通比对阈值，确保安全性
    assert cfg.id_comparison_threshold >= cfg.similarity_threshold
    assert 0.0 <= cfg.id_comparison_threshold <= 1.0


def test_id_comparison_threshold_from_env(monkeypatch):
    """测试 FACE_ID_THRESHOLD 环境变量可覆盖人证比对阈值"""
    monkeypatch.setenv("FACE_ID_THRESHOLD", "0.75")
    cfg = FaceConfig.from_env()
    assert cfg.id_comparison_threshold == 0.75


def test_id_comparison_no_face_returns_correct_structure():
    """
    人证比对底层逻辑测试：无人脸时 is_same_face=-1，
    confidence 百分化后仍为 0.0
    """
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = gen_verify_res(black_img, black_img, threshold=config.id_comparison_threshold)

    # 模拟 API 层的百分化处理
    result["confidence"] = round(result["confidence"] * 100, 2)

    assert result["is_face_exist"] == 0
    assert result["is_same_face"] == -1
    assert result["confidence"] == 0.0
    assert "detection_result" in result


def test_id_comparison_confidence_is_percentage():
    """
    验证百分化逻辑：将余弦相似度 × 100 后值在合理范围内
    """
    # 模拟 gen_verify_res 返回的原始结构（两张无人脸）
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = gen_verify_res(black_img, black_img)
    raw_confidence = result["confidence"]

    # 百分化
    pct_confidence = round(raw_confidence * 100, 2)

    # 百分制置信度应在 -100 ~ 100 之间（负值表示极度不相似，属正常）
    assert -100.0 <= pct_confidence <= 100.0
    # 无人脸时原始 confidence 为 0.0
    assert pct_confidence == 0.0



