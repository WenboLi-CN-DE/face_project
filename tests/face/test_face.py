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

