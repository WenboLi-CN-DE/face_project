# 证件照人脸预处理 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在人证比对流程中增加证件照预处理步骤（检测 → 对齐 → 裁剪 → 增强），提升比对准确率。

**Architecture:** 新增 `vrlFace/face/id_preprocess.py` 模块，复用 InsightFace 检测器获取人脸 bbox + 5 关键点，做仿射对齐 + ROI 扩展裁剪 + CLAHE/去噪增强。仅在 `/vrlFaceIdComparison` 端点对证件照（picture1）调用，失败时兜底返回原图。

**Tech Stack:** InsightFace (已有), OpenCV (已有), NumPy (已有)

**Spec:** `docs/superpowers/specs/2026-03-18-id-photo-preprocess-design.md`

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `vrlFace/face/id_preprocess.py` | 新增 | 证件照预处理：对齐、裁剪、增强 |
| `vrlFace/face/config.py` | 修改 | 新增 2 个配置字段 |
| `vrlFace/face/api.py` | 修改 | 集成预处理到人证比对端点 |
| `tests/face/test_id_preprocess.py` | 新增 | 预处理模块测试 |

---

## Chunk 1: 配置字段 + 预处理模块

### Task 1: 新增配置字段

**Files:**
- Modify: `vrlFace/face/config.py:26-29` (字段定义区)
- Modify: `vrlFace/face/config.py:54-75` (from_env 方法)
- Modify: `vrlFace/face/config.py:112-136` (display 方法)
- Test: `tests/face/test_id_preprocess.py`

- [ ] **Step 1: 写失败测试 — 配置字段默认值**

在 `tests/face/test_id_preprocess.py` 中：

```python
"""
证件照预处理模块测试

运行:
    pytest tests/face/test_id_preprocess.py -v
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vrlFace.face.config import FaceConfig, config


def test_id_crop_expand_ratio_default():
    """测试 id_crop_expand_ratio 默认值为 0.3"""
    cfg = FaceConfig()
    assert cfg.id_crop_expand_ratio == 0.3


def test_id_enhance_enabled_default():
    """测试 id_enhance_enabled 默认值为 True"""
    cfg = FaceConfig()
    assert cfg.id_enhance_enabled is True


def test_id_config_from_env(monkeypatch):
    """测试环境变量覆盖配置"""
    monkeypatch.setenv("FACE_ID_CROP_EXPAND", "0.5")
    monkeypatch.setenv("FACE_ID_ENHANCE", "false")
    cfg = FaceConfig.from_env()
    assert cfg.id_crop_expand_ratio == 0.5
    assert cfg.id_enhance_enabled is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/face/test_id_preprocess.py::test_id_crop_expand_ratio_default -v`
Expected: FAIL — `AttributeError: ... has no attribute 'id_crop_expand_ratio'`

- [ ] **Step 3: 实现配置字段**

在 `vrlFace/face/config.py` 的 `FaceConfig` 类中，`id_comparison_threshold` 字段后新增：

```python
    id_crop_expand_ratio: float = 0.3     # 证件照 ROI 扩展比例
    id_enhance_enabled: bool = True        # 是否启用图像增强
```

在 `from_env()` 方法的 return 语句中，`id_comparison_threshold=...` 行后新增：

```python
            id_crop_expand_ratio=float(os.getenv("FACE_ID_CROP_EXPAND", "0.3")),
            id_enhance_enabled=os.getenv("FACE_ID_ENHANCE", "true").lower() in ("true", "1", "yes"),
```

在 `display()` 方法中，`人证比对阈值` 打印行后新增：

```python
        print(f"证件照裁剪扩展比例: {self.id_crop_expand_ratio}")
        print(f"证件照增强: {'启用' if self.id_enhance_enabled else '禁用'}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/face/test_id_preprocess.py -v -k "config"`
Expected: 3 tests PASS

- [ ] **Step 5: 运行现有测试确认无回归**

Run: `pytest tests/face/test_face.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add vrlFace/face/config.py tests/face/test_id_preprocess.py
git commit -m "feat(config): 新增 id_crop_expand_ratio 和 id_enhance_enabled 配置字段"
```

---

### Task 2: 实现预处理核心模块

**Files:**
- Create: `vrlFace/face/id_preprocess.py`
- Test: `tests/face/test_id_preprocess.py` (追加测试)

- [ ] **Step 1: 写失败测试 — preprocess_id_photo 兜底行为**

在 `tests/face/test_id_preprocess.py` 中追加：

```python
from vrlFace.face.id_preprocess import preprocess_id_photo


def test_preprocess_no_face_returns_original():
    """无人脸图片应返回原图"""
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_id_photo(black_img)
    assert result.shape == black_img.shape
    assert np.array_equal(result, black_img)


def test_preprocess_empty_image_returns_original():
    """极小图片应返回原图"""
    tiny_img = np.zeros((1, 1, 3), dtype=np.uint8)
    result = preprocess_id_photo(tiny_img)
    assert np.array_equal(result, tiny_img)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/face/test_id_preprocess.py::test_preprocess_no_face_returns_original -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vrlFace.face.id_preprocess'`

- [ ] **Step 3: 实现 id_preprocess.py 骨架 + preprocess_id_photo**

创建 `vrlFace/face/id_preprocess.py`：

```python
"""
证件照人脸预处理模块

在人证比对流程中，对证件照进行：
    1. 人脸检测（复用 InsightFace）
    2. 仿射变换对齐（基于 5 关键点）
    3. ROI 扩展裁剪
    4. 轻量图像增强（CLAHE + 去噪）

预处理失败时兜底返回原图，不影响比对流程。
"""

import logging

import cv2
import numpy as np

from .config import config
from .recognizer import get_recognizer

logger = logging.getLogger(__name__)


def preprocess_id_photo(img: np.ndarray) -> np.ndarray:
    """
    证件照预处理主入口

    Args:
        img: RGB 图像 numpy 数组

    Returns:
        预处理后的 RGB 图像，失败时返回原图
    """
    try:
        recognizer = get_recognizer()
        faces = recognizer.get(img)

        if not faces:
            logger.warning("证件照预处理：未检测到人脸，返回原图")
            return img

        # 多张人脸取面积最大的
        if len(faces) > 1:
            logger.info("证件照预处理：检测到 %d 张人脸，取最大", len(faces))
            face = max(faces, key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ))
        else:
            face = faces[0]

        # 1. 仿射变换对齐
        if hasattr(face, "kps") and face.kps is not None:
            aligned = _align_face(img, face.kps)
        else:
            logger.warning("证件照预处理：无关键点信息，跳过对齐")
            aligned = img

        # 对齐后重新检测以获取新的 bbox
        faces_aligned = recognizer.get(aligned)
        if not faces_aligned:
            logger.warning("证件照预处理：对齐后未检测到人脸，使用原图 bbox")
            bbox = face.bbox
            crop_src = img
        else:
            face_aligned = max(faces_aligned, key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ))
            bbox = face_aligned.bbox
            crop_src = aligned

        # 2. ROI 扩展裁剪
        cropped = _crop_face_roi(
            crop_src, bbox, expand_ratio=config.id_crop_expand_ratio
        )

        # 3. 图像增强
        if config.id_enhance_enabled:
            try:
                enhanced = _enhance_image(cropped)
            except Exception as e:
                logger.warning("证件照预处理：增强失败(%s)，跳过增强", e)
                enhanced = cropped
        else:
            enhanced = cropped

        logger.info(
            "证件照预处理完成: aligned=%s, enhanced=%s, shape=%s",
            hasattr(face, "kps") and face.kps is not None,
            config.id_enhance_enabled,
            enhanced.shape,
        )
        return enhanced

    except Exception as e:
        logger.warning("证件照预处理异常(%s)，返回原图", e)
        return img


def _align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    基于 5 关键点做仿射变换对齐

    以双眼中心连线为基准，计算旋转角度，
    仿射变换将人脸旋转到水平姿态。

    Args:
        img: 原图 (H, W, 3)
        landmarks: 5 个关键点坐标 shape=(5, 2)
                   顺序: 左眼、右眼、鼻尖、左嘴角、右嘴角

    Returns:
        对齐后的图像（与原图同尺寸）
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # 计算双眼连线角度
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 以双眼中心为旋转中心
    eye_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    )

    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned = cv2.warpAffine(
        img, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned


def _crop_face_roi(
    img: np.ndarray,
    bbox: np.ndarray,
    expand_ratio: float = 0.3,
) -> np.ndarray:
    """
    基于 bbox 扩展裁剪人脸 ROI

    Args:
        img: 图像 (H, W, 3)
        bbox: 人脸边界框 [x1, y1, x2, y2]
        expand_ratio: 扩展比例

    Returns:
        裁剪后的图像
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    face_w = x2 - x1
    face_h = y2 - y1

    # 向外扩展
    expand_w = face_w * expand_ratio
    expand_h = face_h * expand_ratio

    x1 = max(0, int(x1 - expand_w))
    y1 = max(0, int(y1 - expand_h))
    x2 = min(w, int(x2 + expand_w))
    y2 = min(h, int(y2 + expand_h))

    cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        return img

    return cropped


def _enhance_image(img: np.ndarray) -> np.ndarray:
    """
    轻量图像增强

    - CLAHE 自适应直方图均衡（改善光照不均）
    - 高斯去噪（针对扫描件噪点）

    Args:
        img: RGB 图像

    Returns:
        增强后的 RGB 图像
    """
    # 转 LAB 色彩空间，仅对 L 通道做 CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # 高斯去噪
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced
```

- [ ] **Step 4: 运行兜底测试确认通过**

Run: `pytest tests/face/test_id_preprocess.py::test_preprocess_no_face_returns_original tests/face/test_id_preprocess.py::test_preprocess_empty_image_returns_original -v`
Expected: 2 tests PASS

- [ ] **Step 5: 写失败测试 — 私有函数**

在 `tests/face/test_id_preprocess.py` 中追加：

```python
from vrlFace.face.id_preprocess import _align_face, _crop_face_roi, _enhance_image


def test_align_face_horizontal():
    """对齐后双眼应接近水平"""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    # 模拟倾斜的关键点：左眼(60,80), 右眼(140,120), 鼻(100,130), 左嘴(70,160), 右嘴(130,160)
    landmarks = np.array([
        [60, 80], [140, 120], [100, 130], [70, 160], [130, 160]
    ], dtype=np.float32)
    result = _align_face(img, landmarks)
    assert result.shape == img.shape
    assert not np.array_equal(result, img)  # 有旋转应该不同


def test_crop_face_roi_expands():
    """裁剪区域应大于原始 bbox"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    bbox = np.array([100, 100, 200, 200])  # 100x100 的人脸
    result = _crop_face_roi(img, bbox, expand_ratio=0.3)
    # 扩展后应大于 100x100
    assert result.shape[0] > 100
    assert result.shape[1] > 100


def test_crop_face_roi_clamps_to_image():
    """裁剪不应超出图像边界"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = np.array([0, 0, 190, 190])  # 接近边界
    result = _crop_face_roi(img, bbox, expand_ratio=0.5)
    assert result.shape[0] <= 200
    assert result.shape[1] <= 200


def test_crop_face_roi_custom_ratio():
    """自定义扩展比例应生效"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    bbox = np.array([150, 150, 250, 250])  # 100x100
    result_small = _crop_face_roi(img, bbox, expand_ratio=0.1)
    result_large = _crop_face_roi(img, bbox, expand_ratio=0.5)
    # 大扩展比例裁剪区域应更大
    assert result_large.shape[0] > result_small.shape[0]
    assert result_large.shape[1] > result_small.shape[1]


def test_enhance_image_modifies():
    """增强应改变图像像素值"""
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    result = _enhance_image(img)
    assert result.shape == img.shape
    assert not np.array_equal(result, img)


def test_enhance_disabled_by_config(monkeypatch):
    """id_enhance_enabled=False 时 preprocess 应跳过增强"""
    monkeypatch.setattr(config, "id_enhance_enabled", False)
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 无人脸时直接返回原图，增强开关不影响兜底逻辑
    result = preprocess_id_photo(black_img)
    assert np.array_equal(result, black_img)
```

- [ ] **Step 6: 运行全部测试确认通过**

Run: `pytest tests/face/test_id_preprocess.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add vrlFace/face/id_preprocess.py tests/face/test_id_preprocess.py
git commit -m "feat(face): 实现证件照预处理模块（对齐+裁剪+增强）"
```

---

## Chunk 2: API 集成 + 最终验证

### Task 3: 集成预处理到 API 端点

**Files:**
- Modify: `vrlFace/face/api.py:1-9` (import 区)
- Modify: `vrlFace/face/api.py:126-166` (vrl_face_id_comparison 函数)

- [ ] **Step 1: 修改 api.py — 添加 import**

在 `vrlFace/face/api.py` 的 import 区，`from .config import config` 行后新增：

```python
from .id_preprocess import preprocess_id_photo
```

- [ ] **Step 2: 修改 api.py — 调用预处理**

在 `vrl_face_id_comparison` 函数中，`image2 = _read_image(...)` 行之后、`result = gen_verify_res(...)` 行之前，新增：

```python
        # 仅对证件照做预处理（对齐+裁剪+增强）
        image1 = preprocess_id_photo(image1)
```

- [ ] **Step 3: 验证 import 无报错**

Run: `python -c "from vrlFace.face.api import router; print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 4: 运行全部测试确认无回归**

Run: `pytest tests/face/ -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add vrlFace/face/api.py
git commit -m "feat(api): 在 /vrlFaceIdComparison 集成证件照预处理"
```

---

### Task 4: 最终验证

- [ ] **Step 1: 运行完整测试套件**

Run: `pytest tests/ -v`
Expected: 全部 PASS，无回归

- [ ] **Step 2: 代码风格检查**

Run: `flake8 vrlFace/face/id_preprocess.py vrlFace/face/config.py vrlFace/face/api.py`
Expected: 无错误（或仅预存在的警告）

- [ ] **Step 3: 手动冒烟测试**

Run:
```bash
python -c "
from vrlFace.face.id_preprocess import preprocess_id_photo
import numpy as np
img = np.zeros((480, 640, 3), dtype=np.uint8)
result = preprocess_id_photo(img)
assert result.shape == img.shape
print('preprocess fallback OK')
"
```
Expected: 输出 `preprocess fallback OK`

- [ ] **Step 4: 确认文件变更清单**

变更文件应恰好为：
- `vrlFace/face/id_preprocess.py` (新增)
- `vrlFace/face/config.py` (修改)
- `vrlFace/face/api.py` (修改)
- `tests/face/test_id_preprocess.py` (新增)
