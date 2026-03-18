# 证件照人脸预处理设计文档

**日期**: 2026-03-18  
**状态**: 已批准  
**方案**: 方案 B — InsightFace 检测 + 关键点对齐 + 图像增强

---

## 1. 背景与目标

### 问题

当前 `/vrlFaceIdComparison` 人证比对端点直接将证件照原图送入 `gen_verify_res` 进行比对。证件照（身份证、护照等）通常包含证件边框、文字、背景等干扰信息，且实拍场景下可能存在倾斜、反光等问题，影响比对准确率。

### 目标

在人证比对流程中增加证件照预处理步骤，自动从证件照中定位、对齐、裁剪人脸区域并做轻量图像增强，提升比对准确率。

### 约束

- 支持多国证件（不限定具体证件类型）
- 支持实拍照片（可能倾斜、反光）和规整证件图片
- 仅作为人证比对内部预处理，不对外暴露独立 API
- 零新依赖（复用 InsightFace + OpenCV）
- 预处理失败时兜底返回原图，不影响现有流程

---

## 2. 技术方案

### 2.1 方案选型

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A: bbox 裁剪 | InsightFace 检测 + ROI 扩展裁剪 | 简单，延迟低 | 对倾斜证件照鲁棒性差 |
| **B: 关键点对齐（选定）** | InsightFace 检测 + 5点仿射对齐 + 裁剪 + 增强 | 零新依赖，对倾斜鲁棒 | 实现稍复杂 |
| C: 证件区域检测 | 边缘检测定位证件 + 透视矫正 + 人脸裁剪 | 对实拍最鲁棒 | 复杂度高，泛化性差 |

选定方案 B：利用 InsightFace 已返回的 5 个关键点做仿射变换对齐，投入产出比最优。

### 2.2 处理流程

```
输入: 证件照原图 (np.ndarray, RGB)
  │
  ├─ 1. InsightFace 人脸检测
  │     → 获取 bbox + 5 关键点 (左眼、右眼、鼻尖、左嘴角、右嘴角)
  │     → 未检测到人脸 → 返回原图兜底
  │     → 多张人脸 → 取面积最大的
  │
  ├─ 2. 仿射变换对齐 (_align_face)
  │     → 以双眼中心为基准计算旋转角度
  │     → 仿射变换将人脸旋转到水平姿态
  │
  ├─ 3. ROI 扩展裁剪 (_crop_face_roi)
  │     → bbox 向外扩展 30%（可配置）
  │     → 确保额头、下巴、耳朵不被截断
  │     → 边界裁剪到图像范围内
  │
  └─ 4. 图像增强 (_enhance_image)
        → CLAHE 自适应直方图均衡（改善光照不均）
        → 高斯去噪（针对扫描件噪点）
        → 不做锐化（避免伪影）

输出: 预处理后的人脸图像 (np.ndarray, RGB)
```

---

## 3. 模块设计

### 3.1 新增文件

**`vrlFace/face/id_preprocess.py`** — 证件照人脸预处理模块

公开函数：

```python
def preprocess_id_photo(img: np.ndarray) -> np.ndarray:
    """
    证件照预处理主入口

    流程：检测 → 对齐 → 裁剪 → 增强
    如果预处理失败（未检测到人脸等），返回原图兜底

    Args:
        img: RGB 图像 numpy 数组

    Returns:
        预处理后的 RGB 图像 numpy 数组
    """
```

私有函数：

```python
def _align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    基于 5 关键点做仿射变换对齐

    以双眼中心连线为基准，计算旋转角度，
    仿射变换将人脸旋转到水平姿态。

    Args:
        img: 原图
        landmarks: 5 个关键点坐标 shape=(5, 2)

    Returns:
        对齐后的图像
    """

def _crop_face_roi(
    img: np.ndarray,
    bbox: np.ndarray,
    expand_ratio: float = 0.3,
) -> np.ndarray:
    """
    基于 bbox 扩展裁剪人脸 ROI

    Args:
        img: 图像
        bbox: 人脸边界框 [x1, y1, x2, y2]
        expand_ratio: 扩展比例（默认 0.3）

    Returns:
        裁剪后的图像
    """

def _enhance_image(img: np.ndarray) -> np.ndarray:
    """
    轻量图像增强

    - CLAHE 自适应直方图均衡
    - 高斯去噪

    Args:
        img: RGB 图像

    Returns:
        增强后的 RGB 图像
    """
```

### 3.2 修改文件

**`vrlFace/face/config.py`** — 新增 2 个配置字段：

```python
# FaceConfig 类中新增
id_crop_expand_ratio: float = 0.3    # 证件照 ROI 扩展比例
id_enhance_enabled: bool = True       # 是否启用图像增强

# from_env() 中新增
id_crop_expand_ratio=float(os.getenv("FACE_ID_CROP_EXPAND", "0.3")),
id_enhance_enabled=os.getenv("FACE_ID_ENHANCE", "true").lower() in ("true", "1", "yes"),
```

**`vrlFace/face/api.py`** — `/vrlFaceIdComparison` 端点增加一行调用：

```python
from .id_preprocess import preprocess_id_photo

# 在 vrl_face_id_comparison 函数中，gen_verify_res 调用前：
image1 = preprocess_id_photo(image1)  # 仅对证件照做预处理
```

### 3.3 不改动的部分

- `recognizer.py` — 核心识别逻辑不动
- `/vrlFaceComparison` — 普通比对不受影响
- `/vrlFaceDetection`、`/vrlFaceSearch` — 不受影响

---

## 4. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 未检测到人脸 | `logger.warning`，返回原图 |
| 多张人脸 | 取面积最大的，`logger.info` 记录 |
| 仿射变换异常 | catch + `logger.warning`，返回原图 |
| 增强处理异常 | catch + `logger.warning`，返回对齐裁剪后的图（跳过增强） |
| 任何未预期异常 | 最外层 catch，返回原图 |

核心原则：预处理是增强手段，绝不因预处理失败导致比对请求失败。

---

## 5. 配置项

| 配置字段 | 环境变量 | 默认值 | 说明 |
|----------|----------|--------|------|
| `id_crop_expand_ratio` | `FACE_ID_CROP_EXPAND` | `0.3` | ROI 扩展比例 |
| `id_enhance_enabled` | `FACE_ID_ENHANCE` | `true` | 是否启用图像增强 |
| `id_comparison_threshold` | `FACE_ID_THRESHOLD` | `0.60` | 人证比对阈值（已有） |

---

## 6. 性能影响

| 步骤 | 预计耗时 |
|------|----------|
| InsightFace 检测 | ~0ms（复用已有检测结果） |
| 仿射变换对齐 | ~5ms |
| ROI 裁剪 | ~1ms |
| CLAHE + 去噪 | ~10ms |
| **总计** | **~15-20ms** |

无新依赖引入。内存开销可忽略（仅临时图像副本）。

---

## 7. 测试策略

新增 `tests/face/test_id_preprocess.py`：

| 测试用例 | 验证内容 |
|----------|----------|
| 正常证件照输入 | 返回裁剪对齐后的图像，尺寸合理 |
| 无人脸图片 | 返回原图，不抛异常 |
| 多人脸图片 | 取面积最大的人脸 |
| 倾斜证件照 | 对齐后人脸水平 |
| 增强开关关闭 | `id_enhance_enabled=False` 时跳过增强 |
| 配置项生效 | `id_crop_expand_ratio` 影响裁剪范围 |

---

## 8. 文件变更清单

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `vrlFace/face/id_preprocess.py` | 新增 | ~120 行 |
| `vrlFace/face/config.py` | 修改 | +6 行（2 字段 + 2 env + display） |
| `vrlFace/face/api.py` | 修改 | +2 行（import + 调用） |
| `tests/face/test_id_preprocess.py` | 新增 | ~80 行 |
