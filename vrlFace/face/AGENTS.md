# vrlFace.face — 人脸识别模块

**位置**: `vrlFace/face/`  
**职责**: 基于 InsightFace 的人脸检测、1:1 比对、1:N 搜索

## 结构

```
face/
├── __init__.py       # 导出：face_detection, gen_verify_res, face_search
├── config.py         # FaceConfig 类（环境变量支持）
├── recognizer.py     # 核心识别逻辑（8.4KB）
├── api.py            # FastAPI 路由（5.6KB）
└── cli.py            # 命令行工具（5.8KB）
```

## 核心 API

### 人脸检测
```python
from vrlFace.face import face_detection

result = face_detection("photo.jpg")
# 返回：{
#   'is_face_exist': True,
#   'face_num': 1,
#   'faces_detected': [{'facial_area': {...}, 'confidence': 0.99}]
# }
```

### 1:1 比对
```python
from vrlFace.face import gen_verify_res

result = gen_verify_res("img1.jpg", "img2.jpg")
# 返回：{
#   'is_face_exist': 1,
#   'is_same_face': 1,
#   'confidence': 0.85,
#   'confidence_exist': [0.99, 0.98]
# }
```

### 1:N 搜索
```python
from vrlFace.face import face_search

result = face_search("query.jpg", db_path="data/dataset", top_n=3)
# 返回：{
#   'has_similar_picture': 1,
#   'searched_similar_pictures': [{'picture': '...', 'confidence': 0.95}]
# }
```

## 配置

### 环境变量
```bash
FACE_MODEL_NAME=buffalo_l      # 模型名称
FACE_DET_SIZE=640,640          # 检测尺寸
FACE_GPU_ID=-1                 # GPU ID（-1=CPU）
FACE_THRESHOLD=0.55            # 相似度阈值
FACE_IMAGES_BASE=/app/data     # 人脸库路径
```

### 阈值推荐
| 场景 | 阈值 | 说明 |
|------|------|------|
| 人证比对 | 0.70 | 高安全场景 |
| 普通比对 | 0.55 | 日常应用 |
| 宽松模式 | 0.40 | 提高召回率 |

**注意**: 最近新增阈值自动调整功能，支持根据场景动态调整

## 关键类/函数

| 符号 | 文件 | 职责 |
|------|------|------|
| `FaceConfig` | `config.py` | 配置管理（懒加载 InsightFace） |
| `face_detection()` | `recognizer.py` | 人脸检测入口 |
| `gen_verify_res()` | `recognizer.py` | 1:1 比对入口 |
| `face_search()` | `recognizer.py` | 1:N 搜索入口 |
| `get_recognizer()` | `recognizer.py` | 单例 InsightFace 实例 |

## API 路由

| 端点 | 方法 | 说明 |
|------|------|------|
| `/vrlFaceDetection` | POST | 人脸检测 |
| `/vrlFaceComparison` | POST | 1:1 比对 |
| `/vrlFaceSearch` | POST | 1:N 搜索 |
| `/vrlFaceIdComparison` | POST | 人证比对 |

## 技术细节

**特征提取**: InsightFace buffalo_l 模型 → 512 维向量  
**相似度计算**: 余弦相似度  
**单例模式**: `get_recognizer()` 避免重复加载模型  
**输入支持**: 图片路径 / numpy 数组 / PIL.Image  

## 命令行

```bash
# 演示模式
uv run python -m vrlFace.face.cli --demo

# 1:1 比对
uv run python -m vrlFace.face.cli --img1 a.jpg --img2 b.jpg

# 安装入口点后
face-verify --demo
face-verify --img1 a.jpg --img2 b.jpg
```

## 性能

- 单次检测：~80ms
- 单次比对：~100ms
- 内存占用：~300MB

## 故障排查

**未检测到人脸**:
- 增大检测尺寸：`FACE_DET_SIZE=1024,1024`
- 使用轻量模型：`FACE_MODEL_NAME=buffalo_s`

**比对不准确**:
- 调整阈值：`FACE_THRESHOLD=0.60`

**性能慢**:
- 使用 GPU：`FACE_GPU_ID=0`
- 减小检测尺寸：`FACE_DET_SIZE=320,320`
