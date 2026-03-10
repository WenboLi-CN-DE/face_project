# vrlFace 人脸识别模块

基于 **InsightFace** 的高性能人脸识别模块。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r ../requirements.txt
```

### 2. 基本使用

```python
from vrlFace.models import face_detection, gen_verify_res, face_search

# 人脸检测
result = face_detection("test.jpg")
print(f"检测到 {result['face_num']} 张人脸")

# 人脸比对
result = gen_verify_res("img1.jpg", "img2.jpg")
if result['is_same_face']:
    print(f"同一人，置信度：{result['confidence']:.2%}")

# 人脸搜索
result = face_search("query.jpg", db_path="data/dataset", top_n=3)
for item in result['searched_similar_pictures']:
    print(f"匹配：{item['confidence']:.2%}")
```

## 📦 API 参考

### face_detection(img)
检测图片中的人脸。

**返回:**
```python
{
    'is_face_exist': True,
    'face_num': 1,
    'faces_detected': [{
        'facial_area': {'x': 100, 'y': 100, 'width': 200, 'height': 200},
        'confidence': 0.99
    }]
}
```

### gen_verify_res(img1, img2)
比对两张图片的人脸。

**返回:**
```python
{
    'is_face_exist': 1,
    'is_same_face': 1,
    'confidence': 0.85,
    'confidence_exist': [0.99, 0.98]
}
```

### face_search(img, db_path, top_n=3)
在数据库中搜索相似人脸。

**返回:**
```python
{
    'has_similar_picture': 1,
    'searched_similar_pictures': [
        {'picture': 'path/to/img.jpg', 'confidence': 0.95}
    ]
}
```

## ⚙️ 配置

在根目录的 `config.py` 中统一配置：

```python
# 环境变量方式
export FACE_MODEL_NAME=buffalo_l
export FACE_DET_SIZE=640,640
export FACE_THRESHOLD=0.55
```

## 📈 性能

- 单次检测：~80ms
- 单次比对：~100ms
- 内存占用：~300MB

## 📁 目录结构

```
vrlFace/
├── models.py              # 核心识别逻辑
├── README.md              # 使用说明
└── data/                  # 测试数据
    ├── t1.jpg
    ├── t2.jpg
    └── dataset/           # 人脸数据库
```

## 🔧 故障排查

**未检测到人脸？**
- 增大检测尺寸：`FACE_DET_SIZE=1024,1024`
- 使用轻量模型：`FACE_MODEL_NAME=buffalo_s`

**比对不准确？**
- 调整阈值：`FACE_THRESHOLD=0.60`

**性能慢？**
- 使用 GPU：`FACE_GPU_ID=0`
- 减小检测尺寸：`FACE_DET_SIZE=320,320`
