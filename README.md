# vrlFace — 人脸识别 & 活体检测系统

基于 **InsightFace + MediaPipe** 的高性能人脸识别与活体检测系统。

## 🏗️ 项目结构

```
face_cls/
├── Dockerfile                    # 多阶段构建镜像
├── .dockerignore
├── pyproject.toml
├── requirements.txt
├── run.py                        # 启动 API 服务
├── start_dev.sh                  # 本地开发启动
│
├── vrlFace/                      # 主包
│   ├── __init__.py               # 统一导出两大模块 API
│   │
│   ├── face/                     # 🔷 人脸识别子包
│   │   ├── __init__.py
│   │   ├── config.py             # FaceConfig（支持环境变量）
│   │   ├── recognizer.py         # 人脸检测 / 1:1比对 / 1:N搜索
│   │   ├── api.py                # FastAPI 路由
│   │   └── cli.py                # 命令行工具
│   │
│   ├── liveness/                 # 🟢 活体检测子包
│   │   ├── __init__.py
│   │   ├── config.py             # LivenessConfig
│   │   ├── mediapipe_detector.py # MediaPipe 推理检测器
│   │   ├── fast_detector.py      # 轻量状态机检测器
│   │   ├── fusion_engine.py      # 多信号融合引擎
│   │   ├── head_action.py        # 头部动作事件检测
│   │   ├── insightface_quality.py# 质量评估
│   │   ├── utils.py              # 共享工具函数
│   │   ├── cli.py                # 命令行：摄像头/视频检测
│   │   ├── recorder.py           # CSV 录制器
│   │   ├── detectors/            # 检测器子包（聚合导出）
│   │   └── quality/              # 质量评估子包（聚合导出）
│   │
│   ├── main_fastapi.py           # FastAPI 应用入口
│   │
│   # ── 向后兼容层（旧路径仍可用）──
│   ├── config.py  → face/config.py
│   ├── models.py  → face/recognizer.py
│   ├── main.py    → face/cli.py
│   ├── demo.py    → face/cli.py
│   ├── liveness_main.py   → liveness/cli.py
│   ├── save_csv.py        → liveness/recorder.py
│   └── liveness_example.py→ liveness/cli.py
│
├── tests/
│   ├── face/test_face.py         # 人脸识别单元测试
│   └── liveness/test_liveness.py # 活体检测测试
│
├── scripts/                      # 分析/验证工具脚本
│
└── deploy/                       # 部署配置
    ├── .env.example              # 环境变量模板
    ├── dev/                      # 开发环境（热重载）
    ├── tra/                      # 测试环境（1 worker, DEBUG日志）
    ├── pro/                      # 生产环境（2 workers, INFO日志）
    └── scripts/
        ├── deploy_bare.sh        # Ubuntu 裸机一键部署
        └── install_dev_deps.bat  # Windows 开发依赖安装
```

---

## 🚀 快速开始

### 安装

```bash
# 1. 克隆项目
git clone <repository-url>
cd face_cls

# 2. 安装依赖
pip install -r requirements.txt
# 或
pip install -e ".[api]"
```

### 启动 API 服务

```bash
# 开发模式（热重载）
bash start_dev.sh
# 或
uvicorn vrlFace.main_fastapi:app --reload --host 0.0.0.0 --port 8070

# 访问文档
# http://localhost:8070/docs
```

### Python 使用

```python
# 人脸识别
from vrlFace.face import face_detection, gen_verify_res, face_search

result = face_detection("photo.jpg")
result = gen_verify_res("img1.jpg", "img2.jpg")
result = face_search("query.jpg", db_path="data/dataset")

# 活体检测
from vrlFace.liveness import LivenessFusionEngine, LivenessConfig

config = LivenessConfig.cpu_fast_config()
engine = LivenessFusionEngine(config)
result = engine.process_frame(frame)
print(result.is_live, result.score)
```

---

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/healthz` | GET | 健康检查 |
| `/vrlFaceDetection` | POST | 人脸检测 |
| `/vrlFaceComparison` | POST | 1:1 人脸比对 |
| `/vrlFaceSearch` | POST | 1:N 人脸搜索 |

---

## 🖥️ 命令行工具

```bash
# 人脸识别
python -m vrlFace.face.cli --demo
python -m vrlFace.face.cli --img1 a.jpg --img2 b.jpg

# 活体检测（摄像头）
python -m vrlFace.liveness.cli --camera 0

# 活体检测（视频文件）
python -m vrlFace.liveness.cli --video path/to/video.mp4 --config video-anti

# 视频 CSV 录制
python -m vrlFace.liveness.recorder --video path/to/video.mp4

# 安装入口点后可直接使用
face-verify --demo
face-liveness --camera 0
face-recorder --video video.mp4
```

---

## ⚙️ 配置

### 人脸识别配置（环境变量）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `FACE_MODEL_NAME` | `buffalo_l` | 模型名称 |
| `FACE_DET_SIZE` | `640,640` | 检测尺寸 |
| `FACE_GPU_ID` | `-1` | GPU ID（-1=CPU）|
| `FACE_THRESHOLD` | `0.55` | 相似度阈值 |
| `FACE_IMAGES_BASE` | `/app/data/dataset` | 人脸库路径 |

### 活体检测配置预设

```python
from vrlFace.liveness import LivenessConfig

LivenessConfig.cpu_fast_config()          # CPU 快速模式
LivenessConfig.realtime_config()          # 实时摄像头模式
LivenessConfig.video_anti_spoofing_config() # 视频防伪模式
```

---

## 🐳 Docker 部署

```bash
# 开发环境
cd deploy/dev && bash up.sh

# 测试环境
cd deploy/tra && bash up.sh

# 生产环境（先复制 .env.example 为 .env 并填写配置）
cd deploy/pro
cp .env.example .env
bash up.sh
```

---

## 🧪 测试

```bash
pytest tests/ -v
pytest tests/face/ -v       # 仅人脸识别测试
pytest tests/liveness/ -v   # 仅活体检测测试
```

---

## 📦 依赖

- `insightface >= 0.7.3` — 人脸识别引擎
- `mediapipe >= 0.10.0` — 活体检测（面部关键点）
- `opencv-python >= 4.8.0` — 图像处理
- `onnxruntime >= 1.16.0` — 模型推理
- `fastapi` + `uvicorn` — REST API 服务
