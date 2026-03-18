# vrlFace 核心包

**位置**: `vrlFace/`  
**职责**: 人脸识别与活体检测核心模块

## 结构

```
vrlFace/
├── __init__.py           # 懒加载顶层导出（__getattr__）
├── main_fastapi.py       # FastAPI 合并应用入口
├── face_app.py           # 人脸识别独立 API 服务
├── liveness_app.py       # 活体检测独立 API 服务
│
├── face/                 # 人脸识别子包
│   ├── __init__.py
│   ├── config.py         # FaceConfig（环境变量支持）
│   ├── recognizer.py     # 核心：检测/比对/搜索
│   ├── api.py            # FastAPI 路由
│   └── cli.py            # 命令行工具
│
├── liveness/             # 活体检测子包（见 liveness/AGENTS.md）
│   ├── __init__.py
│   ├── config.py         # LivenessConfig（预设配置）
│   ├── fusion_engine.py  # 多信号融合引擎
│   ├── mediapipe_detector.py  # MediaPipe 478 关键点
│   ├── fast_detector.py  # 轻量状态机检测器
│   ├── video_analyzer.py # 视频逐段分析
│   ├── head_action.py    # 头部动作事件
│   ├── async_processor.py# 异步任务处理器
│   ├── callback.py       # HTTP 回调客户端
│   └── ...
│
└── data/                 # 数据包
    ├── dataset/          # 人脸库
    └── dataset-o/        # 备用数据集
```

## 入口点

### API 服务
| 文件 | 端口 | 模式 |
|------|------|------|
| `run.py` (根目录) | 8070 | 开发启动（热重载） |
| `main_fastapi.py` | 8070 | 合并部署（人脸 + 活体） |
| `face_app.py` | 8070 | 独立部署（仅人脸） |
| `liveness_app.py` | 8071 | 独立部署（仅活体） |

### CLI 工具
| 文件 | 命令 | 功能 |
|------|------|------|
| `face/cli.py` | `python -m vrlFace.face.cli` | 人脸比对演示 |
| `liveness/cli.py` | `python -m vrlFace.liveness.cli` | 摄像头/视频检测 |
| `liveness/recorder.py` | `python -m vrlFace.liveness.recorder` | 视频 CSV 录制 |

### pyproject.toml 入口点 (5 个)
```
face-verify     → vrlFace.face.cli:main
face-liveness   → vrlFace.liveness.cli:main
face-recorder   → vrlFace.liveness.recorder:main
face-server     → vrlFace.face_app:app
liveness-server → vrlFace.liveness_app:app
```

## 懒加载设计

`vrlFace/__init__.py` 使用 `__getattr__` 实现懒加载：
- `face_detection`, `gen_verify_res`, `face_search` → 导入自 `vrlFace.face`
- `LivenessFusionEngine`, `LivenessConfig` → 导入自 `vrlFace.liveness`

**优势**: 服务启动时不强制加载不需要的模块

## 向后兼容层

以下文件仅作为兼容层，实际逻辑在子包：
- `config.py` → `face/config.py`
- `models.py` → `face/recognizer.py`
- `main.py`, `demo.py` → `face/cli.py`
- `liveness_main.py`, `liveness_example.py` → `liveness/cli.py`
- `save_csv.py` → `liveness/recorder.py`

## 查找位置

| 任务 | 文件 |
|------|------|
| 修改人脸识别 API | `face/api.py` |
| 修改活体检测 API | `liveness/api.py` |
| 调整相似度阈值 | `face/config.py` |
| 调整活体阈值 | `liveness/config.py` |
| 添加新动作类型 | `liveness/head_action.py` |
| 修改融合逻辑 | `liveness/fusion_engine.py` |

## 依赖关系

```
vrlFace/
├── face/ → insightface, onnxruntime, opencv
└── liveness/ → mediapipe, opencv, httpx (回调)

# 开发依赖
# pytest, black, flake8, mypy, paramiko (SSH), tqdm (进度条)
```

## 部署模式

### Docker 三环境
| 环境 | Workers | 日志 | 内存 | 特性 |
|------|---------|------|------|------|
| `deploy/dev` | 默认 | 默认 | 4G | 热重载，代码挂载 |
| `deploy/tra` | 1 | debug | 4G | 日志轮转，unless-stopped |
| `deploy/pro` | 2 | info | 8G | 持久化模型卷，外部视频路径 |

### 服务分离部署
```bash
bash up.sh face       # 仅人脸 (8070)
bash up.sh liveness   # 仅活体 (8071)
bash up.sh all        # 全部服务
```

### 单端口部署（推荐）
```bash
uvicorn vrlFace.main_fastapi:app --port 8070
```

### 双端口部署（独立扩展）
```bash
uvicorn vrlFace.face_app:app --port 8070
uvicorn vrlFace.liveness_app:app --port 8071
```

### 裸机部署
`deploy/scripts/deploy_bare.sh` — Ubuntu 20.04/22.04 systemd+Nginx

## 最近变更

- **阈值自动调整**: 人脸识别和活体检测均支持动态阈值
- **基准帧校准**: 活体检测新增防替换攻击功能
- **持久化模型**: InsightFace 模型持久化，避免重复下载
- **头部动作优化**: 从基线追踪改为峰峰值检测
