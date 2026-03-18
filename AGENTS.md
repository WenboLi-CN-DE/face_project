# vrlFace 项目知识库

**生成时间**: 2026-03-16  
**更新时间**: 2026-03-16 (init-deep)
**提交**: b45550c fix: 优化 mouth_open 张嘴检测逻辑
**项目类型**: Python 人脸识别与活体检测系统  
**核心栈**: InsightFace + MediaPipe + FastAPI  
**版本**: 3.0.0

---

## 快速导航

| 目录 | AGENTS.md | 说明 |
|------|-----------|------|
| `.` | 本文档 | 项目总览、命令、部署 |
| `vrlFace/` | [vrlFace/AGENTS.md](vrlFace/AGENTS.md) | 核心包结构、入口点 |
| `vrlFace/face/` | [vrlFace/face/AGENTS.md](vrlFace/face/AGENTS.md) | 人脸识别模块 |
| `vrlFace/liveness/` | [vrlFace/liveness/AGENTS.md](vrlFace/liveness/AGENTS.md) | 活体检测模块 |
| `scripts/` | [scripts/AGENTS.md](scripts/AGENTS.md) | 运维脚本工具 |
| `deploy/` | [deploy/AGENTS.md](deploy/AGENTS.md) | Docker 三环境部署 |
| `tests/` | [tests/AGENTS.md](tests/AGENTS.md) | 测试套件结构 |

---

## 概述

vrlFace 是基于 **InsightFace + MediaPipe** 的高性能人脸识别与活体检测系统，提供 RESTful API 服务。

- **vrlFace.face** — 人脸识别（检测/1:1 比对/1:N 搜索）
- **vrlFace.liveness** — 活体检测（MediaPipe 动作验证/视频分析）

---

## 项目结构

```
.
├── vrlFace/              # 核心包（人脸识别 + 活体检测）
│   ├── face/             # 人脸识别子包（5 个文件）
│   └── liveness/         # 活体检测子包（22 个文件，6 个大文件）
├── tests/                # 测试目录（镜像源码结构）
│   ├── face/             # 人脸识别测试
│   └── liveness/         # 活体检测测试（6 个测试文件）
├── deploy/               # Docker 部署配置（dev/tra/pro）
├── scripts/              # 运维/分析工具脚本（15+ 个）
│   └── manual_tests/     # 手动测试/诊断脚本（10+ 个）
│       └── remote_fetch_tests/  # 远程拉取视频测试（5 个）
├── docs/                 # 文档（docker-log-workflow, remote_fetch 等）
├── models/buffalo_l/     # InsightFace 模型
├── data/                 # 测试数据/人脸库
└── output/               # 输出结果
    ├── results/          # 测试结果、CSV、分析报告
    ├── logs/             # 日志文件
    ├── remote_fetch/     # 远程拉取数据
    │   ├── videos/       # 拉取的视频文件
    │   ├── 动作对应.txt   # 视频动作配置
    │   └── archive/      # 历史报告归档
    └── temp/             # 临时文件
```

## 查找位置

### 核心功能
| 任务 | 位置 | 说明 |
|------|------|------|
| 启动 API 服务 | `run.py` | 统一入口，端口 8070 |
| 人脸识别 CLI | `vrlFace/face/cli.py` | 命令行工具 |
| 活体检测 CLI | `vrlFace/liveness/cli.py` | 摄像头/视频检测 |
| 修改人脸 API 路由 | `vrlFace/face/api.py` | FastAPI 路由 |
| 修改活体 API 路由 | `vrlFace/liveness/api.py` | FastAPI 路由 |
| 调整人脸阈值 | `vrlFace/face/config.py` | FaceConfig 类 |
| 调整活体阈值 | `vrlFace/liveness/config.py` | LivenessConfig 类 |
| 核心识别逻辑 | `vrlFace/face/recognizer.py` | 特征提取/相似度计算 |
| 活体融合引擎 | `vrlFace/liveness/fusion_engine.py` | 多信号融合 |

### 活体检测
| 组件 | 位置 | 说明 |
|------|------|------|
| MediaPipe 检测 | `vrlFace/liveness/mediapipe_detector.py` | 478 关键点检测 |
| 轻量检测器 | `vrlFace/liveness/fast_detector.py` | 状态机检测器 |
| 视频分析器 | `vrlFace/liveness/video_analyzer.py` | 逐段分析 |
| 头部动作 | `vrlFace/liveness/head_action.py` | 点头/摇头事件 |
| 异步回调 | `vrlFace/liveness/async_processor.py` + `callback.py` | 后台任务 +HTTP 回调 |
| 视频旋转 | `vrlFace/liveness/video_rotation.py` | 自动旋转修正 |
| 基准校准 | `vrlFace/liveness/benchmark_calibrator.py` | 防替换攻击 |

### 运维工具
| 工具 | 位置 | 说明 |
|------|------|------|
| 日志分析 | `scripts/log_parser.py` | 日志解析为 JSON/CSV |
| 视频日志分析 | `scripts/log_video_analyzer.py` | 关联日志与视频帧 |
| SSH 配置 | `scripts/ssh_config.py` | SSH 配置管理 |
| SSH 设置 | `scripts/setup_ssh_access.sh` | 一键设置 SSH |
| 远程获取 | `scripts/remote_fetch.py` | Docker 数据拉取 |
| 批量测试 | `scripts/manual_tests/batch_test_videos.py` | 批量视频动作测试 |
| 活体调试 | `scripts/manual_tests/debug_is_liveness.py` | 逐帧活体判定调试 |

## 代码地图

| 符号 | 类型 | 位置 | 作用 |
|------|------|------|------|
| `LivenessFusionEngine` | 类 | `liveness/fusion_engine.py` | 活体检测主入口，融合多信号 |
| `LivenessConfig` | 类 | `liveness/config.py` | 配置预设（CPU 快速/实时/视频防伪） |
| `MediaPipeLivenessDetector` | 类 | `liveness/mediapipe_detector.py` | MediaPipe 完整推理检测器 |
| `FastLivenessDetector` | 类 | `liveness/fast_detector.py` | 轻量状态机检测器 |
| `VideoLivenessAnalyzer` | 类 | `liveness/video_analyzer.py` | 视频逐段分析器 |
| `HeadActionDetector` | 类 | `liveness/head_action.py` | 头部动作事件检测 |
| `BenchmarkCalibrator` | 类 | `liveness/benchmark_calibrator.py` | 基准帧校准器（防替换攻击） |
| `face_detection` | 函数 | `face/recognizer.py` | 人脸检测入口 |
| `gen_verify_res` | 函数 | `face/recognizer.py` | 1:1 比对入口 |
| `face_search` | 函数 | `face/recognizer.py` | 1:N 搜索入口 |
| `FaceConfig` | 类 | `face/config.py` | 人脸识别配置（支持环境变量） |

## 约定

**代码风格**:
- 遵循 PEP 8，最大行长度 **100 字符**
- 所有函数签名使用类型提示
- 使用双引号 `"` 作为字符串引号
- Google 风格文档字符串

**命名约定**:
- 变量/函数：`snake_case`（如 `user_name`）
- 类：`CamelCase`（如 `FaceVerification`）
- 常量：`ALL_CAPS`（如 `THRESHOLD_STRICT`）

**配置阈值**:
```python
# 人脸识别
THRESHOLD_STRICT = 0.70    # 人证比对
THRESHOLD_NORMAL = 0.55    # 普通比对
THRESHOLD_LOOSE = 0.40     # 宽松模式
DET_SIZE = (640, 640)      # 检测尺寸

# 活体检测（默认值）
ear_threshold = 0.20       # 眨眼
mar_threshold = 0.55       # 张嘴
yaw_threshold = 15°        # 左右转头
pitch_threshold = 15°      # 上下点头
```

## 反模式（本项目禁止）

- ❌ 硬编码密钥 — 使用 `os.getenv()` 或 `.env`
- ❌ 裸 `except` — 必须捕获特定异常
- ❌ 删除失败测试 — 修复而非删除
- ❌ 类型错误抑制 — 禁止 `as any`、`@ts-ignore`（Python 中为 `# type: ignore` 也应避免）
- ❌ 在根目录放置临时脚本 — 移至 `scripts/`
- ❌ 输出文件（CSV/log）放在根目录 — 移至 `output/`

## 独特风格

- **懒加载设计**: `vrlFace/__init__.py` 使用 `__getattr__` 实现懒加载，避免服务启动时强制加载不需要的模块
- **向后兼容层**: `vrlFace/` 根目录的 `main.py`, `config.py`, `models.py` 等仅作为兼容层，实际导入自子包
- **分离式服务**: 人脸识别和活体检测可独立部署（端口 8070/8071），也可合并部署
- **异步回调架构**: 活体检测支持长视频后台处理，完成后 HTTP 回调通知（HMAC-SHA256 签名）
- **基准帧校准**: 防替换攻击，动态采集高质量正面人脸帧作为基准（`benchmark_calibrator.py`）
- **阈值自动调整**: 最近新增功能，支持根据场景自动调整检测阈值

## 命令

```bash
# 环境设置
uv sync                    # 同步依赖
uv sync --extra dev        # 同步开发依赖
uv venv                    # 创建虚拟环境

# 运行
uv run python run.py                     # 启动 API 服务（8070 端口）
uv run python -m vrlFace.face.cli --demo  # 人脸识别演示
uv run python -m vrlFace.liveness.cli --camera 0  # 摄像头活体检测
uv run python -m vrlFace.liveness.cli --video path/to/video.mp4  # 视频检测

# 测试
uv run python tests/test_face.py         # 人脸识别测试
uv run python tests/test_liveness.py     # 活体检测测试
pytest tests/ -v                         # 运行所有测试

# 代码检查
uv run flake8 .              # 风格检查
uv run mypy .                # 类型检查
uv run black --check .       # 格式检查
uv run black .               # 格式化
uv run isort .               # 排序导入
```

## CI/CD 与部署

**GitHub Actions** (`.github/workflows/deploy.yml`):
```
push → test (pytest+flake8) → build (Docker) → deploy (SSH)
```

**Docker 三环境**:
| 环境 | Workers | 日志 | 内存 | 特性 |
|------|---------|------|------|------|
| `deploy/dev` | 默认 | 默认 | 4G | 热重载，代码挂载 |
| `deploy/tra` | 1 | debug | 4G | 日志轮转，unless-stopped |
| `deploy/pro` | 2 | info | 8G | 持久化模型卷，外部视频路径 |

**服务分离**: 支持独立部署人脸 (8070) 和活体 (8071) 服务
```bash
bash up.sh face    # 仅人脸
bash up.sh liveness # 仅活体
bash up.sh all     # 全部
```

**裸机部署**: `deploy/scripts/deploy_bare.sh` — Ubuntu 20.04/22.04 systemd+Nginx

---

## 注意事项

- **GPU 加速**: 推荐使用 `onnxruntime-gpu`，设置 `FACE_GPU_ID=0`
- **模型位置**: buffalo_l 模型存放在 `models/buffalo_l/`；生产环境使用持久化卷 `insightface-models`
- **环境变量**: 参考 `.env.example`，生产环境使用 `deploy/pro/.env`
- **架构文档**: 详细架构见 `ARCHITECTURE.md`
- **文档目录**: `docs/` 包含 docker-log-workflow、remote_fetch、ssh-config-usage 等文档

