# vrlFace 部署指南

> 版本 3.1.0 · 更新于 2026-03-22

## 服务架构

```
┌─────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐
│   vrl-face      │  │   vrl-liveness      │  │   vrl-silent         │
│   :8070         │  │   :8071             │  │   :8060              │
│                 │  │                     │  │                      │
│  InsightFace    │  │  MediaPipe          │  │  deepface-           │
│  ONNX Runtime   │  │  动作活体检测        │  │  antispoofing        │
│  人脸识别        │  │  (眨眼/张嘴/点头)    │  │  静默活体检测         │
├─────────────────┤  ├─────────────────────┤  ├──────────────────────┤
│  Dockerfile     │  │  Dockerfile         │  │  Dockerfile.silent   │
│  requirements   │  │  requirements       │  │  requirements-silent │
│  .txt           │  │  .txt               │  │  .txt                │
└─────────────────┘  └─────────────────────┘  └──────────────────────┘
```

| 服务 | 端口 | 镜像 | 技术栈 | 接口 |
|------|------|------|--------|------|
| 人脸识别 | 8070 | `Dockerfile` | InsightFace + ONNX | `/vrlFaceDetection`, `/vrlFaceComparison`, `/vrlFaceSearch` |
| 动作活体 | 8071 | `Dockerfile` | MediaPipe | `/vrlMoveLiveness` |
| 静默活体 | 8060 | `Dockerfile.silent` | deepface-antispoofing + TensorFlow | `/vrlSilentLiveness` |

## 前置条件

- Docker Engine ≥ 20.10
- Docker Compose V2 (`docker compose`)
- 服务器可用内存 ≥ 8GB（全量部署建议 16GB+）
- `/data/videos` 目录存在（活体检测视频/图片存放路径）

## 快速开始

### 1. 克隆代码

```bash
git clone git@github.com:WenboLi-CN-DE/face_project.git
cd face_project
```

### 2. 配置环境变量

```bash
cd deploy/pro          # 或 deploy/dev、deploy/tra
cp ../.env.example .env
nano .env              # 编辑实际配置
```

`.env` 关键配置项：

```bash
FACE_MODEL_NAME=buffalo_l       # InsightFace 模型
FACE_DET_SIZE=640,640           # 检测尺寸
FACE_GPU_ID=-1                  # GPU ID（-1=CPU）
FACE_THRESHOLD=0.55             # 人脸比对阈值
FACE_IMAGES_BASE=/app/data/dataset  # 人脸库路径
```

### 3. 准备模型文件

```bash
# 确保 InsightFace 模型存在
ls models/buffalo_l/
# 应包含: det_10g.onnx, w600k_r50.onnx 等

# 确保 MediaPipe 模型存在
ls models/face_landmarker.task
```

### 4. 启动服务

```bash
cd deploy/pro    # 生产环境
bash up.sh all   # 启动全部 3 个服务
```

## 部署命令

### 启动

```bash
cd deploy/{dev|tra|pro}

bash up.sh face      # 仅人脸识别 (8070)
bash up.sh liveness  # 仅动作活体 (8071)
bash up.sh silent    # 仅静默活体 (8060)
bash up.sh all       # 全部服务
```

### 停止

```bash
bash down.sh face
bash down.sh liveness
bash down.sh silent    # docker compose -f docker-compose.silent.yaml down
bash down.sh all
```

### 查看状态

```bash
docker compose ps
docker compose -f docker-compose.silent.yaml ps
```

### 查看日志

```bash
docker compose logs -f vrl-face
docker compose logs -f vrl-liveness
docker compose -f docker-compose.silent.yaml logs -f vrl-silent
```

## 三环境对比

| | 开发 (`dev/`) | 测试 (`tra/`) | 生产 (`pro/`) |
|---|---|---|---|
| Workers | 默认(1) | 1 | 2 |
| 日志级别 | 默认(info) | debug | info |
| 热重载 | ✅ `--reload` | ❌ | ❌ |
| 代码挂载 | ✅ 宿主机代码 | ❌ | ❌ |
| 自动重启 | ❌ | ✅ `unless-stopped` | ✅ `unless-stopped` |
| 内存限制 | 4G | 4G | 8G |
| 日志轮转 | ❌ | ✅ 10m×3 | ✅ 10m×3 |
| 构建方式 | `up -d` | `up -d` | `up -d --build` |
| 健康检查启动等待 | 60s (face/liveness), 120s (silent) | 同左 | 同左 |

## 健康检查

```bash
curl http://localhost:8070/healthz   # 人脸识别
curl http://localhost:8071/healthz   # 动作活体
curl http://localhost:8060/healthz   # 静默活体
```

正常响应：

```json
{"status": "healthy", "service": "face", "version": "3.1.0"}
```

## API 快速验证

### 人脸检测

```bash
curl -X POST http://localhost:8070/vrlFaceDetection \
  -F "file=@test.jpg"
```

### 动作活体检测

```bash
curl -X POST http://localhost:8071/vrlMoveLiveness \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "task_id": "task-001",
    "video_path": "/data/videos/test.mp4",
    "actions": ["blink"]
  }'
```

### 静默活体检测

```bash
curl -X POST http://localhost:8060/vrlSilentLiveness \
  -H "Content-Type: application/json" \
  -d '{"picture_path": "/data/videos/test.jpg"}'
```

**多路径支持**（远程服务器调用）：

```bash
# 远程服务器路径 /opt/test -> 容器内 /data/videos
curl -X POST http://localhost:8060/vrlSilentLiveness \
  -H "Content-Type: application/json" \
  -d '{"picture_path": "/opt/test/test.jpg"}'

# 远程服务器路径 /opt/test2026 -> 容器内 /data/videos
curl -X POST http://localhost:8060/vrlSilentLiveness \
  -H "Content-Type: application/json" \
  -d '{"picture_path": "/opt/test2026/test.jpg"}'
```

> **配置说明**: 路径映射通过环境变量 `SILENT_PICTURE_PATHS` 配置，见下文"静默活体多路径配置"。

响应示例：

```json
{
  "code": 0,
  "msg": "silent liveness checking successful",
  "liveness_results": {
    "is_liveness": 1,
    "confidence": 0.5886,
    "is_face_exist": 1,
    "face_exist_confidence": 0.95
  },
  "filename": "/data/videos/test.jpg"
}
```

## Swagger 文档

启动后访问各服务的交互式 API 文档：

- 人脸识别：http://localhost:8070/docs
- 动作活体：http://localhost:8071/docs
- 静默活体：http://localhost:8060/docs

## 卷挂载说明

| 路径 | 用途 | 使用服务 |
|------|------|---------|
| `models/` | InsightFace + MediaPipe 模型 | face, liveness |
| `data/` | 人脸库数据集 | face |
| `/data/videos` | 视频/图片 文件（宿主机路径） | liveness, silent |

---

## 静默活体多路径配置

静默活体检测支持**多路径映射**，允许不同环境的客户端使用各自熟悉的路径调用 API。

### 使用场景

- **本地开发**: 客户端传入 `/data/videos/test.jpg`
- **远程服务器**: 客户端传入 `/opt/test/test.jpg` 或 `/opt/test2026/test.jpg`
- **API 自动映射**: 所有路径统一映射到容器内 `/data/videos/test.jpg`

### 环境变量配置

在 `deploy/{dev|tra|pro}/.env` 中配置：

```bash
# 路径映射：外部路径=内部路径，多个用分号分隔
SILENT_PICTURE_PATHS=/opt/test=/data/videos;/opt/test2026=/data/videos

# 允许的路径前缀（安全限制）
SILENT_ALLOWED_PATH_PREFIXES=/data/videos,/opt/test,/opt/test2026
```

### 工作原理

```
客户端请求                        API 处理                        容器内文件
────────────                     ────────────                    ────────────
picture_path=/opt/test/a.jpg  →  路径映射                     →  /data/videos/a.jpg
                                 前缀校验
                                 文件存在检查
```

### 安全限制

- 只允许配置了前缀的路径被访问（防止路径遍历攻击）
- 未授权路径返回 400 错误：`不允许的路径前缀：/etc/passwd`

### 新增路径映射

如需支持新的远程路径，例如 `/remote/data`：

```bash
# 1. 编辑 .env
SILENT_PICTURE_PATHS=/opt/test=/data/videos;/opt/test2026=/data/videos;/remote/data=/data/videos
SILENT_ALLOWED_PATH_PREFIXES=/data/videos,/opt/test,/opt/test2026,/remote/data

# 2. 重启服务
cd deploy/pro && bash down.sh silent && bash up.sh silent
```

### 测试

```bash
# 使用测试脚本验证配置
uv run python scripts/manual_tests/test_silent_multipath.py
```

## 项目结构（部署相关）

```
.
├── Dockerfile               # 人脸识别 + 动作活体镜像
├── Dockerfile.silent         # 静默活体镜像（TensorFlow 隔离）
├── requirements.txt          # 人脸 + 活体依赖
├── requirements-silent.txt   # 静默活体依赖
│
├── vrlFace/
│   ├── main_fastapi.py       # 合并服务入口（单端口）
│   ├── apps/                 # 独立服务入口
│   │   ├── face_app.py       # → :8070
│   │   ├── liveness_app.py   # → :8071
│   │   └── silent_app.py     # → :8060
│   ├── face/                 # 人脸识别模块
│   ├── liveness/             # 动作活体模块
│   └── silent_liveness/      # 静默活体模块
│
├── deploy/
│   ├── .env.example
│   ├── dev/                  # 开发环境
│   │   ├── docker-compose.yaml          # face + liveness
│   │   ├── docker-compose.face.yaml     # 仅 face
│   │   ├── docker-compose.liveness.yaml # 仅 liveness
│   │   ├── docker-compose.silent.yaml   # 仅 silent
│   │   ├── up.sh / down.sh
│   │   └── .env
│   ├── tra/                  # 测试环境（同结构）
│   └── pro/                  # 生产环境（同结构）
│
├── models/                   # 模型文件
│   └── buffalo_l/            # InsightFace 模型
└── data/                     # 人脸库
```

## 常见问题

### 静默活体首次启动慢

`deepface-antispoofing` 首次运行时自动下载模型（~100MB）。确保容器可访问外网，或提前在镜像中预下载。`start_period` 已设为 120s 以适应此延迟。

### 端口冲突

```bash
# 检查端口占用
ss -tlnp | grep -E '8060|8070|8071'
```

### 内存不足（OOM）

单服务建议至少 4GB。全量部署建议 16GB+。可在 docker-compose.yaml 中调整 `mem_limit`。

### 模型文件缺失

```
ERROR: models/face_landmarker.task 不存在
```

需手动下载 MediaPipe face_landmarker 模型放入 `models/` 目录。

### 查看容器内日志

```bash
docker compose exec vrl-face cat /app/logs/face.log
docker compose -f docker-compose.silent.yaml exec vrl-silent bash
```
