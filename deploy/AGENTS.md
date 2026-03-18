# deploy — Docker 与裸机部署

**位置**: `deploy/`  
**职责**: Docker 三环境部署、裸机部署、服务编排

## 结构

```
deploy/
├── .env.example              # 环境变量模板
│
├── dev/                      # 开发环境
│   ├── .env                  # 开发环境变量
│   ├── docker-compose.yaml   # Docker Compose 配置
│   ├── docker-compose.face.yaml      # 人脸服务配置
│   ├── docker-compose.liveness.yaml  # 活体服务配置
│   ├── up.sh                 # 启动脚本
│   └── down.sh               # 停止脚本
│
├── tra/                      # 测试环境
│   ├── .env                  # 测试环境变量
│   ├── docker-compose.yaml
│   ├── docker-compose.face.yaml
│   ├── docker-compose.liveness.yaml
│   ├── up.sh
│   └── down.sh
│
├── pro/                      # 生产环境
│   ├── .env                  # 生产环境变量（需手动创建）
│   ├── docker-compose.yaml
│   ├── docker-compose.face.yaml
│   ├── docker-compose.liveness.yaml
│   ├── up.sh
│   └── down.sh
│
└── scripts/
    ├── deploy_bare.sh        # Ubuntu 裸机部署脚本
    └── install_dev_deps.bat  # Windows 开发依赖安装
```

## Docker 三环境对比

| 环境 | Workers | 日志级别 | 内存限制 | 特性 |
|------|---------|----------|----------|------|
| `dev` | 默认 | 默认 | 4G | 热重载，代码挂载 |
| `tra` | 1 | debug | 4G | 日志轮转，unless-stopped |
| `pro` | 2 | info | 8G | 持久化模型卷，外部视频路径 |

## 快速开始

### 开发环境
```bash
cd deploy/dev
bash up.sh      # 启动所有服务
bash down.sh    # 停止所有服务
```

### 测试环境
```bash
cd deploy/tra
bash up.sh
bash down.sh
```

### 生产环境
```bash
cd deploy/pro
cp .env.example .env
# 编辑 .env 填写实际配置
bash up.sh
bash down.sh
```

## 服务分离部署

### 仅部署人脸服务
```bash
bash up.sh face
```

### 仅部署活体服务
```bash
bash up.sh liveness
```

### 部署所有服务
```bash
bash up.sh all
```

## 端口配置

| 服务 | 端口 | 说明 |
|------|------|------|
| 人脸识别 API | 8070 | FastAPI 服务 |
| 活体检测 API | 8071 | FastAPI 服务 |
| 健康检查 | 8070/healthz, 8071/healthz | GET 请求 |

## Docker Compose 配置

### 开发环境特点
- **热重载**: 代码挂载，修改即时生效
- **调试友好**: 完整日志输出
- **资源限制**: 4G 内存上限

### 生产环境特点
- **多 Worker**: 2 个 Gunicorn Worker 并发
- **日志轮转**: 自动清理旧日志
- **持久化**: 模型卷持久化，避免重复下载
- **资源优化**: 8G 内存，CPU 优先调度

## 环境变量

### 通用变量（所有环境）
```bash
# 人脸识别
FACE_MODEL_NAME=buffalo_l
FACE_DET_SIZE=640,640
FACE_GPU_ID=-1
FACE_THRESHOLD=0.55
FACE_IMAGES_BASE=/app/data/dataset

# 活体检测
LIVENESS_CALLBACK_URL=http://localhost:8092/api/v1/callbacks
LIVENESS_CALLBACK_SECRET=your-secret-key
LIVENESS_CALLBACK_TIMEOUT=10
LIVENESS_CALLBACK_MAX_RETRIES=3
```

### 生产环境专属
```bash
# 安全配置
SECRET_KEY=your-production-secret-key
DEBUG=false

# 性能配置
GUNICORN_WORKERS=2
LOG_LEVEL=info
```

## 裸机部署

### Ubuntu 20.04/22.04 部署

**脚本**: `deploy/scripts/deploy_bare.sh`

**部署目标**:
- systemd 服务管理
- Nginx 反向代理
- Python 虚拟环境
- 自动重启

**运行方式**:
```bash
# 在目标 Ubuntu 服务器上
bash deploy/scripts/deploy_bare.sh
```

**部署后**:
- 服务路径：`/opt/vrlface/`
- 日志路径：`/var/log/vrlface/`
- systemd 服务：`vrlface-face.service`, `vrlface-liveness.service`

## Windows 开发依赖

**脚本**: `deploy/scripts/install_dev_deps.bat`

**安装内容**:
- Python 3.9+
- Git
- Docker Desktop
- UV 包管理器

**运行方式**:
```cmd
deploy\scripts\install_dev_deps.bat
```

## 日志管理

### 查看日志
```bash
# Docker 环境
docker-compose logs -f face
docker-compose logs -f liveness

# 生产环境（日志轮转）
journalctl -u vrlface-face -f
journalctl -u vrlface-liveness -f
```

### 日志位置
| 环境 | 日志路径 |
|------|----------|
| dev | Docker 标准输出 |
| tra | Docker + 日志文件 |
| pro | `/var/log/vrlface/` |
| 裸机 | `/var/log/vrlface/*.log` |

## 健康检查

```bash
# 人脸服务
curl http://localhost:8070/healthz

# 活体服务
curl http://localhost:8071/healthz
```

**响应示例**:
```json
{"status": "healthy", "service": "face", "timestamp": "2026-03-16T10:30:00"}
```

## 故障排查

### 服务无法启动
```bash
# 查看详细日志
docker-compose logs face
docker-compose logs liveness

# 检查端口占用
netstat -tlnp | grep 8070
netstat -tlnp | grep 8071
```

### 内存不足
```bash
# 调整内存限制（编辑 docker-compose.yaml）
deploy:
  resources:
    limits:
      memory: 8G
```

### 模型下载失败
```bash
# 手动下载模型
mkdir -p models/buffalo_l
# 从镜像或本地复制模型文件
```

## 更新部署

### 开发环境（热重载）
代码修改自动生效，无需重启。

### 生产环境
```bash
cd deploy/pro
bash down.sh
git pull
bash up.sh
```

### 滚动更新（零停机）
```bash
# 先启动新版本
docker-compose up -d face --scale face=2

# 验证健康
curl http://localhost:8070/healthz

# 停止旧版本
docker-compose up -d face --scale face=1

# 最终切换
docker-compose up -d
```

## 安全建议

- ❌ 不要将 `.env` 文件提交到 Git
- ✅ 生产环境使用强密码和 HTTPS
- ✅ 定期更新依赖和基础镜像
- ✅ 限制 Docker 容器网络访问
- ✅ 使用 secrets 管理敏感信息

## 相关文档

- `ARCHITECTURE.md` — 系统架构说明
- `README.md` — 项目总览
- `docs/docker-log-workflow.md` — Docker 日志工作流
