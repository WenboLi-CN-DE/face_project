# 服务器部署操作手册

> 适用环境：Ubuntu 20.04 / 22.04 LTS（推荐）
> 部署方式：Docker Compose（推荐）/ 裸机 systemd 两选一
> 服务端口：8070
> 接口文档：`http://<IP>:8070/docs`

---

## 目录

1. [前置要求](#1-前置要求)
2. [上传代码](#2-上传代码)
3. [方案 A：Docker 部署（推荐）](#3-方案-a-docker-部署推荐)
4. [方案 B：裸机部署](#4-方案-b-裸机部署)
5. [Nginx 反向代理](#5-nginx-反向代理配置)
6. [模型文件处理](#6-模型文件处理)
7. [验证服务](#7-验证服务)
8. [日常运维](#8-日常运维)
9. [故障排查](#9-故障排查)

---

## 1. 前置要求

### 服务器最低配置

| 项目 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 20 GB | 50 GB |
| 系统 | Ubuntu 20.04 | Ubuntu 22.04 LTS |
| 网络 | 公网 IP | 公网 IP + 域名 |

### 本地准备清单

- [ ] 服务器 SSH 登录凭证
- [ ] `models/buffalo_l/` 模型文件（约 300 MB）
- [ ] `models/face_landmarker.task` MediaPipe 模型（约 5 MB）
- [ ] 私有镜像仓库地址（可选）

---

## 2. 上传代码

### 方式 A：Git 克隆（推荐）

```bash
# 服务器上执行
sudo mkdir -p /opt/face_cls
sudo chown $USER:$USER /opt/face_cls
cd /opt/face_cls
git clone <your-repo-url> .
```

### 方式 B：scp 直接上传

```bash
# 本地执行 — 打包（排除无用目录）
cd E:\unified_pyprj
tar -czf face_cls.tar.gz \
    --exclude='face_cls/_bmad' \
    --exclude='face_cls/_bmad-output' \
    --exclude='face_cls/__pycache__' \
    --exclude='face_cls/*.pyc' \
    face_cls

# 上传到服务器
scp face_cls.tar.gz user@<server-ip>:/opt/

# 服务器上解压
ssh user@<server-ip>
sudo mkdir -p /opt/face_cls
sudo tar -xzf /opt/face_cls.tar.gz -C /opt/face_cls --strip-components=1
```

---

## 3. 方案 A：Docker 部署（推荐）

### 3.1 安装 Docker

```bash
# 一键安装 Docker + Docker Compose
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker          # 使权限立即生效，无需重新登录

# 验证
docker --version
docker compose version
```

### 3.2 上传模型文件

```bash
# 确保模型目录结构
cd /opt/face_cls
mkdir -p models/buffalo_l

# 从本地上传模型（本地执行）
scp -r E:\unified_pyprj\face_cls\models\buffalo_l user@<server-ip>:/opt/face_cls/models/
scp E:\unified_pyprj\face_cls\models\face_landmarker.task user@<server-ip>:/opt/face_cls/models/

# 验证（服务器执行）
ls -lh /opt/face_cls/models/buffalo_l/
ls -lh /opt/face_cls/models/face_landmarker.task
```

### 3.3 配置环境变量

```bash
cd /opt/face_cls/deploy/pro

# 从模板复制环境变量文件
cp .env.example .env

# 编辑配置（按需修改）
nano .env
```

`.env` 最终内容参考：

```dotenv
FACE_MODEL_NAME=buffalo_l
FACE_DET_SIZE=640,640
FACE_GPU_ID=-1
FACE_THRESHOLD=0.55
FACE_IMAGES_BASE=/app/data/dataset
APP_PORT=8070
APP_WORKERS=2
LOG_LEVEL=INFO
```

### 3.4 构建并启动

```bash
cd /opt/face_cls/deploy/pro

# 构建镜像（首次约 5~10 分钟）
docker compose build

# 启动服务（后台运行）
docker compose up -d

# 查看启动日志
docker compose logs -f --tail=50
```

### 3.5 验证服务正常

```bash
# 等待约 30 秒后检查
curl http://localhost:8070/healthz
# 预期返回：{"status":"healthy","version":"3.1.0"}

curl http://localhost:8070/docs
# 预期：FastAPI Swagger UI HTML
```

---

## 4. 方案 B：裸机部署

### 4.1 安装系统依赖

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    python3.10 python3.10-venv python3-pip \
    git curl wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libusb-1.0-0 \
    nginx
```

### 4.2 创建应用用户和目录

```bash
sudo useradd -m -s /bin/bash deploy
sudo mkdir -p /opt/face_cls
sudo chown deploy:deploy /opt/face_cls
```

### 4.3 创建虚拟环境并安装依赖

```bash
sudo -u deploy bash << 'EOF'
cd /opt/face_cls
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
EOF
```

### 4.4 上传模型文件

```bash
sudo mkdir -p /opt/face_cls/models/buffalo_l
# 本地上传（同 3.2）
```

### 4.5 配置环境变量

```bash
sudo tee /opt/face_cls/.env > /dev/null << 'EOF'
FACE_MODEL_NAME=buffalo_l
FACE_DET_SIZE=640,640
FACE_GPU_ID=-1
FACE_THRESHOLD=0.55
FACE_IMAGES_BASE=/opt/face_cls/data/dataset
APP_PORT=8070
APP_WORKERS=2
LOG_LEVEL=INFO
EOF
sudo chown deploy:deploy /opt/face_cls/.env
```

### 4.6 配置 systemd 服务

```bash
sudo tee /etc/systemd/system/face_cls.service > /dev/null << 'EOF'
[Unit]
Description=Face & Liveness API
After=network.target

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/face_cls
EnvironmentFile=/opt/face_cls/.env
ExecStart=/opt/face_cls/venv/bin/uvicorn vrlFace.main_fastapi:app \
    --host 0.0.0.0 \
    --port 8070 \
    --workers 2 \
    --log-level info
Restart=always
RestartSec=10
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal
SyslogIdentifier=face_cls

[Install]
WantedBy=multi-user.target
EOF

# 启动并设为开机自启
sudo systemctl daemon-reload
sudo systemctl enable face_cls
sudo systemctl start face_cls

# 查看状态
sudo systemctl status face_cls
sudo journalctl -u face_cls -f
```

---

## 5. Nginx 反向代理配置

```bash
sudo tee /etc/nginx/sites-available/face_cls > /dev/null << 'EOF'
upstream face_cls_backend {
    server 127.0.0.1:8070;
    keepalive 32;
}

server {
    listen 80;
    server_name _;                     # 替换为你的域名，如 api.example.com

    access_log /var/log/nginx/face_cls_access.log;
    error_log  /var/log/nginx/face_cls_error.log;

    # 上传文件大小限制（视频文件可能较大）
    client_max_body_size 200M;

    location / {
        proxy_pass         http://face_cls_backend;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # 视频分析可能耗时较长
        proxy_connect_timeout 60s;
        proxy_send_timeout    300s;
        proxy_read_timeout    300s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/face_cls /etc/nginx/sites-enabled/face_cls
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t                 # 测试配置
sudo systemctl restart nginx
```

### 配置 HTTPS（有域名时）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.example.com
# 按提示完成，证书自动续期
```

---

## 6. 模型文件处理

### 目录结构要求

```
/opt/face_cls/models/
├── face_landmarker.task          # MediaPipe 活体模型，约 5 MB
└── buffalo_l/                    # InsightFace 人脸识别模型，约 300 MB
    ├── 1k3d68.onnx
    ├── 2d106det.onnx
    ├── det_10g.onnx
    ├── genderage.onnx
    └── w600k_r50.onnx
```

### 模型路径说明

| 模型 | 默认路径 | 用途 |
|------|----------|------|
| InsightFace `buffalo_l` | `~/.insightface/models/buffalo_l/`（自动下载） | 人脸识别/检测 |
| MediaPipe `face_landmarker.task` | `models/face_landmarker.task` | 活体检测关键点 |

> **注意**：InsightFace 首次运行时会自动从网络下载 `buffalo_l`（需要外网）。
> 如果服务器无法访问外网，需手动上传到 `~/.insightface/models/buffalo_l/`。

### 离线部署模型（无外网时）

```bash
# 在有外网的机器上预下载
python -c "
import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640,640))
print('模型已下载到:', app.root)
"

# 打包并上传
tar -czf buffalo_l.tar.gz ~/.insightface/models/buffalo_l
scp buffalo_l.tar.gz user@<server-ip>:~/

# 服务器上解压
mkdir -p ~/.insightface/models
tar -xzf ~/buffalo_l.tar.gz -C ~/.insightface/models/
```

---

## 7. 验证服务

### 7.1 健康检查

```bash
curl -s http://localhost:8070/healthz | python3 -m json.tool
# 期望：{"status": "healthy", "version": "3.1.0"}
```

### 7.2 人脸检测接口测试

```bash
# 准备一张测试图片
curl -X POST http://localhost:8070/vrlFaceDetection \
  -F "picture=@/path/to/test.jpg" \
  | python3 -m json.tool
```

### 7.3 活体检测接口测试

```bash
# 准备一段测试视频（mp4 格式）
# 注意：服务器路径，视频需在服务器上
curl -X POST http://localhost:8070/vrlMoveLiveness \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "task_id":    "task_001",
    "video_path": "/opt/face_cls/data/test_video.mp4",
    "actions":    ["blink", "mouth_open", "shake_head"],
    "threshold_config": {
      "liveness_threshold": 0.5,
      "action_threshold":   0.85
    },
    "action_config": {
      "max_video_duration": 6,
      "per_action_timeout": 2
    }
  }' | python3 -m json.tool
```

期望响应：

```json
{
  "code": 0,
  "msg": "success",
  "request_id": "test_001",
  "task_id": "task_001",
  "data": {
    "is_liveness": 1,
    "liveness_confidence": 0.87,
    "is_face_exist": 1,
    "face_info": {
      "confidence": 0.92,
      "quality_score": 0.85
    },
    "action_verify": {
      "passed": true,
      "required_actions": ["blink", "mouth_open", "shake_head"],
      "action_details": [
        {"action": "blink",      "passed": true,  "confidence": 0.91, "msg": "检测到有效眨眼"},
        {"action": "mouth_open", "passed": true,  "confidence": 0.88, "msg": "检测到有效张嘴"},
        {"action": "shake_head", "passed": true,  "confidence": 0.86, "msg": "检测到有效摇头"}
      ]
    }
  }
}
```

### 7.4 接口文档

浏览器访问：`http://<server-ip>:8070/docs`，所有接口可在线测试。

---

## 8. 日常运维

### Docker 方式

```bash
cd /opt/face_cls/deploy/pro

# 查看服务状态
docker compose ps

# 查看实时日志
docker compose logs -f --tail=100

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 更新代码后重新构建部署
git pull
docker compose build
docker compose up -d
```

### systemd 方式

```bash
# 查看状态
sudo systemctl status face_cls

# 实时日志
sudo journalctl -u face_cls -f

# 重启
sudo systemctl restart face_cls

# 停止
sudo systemctl stop face_cls

# 更新后重启
cd /opt/face_cls && git pull
sudo systemctl restart face_cls
```

### 防火墙

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8070/tcp  # 人脸识别服务
sudo ufw allow 8071/tcp  # 活体检测服务
sudo ufw --force enable
sudo ufw status
```

---

## 9. 故障排查

### 服务启动失败

```bash
# Docker
docker compose logs face_cls 2>&1 | tail -50

# systemd
sudo journalctl -u face_cls -n 50 --no-pager
```

### 常见错误及解决

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `ModuleNotFoundError: mediapipe` | mediapipe 未安装 | `pip install mediapipe>=0.10.0` |
| `Cannot find model buffalo_l` | InsightFace 模型未下载 | 参考 [第 6 节离线部署](#离线部署模型无外网时) |
| `OSError: face_landmarker.task` | MediaPipe 模型文件缺失 | 上传 `models/face_landmarker.task` |
| `Address already in use :8070` | 端口被占用 | `sudo lsof -i:8070` 查找进程并停止 |
| `Out of memory` | 内存不足 | 增加服务器内存或减少 `APP_WORKERS` 为 1 |
| `video_path 不存在` | 路径不对 | 确认视频已上传到服务器，使用服务器绝对路径 |

### 内存不足时降配

编辑 `deploy/pro/.env`：

```dotenv
APP_WORKERS=1     # 从 2 减为 1
```

然后重启服务。

### 查看接口响应时间

```bash
curl -o /dev/null -s -w "Total: %{time_total}s\n" \
  http://localhost:8070/healthz
```

---

## 附录：接口速查表

| 接口路径 | 方法 | 说明 | 请求格式 |
|----------|------|------|----------|
| `/healthz` | GET | 健康检查 | — |
| `/vrlFaceDetection` | POST | 人脸检测 | multipart/form-data |
| `/vrlFaceComparison` | POST | 1:1 人脸比对 | multipart/form-data |
| `/vrlFaceSearch` | POST | 1:N 搜索 | multipart/form-data |
| `/vrlMoveLiveness` | POST | 视频活体检测 | application/json |
| `/docs` | GET | Swagger 交互文档 | — |

### `vrlMoveLiveness` 参数说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `request_id` | string | ✅ | 调用方唯一请求 ID |
| `task_id` | string | ✅ | 任务 ID |
| `video_path` | string | ✅ | 视频文件在**服务器**上的绝对路径 |
| `actions` | list | ✅ | 动作列表，见下表 |
| `threshold_config.liveness_threshold` | float | ❌ | 活体阈值，默认 0.5 |
| `threshold_config.action_threshold` | float | ❌ | 动作置信度阈值，默认 0.85 |
| `action_config.max_video_duration` | float | ❌ | 最大分析时长（秒） |
| `action_config.per_action_timeout` | float | ❌ | 每动作时间窗口（秒） |

### 支持的动作列表

| action 值 | 说明 |
|-----------|------|
| `blink` | 眨眼 |
| `mouth_open` | 张嘴 |
| `shake_head` | 摇头（左右） |
| `nod` | 点头（上下） |
| `nod_down` | 低头 |
| `nod_up` | 抬头 |
| `turn_left` | 向左转头 |
| `turn_right` | 向右转头 |

