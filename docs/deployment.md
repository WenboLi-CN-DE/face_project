# 服务器部署操作手册

> 适用环境：Ubuntu 20.04 / 22.04 LTS（推荐）
> 部署方式：Docker Compose（推荐）/ 裸机 systemd 两选一
> 服务架构：**两个独立服务**
> &nbsp;&nbsp;&nbsp;&nbsp;`vrl-face`     — 人脸识别，端口 **8070**
> &nbsp;&nbsp;&nbsp;&nbsp;`vrl-liveness` — 活体检测，端口 **8071**

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

### 服务架构说明

部署完成后共有 **2 个容器、2 个端口**：

```
docker ps 输出示例：

CONTAINER ID   IMAGE                   PORTS                    NAMES
a1b2c3d4e5f6   face_cls-vrl-face       0.0.0.0:8070->8070/tcp   face_cls-vrl-face-1
b7c8d9e0f1a2   face_cls-vrl-liveness   0.0.0.0:8071->8071/tcp   face_cls-vrl-liveness-1
```

| 容器名 | 入口模块 | 端口 | 接口 |
|--------|----------|------|------|
| `vrl-face` | `vrlFace.face_app` | **8070** | `/vrlFaceDetection` `/vrlFaceComparison` `/vrlFaceSearch` |
| `vrl-liveness` | `vrlFace.liveness_app` | **8071** | `/vrlMoveLiveness` |

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
cd /your/local/path
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
# 服务器上确保目录存在
mkdir -p /opt/face_cls/models/buffalo_l

# 本地执行 — 上传模型
scp -r ./models/buffalo_l         user@<server-ip>:/opt/face_cls/models/
scp    ./models/face_landmarker.task user@<server-ip>:/opt/face_cls/models/

# 创建视频目录（活体检测视频存放位置）
ssh user@<server-ip> "sudo mkdir -p /data/videos && sudo chmod 777 /data/videos"

# 服务器验证
ls -lh /opt/face_cls/models/buffalo_l/
ls -lh /opt/face_cls/models/face_landmarker.task
```

### 3.3 配置环境变量

```bash
cd /opt/face_cls/deploy/pro

# 从模板复制
cp .env.example .env

# 按需编辑（默认值可直接使用）
nano .env
```

`.env` 各字段说明：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FACE_MODEL_NAME` | `buffalo_l` | InsightFace 模型名称 |
| `FACE_DET_SIZE` | `640,640` | 人脸检测分辨率 |
| `FACE_GPU_ID` | `-1` | GPU ID，`-1` 表示使用 CPU |
| `FACE_THRESHOLD` | `0.55` | 人脸相似度阈值 |
| `FACE_IMAGES_BASE` | `/app/data/dataset` | 人脸库路径（容器内） |
| `APP_PORT` | `8070` | 参考用，实际端口由命令行指定 |
| `APP_WORKERS` | `2` | uvicorn Worker 进程数 |
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` |
| `LIVENESS_THRESHOLD` | `0.5` | 全局活体阈值（可在请求中覆盖） |
| `ACTION_THRESHOLD` | `0.85` | 动作置信度阈值（可在请求中覆盖） |
| `VIDEO_BASE_DIR` | `/app/videos` | 视频目录（容器内挂载路径） |

完整 `.env` 示例：

```dotenv
FACE_MODEL_NAME=buffalo_l
FACE_DET_SIZE=640,640
FACE_GPU_ID=-1
FACE_THRESHOLD=0.55
FACE_IMAGES_BASE=/app/data/dataset
APP_PORT=8070
APP_WORKERS=2
LOG_LEVEL=INFO
LIVENESS_THRESHOLD=0.5
ACTION_THRESHOLD=0.85
VIDEO_BASE_DIR=/app/videos
```

### 3.4 构建并启动

每个环境目录下有三个 compose 文件，支持**整体部署**或**单独部署**任意一个服务：

| 文件 | 说明 |
|------|------|
| `docker-compose.yaml` | 两个服务一起启动/停止 |
| `docker-compose.face.yaml` | 仅人脸识别服务（8070） |
| `docker-compose.liveness.yaml` | 仅活体检测服务（8071） |

```bash
cd /opt/face_cls/deploy/pro

# ── 方式一：使用 up.sh 脚本（推荐）──────────────────────────
bash up.sh              # 同时启动两个服务（默认）
bash up.sh face         # 只启动人脸识别服务 (8070)
bash up.sh liveness     # 只启动活体检测服务 (8071)

# ── 方式二：直接使用 docker compose 命令 ────────────────────
# 两个服务一起
docker compose build && docker compose up -d

# 只启动人脸识别
docker compose -f docker-compose.face.yaml build
docker compose -f docker-compose.face.yaml up -d

# 只启动活体检测
docker compose -f docker-compose.liveness.yaml build
docker compose -f docker-compose.liveness.yaml up -d
```

停止服务：

```bash
bash down.sh              # 停止全部
bash down.sh face         # 只停止人脸识别
bash down.sh liveness     # 只停止活体检测
```

### 3.5 一键部署（使用 up.sh）

```bash
cd /opt/face_cls/deploy/pro
bash up.sh
```

脚本会自动完成：检查 `.env` → 检查模型文件 → 创建视频目录 → 构建镜像 → 启动服务 → 等待两个服务均健康检查通过。

成功输出示例：

```
✅ vrl-face 健康检查通过
{"status":"healthy","service":"face","version":"3.1.0"}

✅ vrl-liveness 健康检查通过
{"status":"healthy","service":"liveness","version":"3.1.0"}

📖 人脸识别文档：http://localhost:8070/docs
📖 活体检测文档：http://localhost:8071/docs
```

---

## 4. 方案 B：裸机部署

> 裸机部署需要手动配置两个 systemd 服务，分别对应人脸识别和活体检测。

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
sudo mkdir -p /data/videos && sudo chmod 777 /data/videos
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

### 4.4 配置环境变量

```bash
sudo tee /opt/face_cls/.env > /dev/null << 'EOF'
FACE_MODEL_NAME=buffalo_l
FACE_DET_SIZE=640,640
FACE_GPU_ID=-1
FACE_THRESHOLD=0.55
FACE_IMAGES_BASE=/opt/face_cls/data/dataset
APP_WORKERS=2
LOG_LEVEL=INFO
LIVENESS_THRESHOLD=0.5
ACTION_THRESHOLD=0.85
VIDEO_BASE_DIR=/data/videos
EOF
sudo chown deploy:deploy /opt/face_cls/.env
```

### 4.5 配置 systemd — 人脸识别服务（8070）

```bash
sudo tee /etc/systemd/system/vrl-face.service > /dev/null << 'EOF'
[Unit]
Description=VRL Face Recognition API
After=network.target

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/face_cls
EnvironmentFile=/opt/face_cls/.env
ExecStart=/opt/face_cls/venv/bin/uvicorn vrlFace.face_app:app \
    --host 0.0.0.0 \
    --port 8070 \
    --workers 2 \
    --log-level info
Restart=always
RestartSec=10
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vrl-face

[Install]
WantedBy=multi-user.target
EOF
```

### 4.6 配置 systemd — 活体检测服务（8071）

```bash
sudo tee /etc/systemd/system/vrl-liveness.service > /dev/null << 'EOF'
[Unit]
Description=VRL Liveness Detection API
After=network.target

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/face_cls
EnvironmentFile=/opt/face_cls/.env
ExecStart=/opt/face_cls/venv/bin/uvicorn vrlFace.liveness_app:app \
    --host 0.0.0.0 \
    --port 8071 \
    --workers 2 \
    --log-level info
Restart=always
RestartSec=10
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vrl-liveness

[Install]
WantedBy=multi-user.target
EOF
```

### 4.7 启动两个服务

```bash
sudo systemctl daemon-reload

# 人脸识别服务
sudo systemctl enable vrl-face
sudo systemctl start vrl-face

# 活体检测服务
sudo systemctl enable vrl-liveness
sudo systemctl start vrl-liveness

# 查看状态
sudo systemctl status vrl-face
sudo systemctl status vrl-liveness
```

---

## 5. Nginx 反向代理配置

两个服务各自配置独立的 upstream，通过不同路径或端口对外暴露。

```bash
sudo tee /etc/nginx/sites-available/vrl-face > /dev/null << 'EOF'
upstream vrl_face_backend {
    server 127.0.0.1:8070;
    keepalive 32;
}

upstream vrl_liveness_backend {
    server 127.0.0.1:8071;
    keepalive 32;
}

# 人脸识别服务（80端口，/face/ 路径）
server {
    listen 80;
    server_name face.example.com;      # 替换为实际域名

    access_log /var/log/nginx/vrl_face_access.log;
    error_log  /var/log/nginx/vrl_face_error.log;

    client_max_body_size 50M;

    location / {
        proxy_pass         http://vrl_face_backend;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout    120s;
        proxy_read_timeout    120s;
    }
}

# 活体检测服务
server {
    listen 80;
    server_name liveness.example.com;  # 替换为实际域名

    access_log /var/log/nginx/vrl_liveness_access.log;
    error_log  /var/log/nginx/vrl_liveness_error.log;

    # 视频文件较大，适当放宽限制
    client_max_body_size 200M;

    location / {
        proxy_pass         http://vrl_liveness_backend;
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

sudo ln -sf /etc/nginx/sites-available/vrl-face /etc/nginx/sites-enabled/vrl-face
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 配置 HTTPS（有域名时）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d face.example.com -d liveness.example.com
```

---

## 6. 模型文件处理

### 目录结构要求

```
/opt/face_cls/models/
├── face_landmarker.task          # MediaPipe 活体检测模型，约 5 MB（vrl-liveness 使用）
└── buffalo_l/                    # InsightFace 人脸识别模型，约 300 MB（vrl-face 使用）
    ├── 1k3d68.onnx
    ├── 2d106det.onnx
    ├── det_10g.onnx
    ├── genderage.onnx
    └── w600k_r50.onnx
```

### 模型与服务对应关系

| 模型 | 使用服务 | 容器内路径 |
|------|----------|-----------|
| `buffalo_l/` | `vrl-face`（8070） | `/app/models/buffalo_l/` |
| `face_landmarker.task` | `vrl-liveness`（8071） | `/app/models/face_landmarker.task` |

> **注意**：InsightFace 首次运行时会自动从网络下载 `buffalo_l`（需要外网）。
> 如果服务器无法访问外网，需手动上传，参考下方离线部署。

### 离线部署模型（无外网时）

```bash
# 在有外网的机器上预下载
python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640,640))
print('模型路径:', app.root)
"

# 打包并上传（本地执行）
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
# 人脸识别服务
curl -s http://localhost:8070/healthz | python3 -m json.tool
# 期望：{"status": "healthy", "service": "face", "version": "3.1.0"}

# 活体检测服务
curl -s http://localhost:8071/healthz | python3 -m json.tool
# 期望：{"status": "healthy", "service": "liveness", "version": "3.1.0"}
```

### 7.2 docker ps 验证

```bash
docker compose ps
```

期望输出（两个容器均为 healthy）：

```
NAME                      IMAGE                   PORTS                    STATUS
face_cls-vrl-face-1       face_cls-vrl-face       0.0.0.0:8070->8070/tcp   Up X minutes (healthy)
face_cls-vrl-liveness-1   face_cls-vrl-liveness   0.0.0.0:8071->8071/tcp   Up X minutes (healthy)
```

### 7.3 人脸检测接口测试

```bash
curl -X POST http://localhost:8070/vrlFaceDetection \
  -F "picture=@/path/to/test.jpg" \
  | python3 -m json.tool
```

### 7.4 人脸比对接口测试

```bash
curl -X POST http://localhost:8070/vrlFaceComparison \
  -F "picture1=@/path/to/person1.jpg" \
  -F "picture2=@/path/to/person2.jpg" \
  | python3 -m json.tool
```

### 7.5 活体检测接口测试

> **注意**：`video_path` 必须是视频文件在**服务器**上的绝对路径。
> 调用方需先将视频上传到服务器 `/data/videos/` 目录，再用该路径调用接口。

```bash
# 先上传视频到服务器（本地执行）
scp ./test_video.mp4 user@<server-ip>:/data/videos/

# 再调用活体检测接口（本地或服务器执行）
curl -X POST http://localhost:8071/vrlMoveLiveness \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "task_id":    "task_001",
    "video_path": "/data/videos/test_video.mp4",
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
        {"action": "blink",      "passed": true, "confidence": 0.91, "msg": "检测到有效眨眼"},
        {"action": "mouth_open", "passed": true, "confidence": 0.88, "msg": "检测到有效张嘴"},
        {"action": "shake_head", "passed": true, "confidence": 0.86, "msg": "检测到有效摇头"}
      ]
    }
  }
}
```

### 7.6 接口文档

| 服务 | Swagger 文档地址 |
|------|-----------------|
| 人脸识别 | `http://<server-ip>:8070/docs` |
| 活体检测 | `http://<server-ip>:8071/docs` |

---

## 8. 日常运维

### Docker 方式

```bash
cd /opt/face_cls/deploy/pro

# 查看两个服务状态
docker compose ps

# 实时日志（所有服务）
docker compose logs -f --tail=100

# 单独查看某个服务日志
docker compose logs -f vrl-face
docker compose logs -f vrl-liveness

# 重启所有服务
docker compose restart

# 仅重启某个服务
docker compose restart vrl-face
docker compose restart vrl-liveness

# 停止所有服务
docker compose down
```

#### 拉取最新代码并重新部署（Docker）

> **执行 `git pull` 前先确认是否为 Git 仓库：**
>
> ```bash
> cd /opt/face_cls
> git status
> ```
>
> 如果提示 `fatal: not a git repository`，说明代码是通过 scp/tar 上传的，没有 `.git` 目录。
> 按下方选择一种方式解决后，再继续后续步骤。

---

##### 情况 A：非 Git 仓库 — 一次性补救（推荐）

在服务器上初始化 Git，绑定远程仓库，之后就可以正常 `git pull`：

```bash
cd /opt/face_cls

# 1. 初始化本地仓库
git init

# 2. 绑定远程仓库
git remote add origin <your-repo-url>

# 3. 拉取远程代码并覆盖本地
git fetch origin main
git reset --hard origin/main

# 4. 验证
git log --oneline -5
git remote -v
```

> 之后每次更新只需执行 `git pull origin main` 即可。

---

##### 情况 B：非 Git 仓库 — 继续用 scp 打包上传

如果没有或不方便配置 Git，每次在**本地**打包上传：

```bash
# 本地执行（Windows PowerShell）
cd E:\unified_pyprj
tar -czf face_cls.tar.gz `
    --exclude='face_cls/_bmad' `
    --exclude='face_cls/_bmad-output' `
    --exclude='face_cls/__pycache__' `
    --exclude='face_cls/.git' `
    --exclude='face_cls/*.pyc' `
    face_cls

scp face_cls.tar.gz root@<server-ip>:/opt/

# 服务器上覆盖解压（不删除 .env 和 models）
cd /opt
tar -xzf face_cls.tar.gz --overwrite
```

> **注意**：`--overwrite` 只覆盖同名文件，`.env` 和 `models/` 目录不在压缩包内（已被 exclude），不会被覆盖。

---

**更新全部服务：**

```bash
cd /opt/face_cls

# 1. 拉取最新代码（已是 Git 仓库时）
git pull origin main          # 若分支不是 main，替换为实际分支名

# 2. 重新构建并重启（两个服务都更新）
cd deploy/pro
docker compose build --no-cache   # --no-cache 强制完整重建，确保依赖也更新
docker compose up -d

# 3. 确认两个服务正常
docker compose ps
curl -s http://localhost:8070/healthz
curl -s http://localhost:8071/healthz
```

**只更新人脸识别服务（不影响活体检测）：**

```bash
cd /opt/face_cls
git pull origin main

cd deploy/pro
docker compose -f docker-compose.face.yaml build --no-cache
docker compose -f docker-compose.face.yaml up -d

# 验证
curl -s http://localhost:8070/healthz
```

**只更新活体检测服务（不影响人脸识别）：**

```bash
cd /opt/face_cls
git pull origin main

cd deploy/pro
docker compose -f docker-compose.liveness.yaml build --no-cache
docker compose -f docker-compose.liveness.yaml up -d

# 验证
curl -s http://localhost:8071/healthz
```

> **说明**：
> - `git pull` 只拉取代码，不会中断正在运行的容器。
> - `docker compose build` 重建镜像时，旧容器继续服务，`up -d` 完成后才切换，**服务中断时间极短**。
> - 如果只改了 Python 代码（未改 `requirements.txt`），可以去掉 `--no-cache` 加快构建速度。

### systemd 方式（裸机）

```bash
# 查看状态
sudo systemctl status vrl-face
sudo systemctl status vrl-liveness

# 实时日志
sudo journalctl -u vrl-face -f
sudo journalctl -u vrl-liveness -f

# 重启
sudo systemctl restart vrl-face
sudo systemctl restart vrl-liveness

# 停止
sudo systemctl stop vrl-face
sudo systemctl stop vrl-liveness
```

#### 拉取最新代码并重新部署（裸机）

> 同样先确认是否为 Git 仓库（`git status`），不是则参考上方 **情况 A / B** 处理。

**更新全部服务：**

```bash
cd /opt/face_cls

# 1. 拉取最新代码
git pull origin main

# 2. 如果 requirements.txt 有变化，重新安装依赖
sudo -u deploy bash -c "
  source /opt/face_cls/venv/bin/activate
  pip install -r /opt/face_cls/requirements.txt
"

# 3. 重启两个服务
sudo systemctl restart vrl-face vrl-liveness

# 4. 验证
sudo systemctl status vrl-face
sudo systemctl status vrl-liveness
curl -s http://localhost:8070/healthz
curl -s http://localhost:8071/healthz
```

**只更新某一个服务：**

```bash
cd /opt/face_cls
git pull origin main

# 只重启人脸识别
sudo systemctl restart vrl-face
curl -s http://localhost:8070/healthz

# 只重启活体检测
sudo systemctl restart vrl-liveness
curl -s http://localhost:8071/healthz
```

> **说明**：裸机方式 `git pull` 后代码立即生效于磁盘，但进程需要 `systemctl restart` 才会加载新代码。`restart` 期间服务会短暂中断（通常 < 5 秒）。

### 防火墙

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP（Nginx）
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8070/tcp  # 人脸识别服务（如不使用 Nginx 可直接开放）
sudo ufw allow 8071/tcp  # 活体检测服务（如不使用 Nginx 可直接开放）
sudo ufw --force enable
sudo ufw status
```

---

## 9. 故障排查

### 查看启动日志

```bash
# Docker
docker compose logs vrl-face     2>&1 | tail -50
docker compose logs vrl-liveness 2>&1 | tail -50

# systemd
sudo journalctl -u vrl-face     -n 50 --no-pager
sudo journalctl -u vrl-liveness -n 50 --no-pager
```

### 常见错误及解决

| 错误信息 | 影响服务 | 原因 | 解决方法 |
|----------|----------|------|----------|
| `ModuleNotFoundError: mediapipe` | vrl-liveness | mediapipe 未安装 | `pip install mediapipe>=0.10.0` |
| `Cannot find model buffalo_l` | vrl-face | InsightFace 模型未下载 | 参考 [第 6 节离线部署](#离线部署模型无外网时) |
| `OSError: face_landmarker.task not found` | vrl-liveness | MediaPipe 模型文件缺失 | 上传 `models/face_landmarker.task` |
| `Address already in use :8070` | vrl-face | 端口被占用 | `sudo lsof -i:8070` 查找并停止 |
| `Address already in use :8071` | vrl-liveness | 端口被占用 | `sudo lsof -i:8071` 查找并停止 |
| `Out of memory` | 任意 | 内存不足 | 将 `APP_WORKERS` 从 2 改为 1 后重启 |
| `video_path 不存在` | vrl-liveness | 路径错误 | 确认视频已上传到 `/data/videos/`，使用服务器绝对路径 |
| `不支持的动作` (400) | vrl-liveness | 动作名称有误 | 参考附录支持的动作列表 |

### 内存不足时降配

编辑 `deploy/pro/.env`：

```dotenv
APP_WORKERS=1
```

重启服务：

```bash
# Docker
docker compose restart

# systemd
sudo systemctl restart vrl-face vrl-liveness
```

### 查看响应时间

```bash
# 人脸识别
curl -o /dev/null -s -w "face    healthz: %{time_total}s\n" http://localhost:8070/healthz

# 活体检测
curl -o /dev/null -s -w "liveness healthz: %{time_total}s\n" http://localhost:8071/healthz
```

---

## 附录：接口速查表

### 人脸识别服务（端口 8070）

| 接口路径 | 方法 | 说明 | 请求格式 |
|----------|------|------|----------|
| `/healthz` | GET | 健康检查 | — |
| `/vrlFaceDetection` | POST | 人脸检测 | multipart/form-data |
| `/vrlFaceComparison` | POST | 1:1 人脸比对 | multipart/form-data |
| `/vrlFaceSearch` | POST | 1:N 人脸搜索 | multipart/form-data |
| `/docs` | GET | Swagger 交互文档 | — |

### 活体检测服务（端口 8071）

| 接口路径 | 方法 | 说明 | 请求格式 |
|----------|------|------|----------|
| `/healthz` | GET | 健康检查 | — |
| `/vrlMoveLiveness` | POST | 视频活体检测 | application/json |
| `/docs` | GET | Swagger 交互文档 | — |

### `vrlMoveLiveness` 请求参数

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `request_id` | string | ✅ | — | 调用方唯一请求 ID |
| `task_id` | string | ✅ | — | 任务 ID |
| `video_path` | string | ✅ | — | 视频文件在**服务器**上的绝对路径 |
| `actions` | list | ✅ | — | 动作列表，见下表 |
| `threshold_config.liveness_threshold` | float | ❌ | `0.5` | 全局活体阈值 |
| `threshold_config.action_threshold` | float | ❌ | `0.85` | 单动作通过阈值 |
| `action_config.max_video_duration` | float | ❌ | 不限 | 最大分析时长（秒） |
| `action_config.per_action_timeout` | float | ❌ | 平均分配 | 每动作时间窗口（秒） |

### 支持的动作列表

| `action` 值 | 说明 |
|-------------|------|
| `blink` | 眨眼 |
| `mouth_open` | 张嘴 |
| `shake_head` | 摇头（左右） |
| `nod` | 点头（上下） |
| `nod_down` | 低头 |
| `nod_up` | 抬头 |
| `turn_left` | 向左转头 |
| `turn_right` | 向右转头 |

