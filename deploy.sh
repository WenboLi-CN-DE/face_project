#!/bin/bash
# 一键部署脚本 - Ubuntu 20.04/22.04

set -e

# ========== 配置 ==========
APP_NAME="face_cls"
APP_DIR="/opt/face_cls"
SERVICE_NAME="face_cls"
PORT=8070
DEPLOY_USER="deploy"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ========== 检查 root 权限 ==========
if [ "$EUID" -ne 0 ]; then
    log_error "请使用 sudo 运行此脚本"
    exit 1
fi

# ========== 1. 系统准备 ==========
log_info "步骤 1/8: 更新系统..."
apt update && apt upgrade -y

log_info "步骤 2/8: 安装基础依赖..."
apt install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw \
    net-tools

# ========== 2. 创建部署用户 ==========
log_info "步骤 3/8: 创建部署用户..."
if id "$DEPLOY_USER" &>/dev/null; then
    log_warn "用户 $DEPLOY_USER 已存在"
else
    useradd -m -s /bin/bash "$DEPLOY_USER"
    usermod -aG sudo "$DEPLOY_USER"
    log_info "用户 $DEPLOY_USER 创建成功"
fi

# ========== 3. 准备目录 ==========
log_info "步骤 4/8: 准备应用目录..."
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/models"
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/logs"
chown -R "$DEPLOY_USER":"$DEPLOY_USER" "$APP_DIR"

# ========== 4. 安装应用 ==========
log_info "步骤 5/8: 安装应用..."
cd "$APP_DIR"

# 检测是否有代码
if [ ! -f "requirements.txt" ]; then
    log_error "未找到 requirements.txt，请先上传代码"
    log_warn "可以使用以下方式上传代码："
    log_warn "  git clone <repo-url> $APP_DIR"
    log_warn "  或 scp -r ./face_cls $DEPLOY_USER@server:$APP_DIR"
    exit 1
fi

# 创建虚拟环境
if [ ! -d "venv" ]; then
    su - "$DEPLOY_USER" -c "cd $APP_DIR && python3 -m venv venv"
fi

# 安装依赖
su - "$DEPLOY_USER" -c "cd $APP_DIR && source venv/bin/activate && pip install --upgrade pip"
su - "$DEPLOY_USER" -c "cd $APP_DIR && source venv/bin/activate && pip install -r requirements.txt"

# ========== 5. 配置环境变量 ==========
log_info "步骤 6/8: 配置环境变量..."
cat > "$APP_DIR/.env" << EOF
# 模型配置
FACE_MODEL_NAME=buffalo_l
FACE_DET_SIZE=640,640
FACE_GPU_ID=-1
FACE_THRESHOLD=0.55

# 服务配置
APP_HOST=0.0.0.0
APP_PORT=$PORT
APP_WORKERS=2

# API Token（可选，用于接口认证）
# API_TOKEN=your-secret-token-here
EOF
chown "$DEPLOY_USER":"$DEPLOY_USER" "$APP_DIR/.env"

# ========== 6. 创建 systemd 服务 ==========
log_info "步骤 7/8: 配置 systemd 服务..."
cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=$APP_NAME - Face Verification API
After=network.target

[Service]
Type=simple
User=$DEPLOY_USER
Group=$DEPLOY_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn vrlFace.main_fastapi:app --host \$APP_HOST --port \$APP_PORT --workers \$APP_WORKERS
Restart=always
RestartSec=10

# 资源限制
LimitNOFILE=65535
MemoryLimit=4G

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$APP_NAME

[Install]
WantedBy=multi-user.target
EOF

# ========== 7. 配置 Nginx ==========
log_info "步骤 8/8: 配置 Nginx..."
cat > "/etc/nginx/sites-available/$APP_NAME" << EOF
upstream ${APP_NAME}_backend {
    server 127.0.0.1:${PORT};
    keepalive 32;
}

server {
    listen 80;
    server_name _;
    
    access_log /var/log/nginx/${APP_NAME}_access.log;
    error_log /var/log/nginx/${APP_NAME}_error.log;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://${APP_NAME}_backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
    
    location /healthz {
        proxy_pass http://${APP_NAME}_backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
    }
}
EOF

ln -sf "/etc/nginx/sites-available/$APP_NAME" "/etc/nginx/sites-enabled/$APP_NAME"
rm -f /etc/nginx/sites-enabled/default

# ========== 8. 配置防火墙 ==========
log_info "配置防火墙..."
ufw --force enable || true
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp

# ========== 9. 启动服务 ==========
log_info "启动服务..."
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"
systemctl enable nginx
systemctl restart nginx

# ========== 完成 ==========
echo ""
echo "=========================================="
log_info "部署完成！"
echo "=========================================="
echo ""
echo "服务状态查看：sudo systemctl status $SERVICE_NAME"
echo "服务日志查看：sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "API 地址：http://$(hostname -I | awk '{print $1}'):$PORT"
echo "API 文档：http://$(hostname -I | awk '{print $1}'):$PORT/docs"
echo ""
echo "下一步："
echo "1. 上传模型文件到 $APP_DIR/models/"
echo "2. 上传数据到 $APP_DIR/data/"
echo "3. 配置域名（可选）：修改 /etc/nginx/sites-available/$APP_NAME"
echo "4. 配置 HTTPS（可选）：sudo certbot --nginx"
echo ""
