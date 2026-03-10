#!/bin/bash
set -eux

# ── 生产部署脚本 ──────────────────────────────────────────────
# 用法：cd deploy/pro && bash up.sh

SERVER_PORT=8070

check_env() {
    if [ ! -f ".env" ]; then
        echo "ERROR: .env 文件不存在，请先执行："
        echo "  cp .env.example .env && nano .env"
        exit 1
    fi
}

check_models() {
    if [ ! -f "../../models/face_landmarker.task" ]; then
        echo "WARNING: models/face_landmarker.task 不存在，活体检测可能失败"
    fi
}

wait_healthy() {
    local name=$1 port=$2
    echo "等待 ${name} 启动 (port ${port})..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${port}/healthz" > /dev/null 2>&1; then
            echo "✅ ${name} 健康检查通过"
            curl -s "http://localhost:${port}/healthz"
            echo ""
            return 0
        fi
        echo "  第 ${i}/30 次检查..."
        sleep 3
    done
    echo "❌ ${name} 启动超时，查看日志："
    echo "   docker compose logs ${name} --tail=50"
    return 1
}

check_env
check_models

# 创建视频上传目录（如果不存在）
mkdir -p /data/videos

docker compose build
docker compose up -d
docker compose ps

wait_healthy vrl-face     8070
wait_healthy vrl-liveness 8071

echo ""
echo "📖 人脸识别文档：http://localhost:8070/docs"
echo "📖 活体检测文档：http://localhost:8071/docs"
