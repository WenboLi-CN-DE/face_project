#!/bin/bash
# ── 生产部署脚本 ─────────────────────────────────────────────
#
# 用法：
#   bash up.sh              # 同时启动两个服务
#   bash up.sh face         # 只启动人脸识别服务 (8070)
#   bash up.sh liveness     # 只启动活体检测服务 (8071)
#
set -eu

TARGET="${1:-all}"

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
    echo "   docker compose -f docker-compose.${name#vrl-}.yaml logs --tail=50"
    return 1
}

check_env
check_models
mkdir -p /data/videos

case "$TARGET" in
  face)
    echo ">>> 启动人脸识别服务 (8070)"
    docker compose -f docker-compose.face.yaml build
    docker compose -f docker-compose.face.yaml up -d
    docker compose -f docker-compose.face.yaml ps
    wait_healthy vrl-face 8070
    echo "📖 人脸识别文档：http://localhost:8070/docs"
    ;;
  liveness)
    echo ">>> 启动活体检测服务 (8071)"
    docker compose -f docker-compose.liveness.yaml build
    docker compose -f docker-compose.liveness.yaml up -d
    docker compose -f docker-compose.liveness.yaml ps
    wait_healthy vrl-liveness 8071
    echo "📖 活体检测文档：http://localhost:8071/docs"
    ;;
  all)
    echo ">>> 启动全部服务 (8070 + 8071)"
    docker compose build
    docker compose up -d
    docker compose ps
    wait_healthy vrl-face     8070
    wait_healthy vrl-liveness 8071
    echo "📖 人脸识别文档：http://localhost:8070/docs"
    echo "📖 活体检测文档：http://localhost:8071/docs"
    ;;
  *)
    echo "用法：bash up.sh [face|liveness|all]"
    echo "  face     — 只启动人脸识别服务 (8070)"
    echo "  liveness — 只启动活体检测服务 (8071)"
    echo "  all      — 启动全部服务（默认）"
    exit 1
    ;;
esac
