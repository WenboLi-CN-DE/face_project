#!/bin/bash
# 用法：bash up.sh [face|liveness|all]
set -eu

TARGET="${1:-all}"

check_env() {
    if [ ! -f ".env" ]; then
        echo "ERROR: .env 文件不存在"; exit 1
    fi
}

wait_healthy() {
    local name=$1 port=$2
    echo "等待 ${name} (port ${port})..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${port}/healthz" > /dev/null 2>&1; then
            echo "✅ ${name} 健康检查通过"
            curl -s "http://localhost:${port}/healthz"; echo ""
            return 0
        fi
        echo "  第 ${i}/30 次..."; sleep 3
    done
    echo "❌ ${name} 启动超时"; return 1
}

check_env

case "$TARGET" in
  face)
    docker compose -f docker-compose.face.yaml up -d
    docker compose -f docker-compose.face.yaml ps
    wait_healthy vrl-face 8070
    echo "📖 http://localhost:8070/docs"
    ;;
  liveness)
    docker compose -f docker-compose.liveness.yaml up -d
    docker compose -f docker-compose.liveness.yaml ps
    wait_healthy vrl-liveness 8071
    echo "📖 http://localhost:8071/docs"
    ;;
  all)
    docker compose up -d
    docker compose ps
    wait_healthy vrl-face     8070
    wait_healthy vrl-liveness 8071
    echo "📖 http://localhost:8070/docs"
    echo "📖 http://localhost:8071/docs"
    ;;
  *)
    echo "用法：bash up.sh [face|liveness|all]"; exit 1
    ;;
esac
