#!/bin/bash
# ── 生产停止脚本 ─────────────────────────────────────────────
#
# 用法：
#   bash down.sh              # 停止全部服务
#   bash down.sh face         # 只停止人脸识别服务
#   bash down.sh liveness     # 只停止活体检测服务
#   bash down.sh silent       # 只停止静默活体服务
#
set -eu

TARGET="${1:-all}"

case "$TARGET" in
  face)
    echo ">>> 停止人脸识别服务"
    docker compose -f docker-compose.face.yaml down
    ;;
  liveness)
    echo ">>> 停止活体检测服务"
    docker compose -f docker-compose.liveness.yaml down
    ;;
  silent)
    echo ">>> 停止静默活体检测服务"
    docker compose -f docker-compose.silent.yaml down
    ;;
  all)
    echo ">>> 停止全部服务"
    docker compose down
    docker compose -f docker-compose.silent.yaml down
    ;;
  *)
    echo "用法：bash down.sh [face|liveness|silent|all]"
    exit 1
    ;;
esac
