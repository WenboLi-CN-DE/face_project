#!/bin/bash
# 用法：bash down.sh [face|liveness|all]
set -eu
TARGET="${1:-all}"
case "$TARGET" in
  face)     docker compose -f docker-compose.face.yaml down ;;
  liveness) docker compose -f docker-compose.liveness.yaml down ;;
  all)      docker compose down ;;
  *)        echo "用法：bash down.sh [face|liveness|all]"; exit 1 ;;
esac
