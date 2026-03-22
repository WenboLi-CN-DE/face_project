#!/bin/bash
# 用法：bash down.sh [face|liveness|silent|all]
set -eu
TARGET="${1:-all}"
case "$TARGET" in
  face)     docker compose -f docker-compose.face.yaml down ;;
  liveness) docker compose -f docker-compose.liveness.yaml down ;;
  silent)   docker compose -f docker-compose.silent.yaml down ;;
  all)      docker compose down; docker compose -f docker-compose.silent.yaml down ;;
  *)        echo "用法：bash down.sh [face|liveness|silent|all]"; exit 1 ;;
esac
