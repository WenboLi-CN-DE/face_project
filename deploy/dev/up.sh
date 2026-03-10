#!/bin/bash
set -eux
SERVER_PORT=${APP_PORT:-8070}
docker compose up -d
docker compose ps
sleep 5
curl -f "http://localhost:${SERVER_PORT}/healthz" || echo "Warning: Health check failed"

