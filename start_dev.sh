#!/bin/bash
set -eux

SERVER_PORT=8070

echo "Starting FastAPI server (dev mode)..."
uvicorn vrlFace.main_fastapi:app --reload --host 0.0.0.0 --port ${SERVER_PORT}
