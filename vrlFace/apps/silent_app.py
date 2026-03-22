"""
静默活体检测服务独立入口

仅包含静默活体检测相关路由，独立部署在 8060 端口。
基于 deepface-antispoofing 的被动式活体检测。

启动:
    uvicorn vrlFace.apps.silent_app:app --host 0.0.0.0 --port 8060
"""

import logging
from fastapi import FastAPI

from vrlFace.silent_liveness.api import router as silent_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Silent Liveness Detection API",
    description="静默活体检测服务（基于 DeepFace Anti-Spoofing）",
    version="3.1.0",
)

app.include_router(silent_router)


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "silent-liveness", "version": "3.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8060)
