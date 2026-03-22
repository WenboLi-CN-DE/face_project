"""
活体检测服务独立入口

仅包含活体检测相关路由，独立部署在 8071 端口。

启动:
    uvicorn vrlFace.apps.liveness_app:app --host 0.0.0.0 --port 8071
"""

import logging
from fastapi import FastAPI

from vrlFace.liveness.api import router as liveness_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Liveness Detection API",
    description="活体检测服务（视频动作检测）",
    version="3.1.0",
)

app.include_router(liveness_router)


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "liveness", "version": "3.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8071)
