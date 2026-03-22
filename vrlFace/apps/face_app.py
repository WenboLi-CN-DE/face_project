"""
人脸识别服务独立入口

仅包含人脸识别相关路由，独立部署在 8070 端口。

启动:
    uvicorn vrlFace.apps.face_app:app --host 0.0.0.0 --port 8070
"""

import logging
from fastapi import FastAPI

from vrlFace.face.api import router as face_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Recognition API",
    description="人脸识别服务（检测 / 比对 / 搜索）",
    version="3.1.0",
)

app.include_router(face_router)


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "face", "version": "3.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8070)
