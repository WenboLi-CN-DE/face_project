"""
合并服务入口（向后兼容，单端口部署用）

两服务独立部署请分别使用：
    vrlFace.face_app     — 人脸识别，端口 8070
    vrlFace.liveness_app — 活体检测，端口 8071

合并单端口部署:
    uvicorn vrlFace.main_fastapi:app --host 0.0.0.0 --port 8070
"""

import logging
from fastapi import FastAPI

from .face.api import router as face_router
from .liveness.api import router as liveness_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face & Liveness API",
    description="人脸识别与活体检测合并服务",
    version="3.1.0",
)

app.include_router(face_router)
app.include_router(liveness_router)


@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "service": "combined", "version": "3.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8070)
