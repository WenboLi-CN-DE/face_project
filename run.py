#!/usr/bin/env python
"""
启动脚本 - 启动 FastAPI 服务

使用:
    python run.py
    或
    uvicorn vrlFace.main_fastapi:app --reload
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("启动人脸识别 API 服务")
    print("=" * 60)
    print("\n访问文档：http://localhost:8070/docs")
    print("健康检查：http://localhost:8070/healthz")
    print("\n按 Ctrl+C 停止服务\n")

    uvicorn.run("vrlFace.main_fastapi:app", host="0.0.0.0", port=8070, reload=True)
