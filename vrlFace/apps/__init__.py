"""
vrlFace.apps — 独立服务入口点

各服务可独立部署，也可通过 main_fastapi.py 合并部署。

    face_app.py     — 人脸识别服务 (8070)
    liveness_app.py — 动作活体检测服务 (8071)
    silent_app.py   — 静默活体检测服务 (8060)
"""
