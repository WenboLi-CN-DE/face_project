@echo off
echo ========================================
echo 安装活体检测依赖（幂等）
echo ========================================

echo.
echo 正在安装 mediapipe...
python -m pip install "mediapipe>=0.10.0"

echo.
echo 检查 requirements.txt 是否已有 mediapipe...
findstr /C:"mediapipe" ..\..\requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo mediapipe^>=0.10.0 >> ..\..\requirements.txt
    echo 已添加 mediapipe 到 requirements.txt
) else (
    echo mediapipe 已存在于 requirements.txt，跳过
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 使用方法:
echo   python -m vrlFace.liveness.cli --camera 0
echo   python -m vrlFace.liveness.cli --video path\to\video.mp4
echo   python -m vrlFace.liveness.recorder --video path\to\video.mp4
echo.
