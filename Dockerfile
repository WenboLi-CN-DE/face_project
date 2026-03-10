# ── Stage 1: builder ─────────────────────────────────────────────────────
# 安装编译期依赖（gcc/cmake 等），构建 wheel 包后丢弃此阶段
FROM python:3.10-slim AS builder

WORKDIR /build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────
# 仅包含运行时库，不含编译工具链，镜像体积更小
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 仅安装运行时系统依赖（OpenCV / InsightFace / MediaPipe 所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段拷贝预编译的 wheel 并安装
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl \
    && rm -rf /wheels

# 拷贝应用代码（.dockerignore 排除了无关目录）
COPY . .

RUN mkdir -p /app/logs /app/models /app/data

EXPOSE 8070

CMD ["uvicorn", "vrlFace.main_fastapi:app", "--host", "0.0.0.0", "--port", "8070", "--workers", "2"]
