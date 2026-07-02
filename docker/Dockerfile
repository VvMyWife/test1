# syntax=docker/dockerfile:1.7

# CUDA 11.8 is the conservative default for older NVIDIA drivers such as 535.xx.
# The image intentionally uses two Python virtualenvs:
# - /opt/venvs/mineru: MinerU + Torch + platform_foundation
# - /opt/venvs/paddle: PaddleOCR + Paddle GPU + platform_foundation Paddle API
#
# Keeping Paddle and Torch in separate environments avoids the cuDNN/NCCL wheel
# conflicts seen when installing both stacks into a single Python environment.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKSPACE_ROOT=/workspace \
    APP_ROOT=/opt/mineru_workspace \
    MINERU_MODEL_SOURCE=modelscope \
    MINERU_API_HOST=0.0.0.0 \
    MINERU_API_PORT=8000 \
    MINERU_API_URL=http://127.0.0.1:8000 \
    MINERU_API_MAX_CONCURRENT_REQUESTS=128 \
    PADDLE_TABLE_API_HOST=0.0.0.0 \
    PADDLE_TABLE_API_PORT=8200 \
    PADDLE_TABLE_API_URL=http://127.0.0.1:8200 \
    PADDLE_TABLE_API_PRELOAD=true \
    PADDLE_TABLE_API_PRELOAD_MODES=layout,ocr,table \
    ENABLE_PADDLE_API=true \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache/huggingface \
    MODELSCOPE_CACHE=/workspace/.cache/modelscope \
    TORCH_HOME=/workspace/.cache/torch \
    PIP_CACHE_DIR=/workspace/.cache/pip

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libmagic1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        poppler-utils \
        python3 \
        python3-pip \
        python3-venv \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venvs/mineru \
    && python3 -m venv /opt/venvs/paddle \
    && /opt/venvs/mineru/bin/python -m pip install -U pip setuptools wheel \
    && /opt/venvs/paddle/bin/python -m pip install -U pip setuptools wheel

# MinerU/Torch environment. Torch is installed from the CUDA 11.8 PyTorch wheel index.
RUN /opt/venvs/mineru/bin/python -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.6.0+cu118 \
        torchvision==0.21.0+cu118 \
        torchaudio==2.6.0+cu118 \
    && /opt/venvs/mineru/bin/python -m pip install \
        -i "${PIP_INDEX_URL}" \
        mineru==3.1.6 \
        daft==0.7.11 \
        fastapi==0.136.1 \
        uvicorn==0.46.0

# Paddle environment. This is isolated from Torch to prevent CUDA dependency conflicts.
RUN /opt/venvs/paddle/bin/python -m pip install \
        -i "${PADDLE_INDEX_URL}" \
        paddlepaddle-gpu==3.3.1 \
    && /opt/venvs/paddle/bin/python -m pip install \
        -i "${PIP_INDEX_URL}" \
        paddleocr==3.5.0 \
        "paddlex[ocr]==3.5.2" \
        fastapi==0.136.1 \
        uvicorn==0.46.0

WORKDIR /opt/mineru_workspace
COPY . /opt/mineru_workspace

RUN /opt/venvs/mineru/bin/python -m pip install -e /opt/mineru_workspace/backend/foundation \
    && /opt/venvs/paddle/bin/python -m pip install -e "/opt/mineru_workspace/backend/foundation[paddle-api]" \
    && mkdir -p /workspace/input /workspace/output /workspace/.cache /workspace/logs /workspace/run \
    && chmod +x /opt/mineru_workspace/docker/entrypoint.sh

ENV PATH=/opt/venvs/mineru/bin:${PATH}
WORKDIR /workspace

EXPOSE 8000 8200
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/mineru_workspace/docker/entrypoint.sh"]
CMD ["server"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /opt/mineru_workspace/docker/entrypoint.sh healthcheck
