ARG BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Standalone PaddleOCR table service.
#
# This image is intentionally separated from the MinerU/Torch runtime. Paddle
# cu118 needs cuDNN 8, while the MinerU/Torch cu118 stack commonly pulls cuDNN 9.
# Keeping Paddle in its own container prevents that dependency conflict during
# migration to machines with different NVIDIA driver versions.
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_VENV=/opt/venvs/paddle \
    PADDLE_TABLE_API_HOST=0.0.0.0 \
    PADDLE_TABLE_API_PORT=8200 \
    PADDLE_TABLE_API_PRELOAD=true \
    PADDLE_TABLE_API_PRELOAD_MODES=table_structure,ppstructurev3 \
    HOME=/workspace \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache/huggingface \
    MODELSCOPE_CACHE=/workspace/.cache/modelscope \
    PIP_CACHE_DIR=/workspace/.cache/pip

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libmagic1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-pip \
        python3-venv \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${PADDLE_VENV}" \
    && "${PADDLE_VENV}/bin/python" -m pip install -U pip setuptools wheel -i "${PIP_INDEX_URL}"

RUN "${PADDLE_VENV}/bin/python" -m pip install \
        -i "${PADDLE_INDEX_URL}" \
        paddlepaddle-gpu==3.3.1 \
    && "${PADDLE_VENV}/bin/python" -m pip install \
        -i "${PIP_INDEX_URL}" \
        "nvidia-cudnn-cu11<9" \
        paddleocr==3.5.0 \
        "paddlex[ocr]==3.5.2" \
        fastapi==0.136.1 \
        uvicorn==0.46.0 \
        pydantic==2.13.3

WORKDIR /opt/mineru_workspace
COPY backend/foundation /opt/mineru_workspace/backend/foundation
COPY docker/paddle-table-api-entrypoint.sh /opt/mineru_workspace/docker/paddle-table-api-entrypoint.sh

RUN "${PADDLE_VENV}/bin/python" -m pip install -e /opt/mineru_workspace/backend/foundation \
    && mkdir -p /workspace/.cache /workspace/logs /workspace/run \
    && chmod +x /opt/mineru_workspace/docker/paddle-table-api-entrypoint.sh

EXPOSE 8200
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/mineru_workspace/docker/paddle-table-api-entrypoint.sh"]
CMD ["serve"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /opt/mineru_workspace/docker/paddle-table-api-entrypoint.sh healthcheck
