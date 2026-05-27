# Paddle Table API Docker 部署

目标：把 PaddleOCR 表格增强从宿主机 conda 环境里拆出来，作为长期运行的 Docker 服务，避免和 MinerU/Torch 环境发生 cuDNN/NCCL 依赖冲突。

## 架构

```text
MinerU API Docker:       http://127.0.0.1:8000
Paddle Table API Docker: http://127.0.0.1:8200

Python 主流程:
  extract_pdf_file(..., table_engine="paddle")
  -> MinerU 做 OCR 和表格候选
  -> 调 Paddle Table API 做表格 cells
  -> 融合成最终 JSON
```

Paddle 容器使用 CUDA 11.8 + Paddle cu118。这样旧驱动机器也能部署；同时因为 Paddle 在独立容器内，不会污染 `mineru312` 的 Torch 依赖。

## 启动

在仓库根目录执行：

```bash
cd /home/kaifang/mineru_workspace/platform-core-public-feature-foundation-base-operator

WORKSPACE_HOST_ROOT=/home/kaifang/mineru_workspace \
PADDLE_TABLE_API_BASE_IMAGE=mineru:latest \
PADDLE_TABLE_API_DOCKER_BUILD=auto \
bash scripts/start-paddle-table-api-docker.sh
```

`PADDLE_TABLE_API_BASE_IMAGE=mineru:latest` 是 kaifang 当前部署的快捷方式：复用本机已有的 MinerU CUDA 镜像，避免从 Docker Hub 拉取 `nvidia/cuda` 基础镜像。其他服务器如果 Docker Hub 或镜像源可用，可以不传这个变量，默认使用 `nvidia/cuda:11.8.0-runtime-ubuntu22.04`。

输出里应看到：

```text
paddle-table-api ready: http://127.0.0.1:8200/health
export PADDLE_TABLE_API_URL=http://127.0.0.1:8200
```

如果当前 8200 被宿主机 Python 进程占用，先停止旧服务：

```bash
pkill -f "platform_foundation.ocr.paddle_table_api:app" || true
```

或者使用旧脚本停止：

```bash
bash scripts/stop-paddle-table-api.sh
```

## 验证

```bash
curl http://127.0.0.1:8200/health
```

单文件验证：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mineru312
cd /home/kaifang/mineru_workspace

export MINERU_API_URL=http://127.0.0.1:8000
export PADDLE_TABLE_API_URL=http://127.0.0.1:8200

python - <<'PY'
from pathlib import Path
from platform_foundation.ocr import extract_pdf_file

workspace = Path("/home/kaifang/mineru_workspace")
result = extract_pdf_file(
    workspace / "data" / "input" / "5.pdf",
    output_dir=workspace / "output_paddle_docker_smoke",
    table_engine="paddle",
    overwrite=True,
)
print(result.model_dump(mode="json"))
PY
```

验收标准：

```text
success = True
table_engine = paddle
table_cell_count > 0
paddle_table_artifact 指向 paddle_table_structure.json
```

## 停止

```bash
bash scripts/stop-paddle-table-api-docker.sh
```

## 关键注意

Paddle API 容器必须能看到主流程传进去的 PDF 路径和输出路径。因此启动脚本会把宿主机工作目录按相同绝对路径挂进容器：

```text
/home/kaifang/mineru_workspace:/home/kaifang/mineru_workspace
```

迁移到其他服务器时，把 `WORKSPACE_HOST_ROOT` 改成目标机器上的 workspace 绝对路径即可。

## Remote client mode

The Paddle Table API can also run on one stable GPU server while other machines
call it over LAN.

Server side example:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
# mineru-api         0.0.0.0:8000->8000/tcp
# paddle-table-api   0.0.0.0:8200->8200/tcp
```

Client side example:

```bash
export MINERU_API_URL=http://192.168.1.173:8000
export PADDLE_TABLE_API_URL=http://192.168.1.173:8200
python scripts/extract_pdf_file_minimal.py
```

When `PADDLE_TABLE_API_URL` points to a non-local host, the Python client sends
the PDF and table crop images in the request payload. The remote Paddle service
does not need to see the client's absolute filesystem paths.
