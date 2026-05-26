# MinerU Docker 使用说明

目标：把 MinerU OCR 算子做成可迁移的 Docker 模块。目标服务器只需要 Docker、NVIDIA 驱动和 nvidia-container-toolkit；Python、MinerU、Torch、Paddle、Daft、`platform_foundation` 都在镜像内部。

## 1. 目标服务器前置条件

```bash
nvidia-smi
docker --version
docker compose version
```

GPU 容器必须能看到显卡：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

如果这一步失败，先安装或修复 `nvidia-container-toolkit`，否则 MinerU/Paddle 都无法使用 GPU。

## 2. 构建镜像

进入项目根目录：

```bash
cd /home/<user>/mineru_workspace/platform-core-public-feature-foundation-base-operator
```

构建镜像：

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile .
```

国内网络可以显式指定镜像源：

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  --build-arg PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
  .
```

说明：镜像里用了两个 Python 虚拟环境：

- `/opt/venvs/mineru`：MinerU + Torch + 算子代码。
- `/opt/venvs/paddle`：PaddleOCR + Paddle GPU + Paddle Table API。

这样做是为了避免 Torch 和 Paddle 在同一个 Python 环境里互相覆盖 cuDNN/NCCL 依赖。

## 3. 准备输入输出目录

```bash
mkdir -p /home/<user>/mineru_docker/input
mkdir -p /home/<user>/mineru_docker/output
cp /home/<user>/mineru_workspace/data/input/5.pdf /home/<user>/mineru_docker/input/
```

## 4. 启动纯 MinerU/OCR 服务

```bash
docker run -d --name mineru-operator \
  --gpus all \
  --shm-size 8g \
  -p 8000:8000 \
  -v /home/<user>/mineru_docker/input:/workspace/input \
  -v /home/<user>/mineru_docker/output:/workspace/output \
  -v mineru-cache:/workspace/.cache \
  -v mineru-logs:/workspace/logs \
  mineru-operator:latest
```

检查服务：

```bash
docker logs -f mineru-operator
curl http://127.0.0.1:8000/health
```

## 5. 在容器里跑目录批处理

OCR 模式：

```bash
docker exec mineru-operator \
  /opt/venvs/mineru/bin/python /opt/mineru_workspace/scripts/run-daft-batch-operate.py \
  /workspace/input \
  --output-dir /workspace/output \
  --table-engine ocr \
  --concurrency 2 \
  --overwrite
```

查看结果：

```bash
ls -lh /home/<user>/mineru_docker/output
cat /home/<user>/mineru_docker/output/batch_report.json
```

报告里会有：

```json
{
  "pdf_count": 1,
  "page_count": 10,
  "pages_per_second": 2.672
}
```

## 6. 启动 Paddle 表格增强

Paddle API 默认不启动，因为会占显存、启动慢。需要 Paddle 时重新启动容器：

```bash
docker rm -f mineru-operator

docker run -d --name mineru-operator \
  --gpus all \
  --shm-size 8g \
  -e ENABLE_PADDLE_API=true \
  -p 8000:8000 \
  -p 8200:8200 \
  -v /home/<user>/mineru_docker/input:/workspace/input \
  -v /home/<user>/mineru_docker/output:/workspace/output \
  -v mineru-cache:/workspace/.cache \
  -v mineru-logs:/workspace/logs \
  mineru-operator:latest
```

检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8200/health
```

Paddle 模式建议先小并发：

```bash
docker exec mineru-operator \
  /opt/venvs/mineru/bin/python /opt/mineru_workspace/scripts/run-daft-batch-operate.py \
  /workspace/input \
  --output-dir /workspace/output_paddle \
  --table-engine paddle \
  --concurrency 1 \
  --limit 1 \
  --overwrite
```

## 7. 一次性批处理模式

不想长期启动服务，也可以让容器启动服务后直接跑批处理：

```bash
docker run --rm \
  --gpus all \
  --shm-size 8g \
  -v /home/<user>/mineru_docker/input:/workspace/input \
  -v /home/<user>/mineru_docker/output:/workspace/output \
  -v mineru-cache:/workspace/.cache \
  mineru-operator:latest \
  batch /workspace/input --output-dir /workspace/output --table-engine ocr --concurrency 2 --overwrite
```

Paddle 一次性批处理：

```bash
docker run --rm \
  --gpus all \
  --shm-size 8g \
  -e ENABLE_PADDLE_API=true \
  -v /home/<user>/mineru_docker/input:/workspace/input \
  -v /home/<user>/mineru_docker/output:/workspace/output_paddle \
  -v mineru-cache:/workspace/.cache \
  mineru-operator:latest \
  batch /workspace/input --output-dir /workspace/output_paddle --table-engine paddle --concurrency 1 --limit 1 --overwrite
```

## 8. docker compose

项目根目录已经提供 `docker-compose.yml`：

```bash
cd /home/<user>/mineru_workspace/platform-core-public-feature-foundation-base-operator
mkdir -p data output
cp /home/<user>/mineru_workspace/data/input/5.pdf data/
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

运行批处理：

```bash
docker compose exec mineru-operator \
  /opt/venvs/mineru/bin/python /opt/mineru_workspace/scripts/run-daft-batch-operate.py \
  /workspace/input --output-dir /workspace/output --table-engine ocr --concurrency 2 --overwrite
```

启用 Paddle：

```bash
ENABLE_PADDLE_API=true docker compose up -d --build
```

`docker-compose.yml` 已经使用 `${ENABLE_PADDLE_API:-false}`，所以这条命令会直接覆盖默认值。

## 9. 常见问题

### ModuleNotFoundError: platform_foundation

Docker 镜像里不会出现这个问题，因为构建时已经执行：

```bash
pip install -e /opt/mineru_workspace/backend/foundation
```

裸机部署才需要手动安装：

```bash
cd backend/foundation
python -m pip install -e .
```

### Paddle 很慢

先确认容器里 Paddle 是 GPU 版：

```bash
docker exec mineru-operator /opt/venvs/paddle/bin/python - <<'PY'
import paddle
print(paddle.__version__)
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())
PY
```

必须看到 `True` 和 `gpu:0`。

### 目标服务器显存小

Paddle 模式先用：

```bash
--concurrency 1 --limit 1
```

确认稳定后再逐步提高。

### 首次运行很慢

首次运行会下载 MinerU/Paddle 模型。建议保留这个卷：

```bash
-v mineru-cache:/workspace/.cache
```

后续容器重建也能复用模型缓存。
