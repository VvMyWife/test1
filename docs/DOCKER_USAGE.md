# Docker 即插即用部署

目标：把 MinerU OCR 算子做成可迁移 Docker 模块。目标服务器只需要 Docker、NVIDIA 驱动和 nvidia-container-toolkit；Python、MinerU、Torch、Paddle、Daft、`platform_foundation` 都在镜像内部。

## 1. 前置检查

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

如果最后一条失败，先修复 GPU Docker 运行时。

## 2. 准备目录

在项目根目录执行：

```bash
mkdir -p data/input output logs run .cache/mineru-operator
cp your.pdf data/input/
```

不要在代码里写服务器绝对路径。所有宿主路径都通过 Docker bind mount 映射到容器固定路径：

```text
data/input              -> /workspace/input
output                  -> /workspace/output
.cache/mineru-operator  -> /workspace/.cache
logs                    -> /workspace/logs
run                     -> /workspace/run
```

## 3. 构建

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile .
```

国内网络：

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile \
  --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  --build-arg PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
  .
```

如果服务器已经存在可用 CUDA/MinerU 基础镜像，可以复用：

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile \
  --build-arg BASE_IMAGE=mineru:latest \
  .
```

镜像使用 CUDA 11.8 运行时，目的是兼容更多服务器驱动。MinerU/Torch 和 Paddle 分别装在两个 venv 里，避免依赖冲突。

## 4. 常驻服务

```bash
docker compose up -d --build
```

默认会启动：

```text
MinerU API:       http://127.0.0.1:8000
Paddle Table API: http://127.0.0.1:8200
```

如果宿主机端口已被占用：

```bash
MINERU_API_HOST_PORT=18000 PADDLE_TABLE_API_HOST_PORT=18200 docker compose up -d --build
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18200/health
```

如果 compose 构建时要复用本机已有基础镜像：

```bash
BASE_IMAGE=mineru:latest docker compose up -d --build
```

验证：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8200/health
```

只跑 OCR、不开 Paddle：

```bash
ENABLE_PADDLE_API=false docker compose up -d --build
```

## 5. 跑 OCR

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input \
  --output-dir /workspace/output/ocr \
  --table-engine ocr \
  --concurrency 12 \
  --overwrite
```

## 6. 跑 Paddle

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input \
  --output-dir /workspace/output/paddle \
  --table-engine paddle \
  --concurrency 2 \
  --overwrite
```

Paddle 模式必须看到 `table_cell_count > 0` 才说明表格增强真正生效。算子不会把 Paddle 失败伪装成 OCR 成功。

## 7. 单文件

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input/5.pdf \
  --output-dir /workspace/output/single_paddle \
  --table-engine paddle \
  --overwrite
```

## 8. 一次性处理

```bash
docker run --rm \
  --gpus all \
  --shm-size 8g \
  -e TABLE_ENGINE=ocr \
  -e CONCURRENCY=12 \
  -v ./data/input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./.cache/mineru-operator:/workspace/.cache \
  -v ./logs:/workspace/logs \
  mineru-operator:latest \
  batch
```

Paddle:

```bash
docker run --rm \
  --gpus all \
  --shm-size 8g \
  -e TABLE_ENGINE=paddle \
  -e CONCURRENCY=2 \
  -v ./data/input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./.cache/mineru-operator:/workspace/.cache \
  -v ./logs:/workspace/logs \
  mineru-operator:latest \
  batch
```

## 9. 迁移到其他服务器

导出镜像：

```bash
docker save mineru-operator:latest | gzip > mineru-operator-image.tar.gz
```

可选：导出模型缓存，避免目标服务器首次运行重新下载模型：

```bash
tar -czf mineru-operator-cache.tar.gz .cache/mineru-operator
```

目标服务器导入：

```bash
docker load -i mineru-operator-image.tar.gz
tar -xzf mineru-operator-cache.tar.gz
docker compose up -d
```

## 10. 排查

GPU：

```bash
docker compose exec mineru-operator nvidia-smi
docker compose exec mineru-operator /opt/venvs/paddle/bin/python - <<'PY'
import paddle
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())
PY
```

日志：

```bash
docker compose exec mineru-operator tail -n 100 /workspace/logs/mineru-api-8000.log
docker compose exec mineru-operator tail -n 100 /workspace/logs/paddle-table-api-8200.log
```
