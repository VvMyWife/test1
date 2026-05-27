# MinerU OCR Operator

这个项目把 MinerU PDF 解析封装成一个可复用算子，调用方只需要关心三件事：

- 要处理哪个 PDF 或哪个目录
- 结果输出到哪里
- 表格使用 `ocr` 还是 `paddle`

推荐交付方式是 Docker。镜像内置两套隔离 Python 环境：

- `/opt/venvs/mineru`: MinerU + Torch + Daft + `platform_foundation`
- `/opt/venvs/paddle`: PaddleOCR + Paddle GPU + Paddle Table API

这样可以避免 Torch 和 Paddle 在同一个 Python 环境里互相污染 CUDA/cuDNN 依赖。OCR 和 Paddle 模式都在同一台机器本地容器内处理文件，不需要跨服务器上传 PDF。

## 目录约定

宿主机只需要准备这几个相对目录：

```bash
mkdir -p data/input output logs run .cache/mineru-operator
```

把 PDF 放入：

```bash
cp your.pdf data/input/
```

容器内固定路径：

```text
/workspace/input   -> 输入 PDF
/workspace/output  -> 输出 JSON、batch_report.json、batch_report.csv
/workspace/.cache  -> MinerU/Paddle 模型缓存
/workspace/logs    -> 服务日志
```

## 构建镜像

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

目标服务器需要能运行 GPU Docker：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

## 启动服务

默认同时启动 MinerU API 和 Paddle Table API，后续可以自由使用 `ocr` 或 `paddle` 模式。

```bash
docker compose up -d --build
```

检查：

```bash
docker compose logs -f mineru-operator
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8200/health
```

如果只想跑纯 OCR、省显存：

```bash
ENABLE_PADDLE_API=false docker compose up -d --build
```

## 批量处理

OCR 模式：

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input \
  --output-dir /workspace/output/ocr \
  --table-engine ocr \
  --concurrency 12 \
  --overwrite
```

Paddle 表格增强模式：

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input \
  --output-dir /workspace/output/paddle \
  --table-engine paddle \
  --concurrency 2 \
  --overwrite
```

输出文件在宿主机：

```bash
ls output/ocr
cat output/ocr/batch_report.json
cat output/ocr/batch_report.csv
```

## 单文件处理

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input/5.pdf \
  --output-dir /workspace/output/single_paddle \
  --table-engine paddle \
  --overwrite
```

## Python 调用方式

在已经安装 `platform_foundation` 的 Python 环境里：

```python
from platform_foundation.ocr import extract_pdf_file, extract_pdf_dir

result = extract_pdf_file(
    "data/input/5.pdf",
    output_dir="output/single",
    table_engine="paddle",
)
print(result.json_path)

report = extract_pdf_dir(
    "data/input",
    output_dir="output/batch",
    table_engine="paddle",
    concurrency=2,
)
print(report.batch_report_path)
print(report.batch_report_csv_path)
```

Docker 容器里已经安装好了内部包，不需要手动写 `sys.path.insert(...)`。

## 一次性运行

不想常驻服务时，可以让容器启动服务后直接处理目录：

```bash
docker run --rm \
  --gpus all \
  --shm-size 8g \
  -v ./data/input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./.cache/mineru-operator:/workspace/.cache \
  -v ./logs:/workspace/logs \
  mineru-operator:latest \
  batch
```

用环境变量切换 Paddle：

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

## 从 kaifang 打包到其他服务器

镜像包含 Python、MinerU、Torch、Paddle、Daft 和项目代码：

```bash
docker save mineru-operator:latest | gzip > mineru-operator-image.tar.gz
```

如果想连模型缓存一起迁移：

```bash
tar -czf mineru-operator-cache.tar.gz .cache/mineru-operator
```

目标服务器导入：

```bash
docker load -i mineru-operator-image.tar.gz
tar -xzf mineru-operator-cache.tar.gz
docker compose up -d
```

## 质量约束

- `table_engine="ocr"` 保持 MinerU 原生输出，不做 Paddle 伪装。
- `table_engine="paddle"` 必须调用 Paddle Table API；如果 Paddle 失败，算子返回失败，不静默降级成 OCR。
- 并发只由调用参数 `--concurrency` 或 `concurrency=` 控制；服务器不再硬编码每台机器相同的算子限流。
- `batch_report.json` 包含 `page_count` 和 `pages_per_second`，方便按页评估吞吐。

## 常用排查

确认 Paddle 使用 GPU：

```bash
docker compose exec mineru-operator /opt/venvs/paddle/bin/python - <<'PY'
import paddle
print(paddle.__version__)
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())
PY
```

查看日志：

```bash
docker compose exec mineru-operator tail -n 100 /workspace/logs/mineru-api-8000.log
docker compose exec mineru-operator tail -n 100 /workspace/logs/paddle-table-api-8200.log
```

清理输出：

```bash
rm -rf output/*
```
