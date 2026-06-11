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
mkdir -p data/input output logs run .cache/mineru-operator .cache/paddlex
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
/root/.paddlex     -> PaddleX/PaddleOCR 模型缓存
/workspace/logs    -> 服务日志
```

## 构建镜像

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile .
```

国内网络可以显式指定镜像源：

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile \
  --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  --build-arg PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
  .
```

如果服务器已经有可用 CUDA 基础镜像，也可以复用，避免重新拉 Docker Hub：

```bash
docker build -t mineru-operator:latest -f docker/Dockerfile \
  --build-arg BASE_IMAGE=mineru:latest \
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

首次启动 Paddle Table API 会下载并预加载 PaddleX 模型，可能需要数分钟；模型会缓存在 `.cache/paddlex`，后续重启会复用。

默认宿主机监听端口是 `18000/18200`，容器内部服务仍然是 `8000/8200`：

```bash
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18200/health
```

如果要让 compose 复用本机已有基础镜像：

```bash
BASE_IMAGE=mineru:latest docker compose up -d --build
```

检查：

```bash
docker compose logs -f mineru-operator
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18200/health
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

Paddle 指定字段坐标与 PDF 标注：

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input/5.pdf \
  --output-dir /workspace/output/paddle_fields \
  --table-engine paddle \
  --field-keywords 身份证号,姓名 \
  --overwrite
```

开启 `--field-keywords` 后，每个文档目录会额外生成：

```text
5.field_coordinates.json
5.field_annotations.pdf
```

`field_coordinates.json` 会按页面和坐标顺序输出命中的字段、所在表格 cell、`x/y/w/h`、四点坐标，以及换算后的 PDF 点坐标；`field_annotations.pdf` 会在对应位置画框标注。该能力主要用于 `table_engine=paddle` 的表格 cell 坐标。

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
    field_keywords=["身份证号", "姓名"],
)
print(result.json_path)
print(result.field_coordinates_path)
print(result.field_annotation_pdf_path)

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
  -v ./.cache/paddlex:/root/.paddlex \
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
  -v ./.cache/paddlex:/root/.paddlex \
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
tar -czf mineru-operator-cache.tar.gz .cache/mineru-operator .cache/paddlex
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
- `batch_report.json` 包含 `page_count` 和 `seconds_per_page`，单位是秒/页，方便按页评估耗时。

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
## Docker 性能边界测试

如果要自动测试当前服务器上 OCR/Paddle 两种模式的最佳并发参数，可以运行：

```bash
cd /root/mineru_workspace

python scripts/docker_benchmark_matrix.py \
  --engines ocr,paddle \
  --mineru-api-concurrency-values 1,2,4,8,12,16,24,32,64,128 \
  --paddle-api-concurrency-values 1,2,4,8,12,16,24,32 \
  --concurrency-values 1,2,4,8,12,16,24,32
```

脚本会反复重启 Docker 服务以分别修改 MinerU API 和 Paddle Table API 的服务端最大并发；服务启动时间不会计入测试结果。输出位于：

```text
output/docker_benchmark/benchmark_state.json
output/docker_benchmark/docker_benchmark.xlsx
```

中途断开后，重新执行同一条命令即可断点继续。详细说明见 `docs/DOCKER_BENCHMARK.md`。

## 图片输入和整页截图

批处理目录现在支持这些输入：

```text
.pdf, .jpg, .jpeg, .png, .bmp, .tif, .tiff
```

图片会在算子内部自动转换成单页 PDF，再复用原来的 MinerU/Paddle 流程。转换产物会保留在对应 artifact 目录下，例如：

```text
output/15/
└── 15.converted.pdf
```

递归处理业务文件夹时，输出会保留输入文件相对目录，避免所有文件被拍平到同一层：

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input/liqizhi_0610 \
  --output-dir /workspace/output/ocr_qizhi \
  --table-engine ocr \
  --concurrency 12 \
  --overwrite \
  --recursive
```

例如输入：

```text
/workspace/input/liqizhi_0610/
└── 3-WS-001/
    ├── 0189.jpg
    └── 0190.pdf
```

输出会保持为：

```text
/workspace/output/ocr_qizhi/
└── 3-WS-001/
    ├── 0189/
    │   ├── 0189.json
    │   ├── 0189.converted.pdf
    │   ├── 0189.converted_middle.json
    │   └── 0189.converted_content_list.json
    └── 0190/
        ├── 0190.json
        ├── 0190_middle.json
        └── 0190_content_list.json
```

整页截图导出默认关闭。推荐直接用 CLI 参数显式开启：

```bash
docker compose exec mineru-operator mineru-operator-batch \
  /workspace/input \
  --output-dir /workspace/output/ocr_with_screenshots \
  --table-engine ocr \
  --concurrency 4 \
  --enable-page-screenshots \
  --page-screenshot-dpi 144 \
  --overwrite
```

如果需要用环境变量控制已经运行的容器，必须通过 `docker compose exec -e` 传入：

```bash
docker compose exec \
  -e ENABLE_PAGE_SCREENSHOTS=true \
  -e PAGE_SCREENSHOT_DPI=144 \
  mineru-operator mineru-operator-batch \
  /workspace/input/5.pdf \
  --output-dir /workspace/output/single_with_screenshots \
  --table-engine ocr \
  --overwrite
```

输出结构：

```text
output/5/
├── 5.json
├── 5_middle.json
├── 5_content_list.json
└── page_screenshots/
    ├── page_0001.png
    ├── page_0002.png
    └── page_manifest.jsonl
```
