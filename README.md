# platform-core public foundation base operator

本项目把可复用的 PDF 文档解析能力封装在 `backend/foundation`。调用方不需要关心 MinerU、Paddle、Daft、锁、产物目录、错误文件、统计指标这些内部细节。

## 关键结论

- 单 PDF 入口：`extract_pdf_file(pdf_path, output_dir=...)`
- 目录批处理入口：`extract_pdf_dir(input_dir, output_dir=..., table_engine=..., concurrency=...)`
- 默认是纯 MinerU OCR 路径：`table_engine="ocr"`，会开启 MinerU 表格 fallback，但只输出粗粒度表格文本
- 需要 Paddle 表格增强时只传：`table_engine="paddle"` 或 `use_paddle_tables=True`
- 算子默认不再额外限制 `max_inflight`，批处理并发只由调用方传入的 `concurrency` 决定
- 弱机器或共享 GPU 环境可以显式传 `ocr_operator_max_inflight` / `paddle_operator_max_inflight` 做保守限流
- 当前压测建议 MinerU API 服务端并发为：`max_concurrent_requests=4`
- 输出是一对一 JSON：`output/<pdf_stem>.json`
- 中间产物在：`output/<pdf_stem>/`
- 单文件错误在：`output/<pdf_stem>.error.json`
- 批处理报告在：`output/batch_report.json` 和 `output/batch_report.csv`

## Python 调用

单个 PDF：

```python
from platform_foundation.ocr import extract_pdf_file

result = extract_pdf_file(
    "/home/kaifang/mineru_workspace/data/input/5.pdf",
    output_dir="/home/kaifang/mineru_workspace/output",
)

if result.success:
    print(result.json_path)
else:
    print(result.error)
    print(result.error_report)
```

批量处理目录：

```python
from platform_foundation.ocr import extract_pdf_dir

report = extract_pdf_dir(
    "/home/kaifang/mineru_workspace/data/input",
    output_dir="/home/kaifang/mineru_workspace/output",
    table_engine="paddle",
    concurrency=2,
)

print(report.batch_report_path)
print(report.batch_report_csv_path)
print(report.success_count)
print(report.failure_count)
```

## CLI 调用

进入服务器工作目录：

```bash
cd /home/kaifang/mineru_workspace
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mineru312
```

纯 MinerU/OCR，处理目录：

```bash
python scripts/run-daft-batch-operate.py input --output-dir output --table-engine ocr --concurrency 8
```

Paddle 表格增强，处理目录：

```bash
python scripts/run-daft-batch-operate.py input --output-dir output_paddle --table-engine paddle --concurrency 2
```

单个 PDF：

```bash
python scripts/run-daft-batch-operate.py input/5.pdf --output-dir output_single
```

单个 PDF 启用 Paddle：

```bash
python scripts/run-daft-batch-operate.py input/5.pdf --output-dir output_paddle_single --table-engine paddle
```

## Docker 即插即用部署

项目已经提供 Docker 交付文件：

- `docker/Dockerfile`
- `docker/entrypoint.sh`
- `docker-compose.yml`
- `docs/DOCKER_USAGE.md`

构建镜像：

```bash
cd /home/<user>/mineru_workspace/platform-core-public-feature-foundation-base-operator
docker build -t mineru-operator:latest -f docker/Dockerfile .
```

启动纯 MinerU/OCR 服务：

```bash
docker run -d --name mineru-operator \
  --gpus all \
  --shm-size 8g \
  -p 8000:8000 \
  -v /home/<user>/mineru_docker/input:/workspace/input \
  -v /home/<user>/mineru_docker/output:/workspace/output \
  -v mineru-cache:/workspace/.cache \
  mineru-operator:latest
```

容器内跑批处理：

```bash
docker exec mineru-operator \
  /opt/venvs/mineru/bin/python /opt/mineru_workspace/scripts/run-daft-batch-operate.py \
  /workspace/input --output-dir /workspace/output --table-engine ocr --concurrency 2 --overwrite
```

启用 Paddle 表格增强时加：

```bash
-e ENABLE_PADDLE_API=true -p 8200:8200
```

完整测试步骤见 [docs/DOCKER_USAGE.md](docs/DOCKER_USAGE.md)。

## 职责边界

`extract_pdf_file()` 负责：

- 路径检查
- 输出目录和中间产物目录创建
- 调用 MinerU
- 按需启用 Paddle 表格增强
- 写单个 JSON
- 写单个 error JSON
- 统计单个 PDF 的 metrics
- 返回结构化 `PdfFileExtractResult`

`extract_pdf_dir()` 负责：

- 扫描目录
- 递归查找 PDF
- 去重
- limit 限制
- 断点续跑
- 并发控制
- 自动选择 sequential/thread/Daft 执行
- 写 `batch_report.json`
- 写 `batch_report.csv`
- 写 `failed_files.json`
- 写 `failed_files.jsonl`
- 返回结构化 `PdfDirExtractReport`

CLI 脚本只负责：

- 解析命令行参数
- 调用 `extract_pdf_file()` 或 `extract_pdf_dir()`
- 打印报告路径

## 输出结构

最终 JSON 默认只保留一份页面结构，读取入口是：

```text
pages[]
pages[].text_blocks[]
pages[].table_blocks[]
pages[].table_blocks[].cells[]
pages[].page_meta
```

不会再把同一份页面内容复制进 `parsed_pdf`。

## 表格模式

纯 MinerU：

- 速度快
- 默认路径
- 当前压测推荐目录批处理 `concurrency=8`
- 默认不设置 `max_inflight`，实际并发由 `extract_pdf_dir(..., concurrency=...)` 控制
- 内部使用 `--table true`，让 MinerU 保留自己的表格输出，但不启用 Paddle 表格增强
- 适合先把 PDF 内容快速转成一对一 JSON
- 不解析、不拆分、不合并 MinerU 表格结果
- MinerU 输出什么就保留什么：`html`、bbox、raw meta 原样挂到 `table_blocks`；不会人为制造 `cells`

Paddle 表格增强：

- 只在 `table_engine="paddle"` 时启用
- 当前推荐从 `concurrency=2` 起测
- 默认不设置算子级 `max_inflight`，但建议调用方先用较小 `concurrency` 找本机稳定边界
- 更适合需要单元格结构增强的表格
- 更慢，更吃 GPU 显存
- 内部自动开启 MinerU 表格候选区，再由 Paddle 做单元格增强
- Paddle 结果写入 `pages[].table_blocks[].cells[]`
- Paddle 原始产物在 `output/<pdf_stem>/paddle_table_structure.json`

Paddle 模式会利用 `pred_html` 对被 OCR 拆开的中文单元格做通用合并，包括同一行横向碎片和跨行换行碎片。例如 `入` + `党` + `时` + `间` 会归并为 `入党时间`，`参加工` + `作时间` 会归并为 `参加工作时间`。

## 批处理报告

查看 JSON 报告：

```bash
cat output/batch_report.json
```

查看 CSV 报告：

```bash
cat output/batch_report.csv
```

关键字段：

- `success_count`
- `failure_count`
- `skipped_count`
- `page_count`
- `pages_per_second`
- `total_elapsed_seconds`
- `items[].processing_seconds`
- `items[].queue_wait_seconds`
- `items[].table_block_count`
- `items[].table_cell_count`

## HTTP API

路径方式：

```bash
curl -X POST http://127.0.0.1:8001/api/v1/operators/mineru/layout-extract-path \
  -H "Content-Type: application/json" \
  -d '{"pdf_path":"/home/kaifang/mineru_workspace/input/5.pdf","output_dir":"/home/kaifang/mineru_workspace/output_api"}'
```

启用 Paddle：

```bash
curl -X POST http://127.0.0.1:8001/api/v1/operators/mineru/layout-extract-path \
  -H "Content-Type: application/json" \
  -d '{"pdf_path":"/home/kaifang/mineru_workspace/input/5.pdf","output_dir":"/home/kaifang/mineru_workspace/output_api_paddle","mineru_options":{"table_engine":"paddle"}}'
```

上传方式：

```bash
curl -F "files=@input/5.pdf" \
  -F "output_dir=/home/kaifang/mineru_workspace/output_api" \
  http://127.0.0.1:8001/api/v1/operators/mineru/layout-extract
```

## 测试

Foundation：

```bash
cd backend/foundation
uv run pytest
uv run ruff check .
```

Platform API：

```bash
cd backend/services/platform-api
uv run pytest
uv run ruff check .
```

## Paddle 并发限流运行步骤

并发限制只放在抽取端：

- `concurrency`：目录批处理线程数。
- `paddle_operator_max_inflight`：最多多少个 PDF 同时进入 Paddle operator。
- `ocr_operator_max_inflight`：最多多少个 PDF 同时进入普通 OCR operator。

`start-mineru-api.sh` 会把 MinerU 自带的 `MINERU_API_MAX_CONCURRENT_REQUESTS` 默认设为 `128`，避免 8100 服务端默认值 `3` 抢先限流。`start-paddle-table-api.sh` 不再设置 Paddle 表格抽取的服务端并发上限。

启动 Paddle Table API：

```bash
cd /home/ubuntu/mineru_workspace
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mineru312
bash scripts/start-paddle-table-api.sh
curl http://127.0.0.1:8200/health
```

运行目录抽取，并限制进入 Paddle operator 的 PDF 数量：

```bash
cd /home/ubuntu/mineru_workspace
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mineru312
python scripts/extract_pdf_dir_minimal.py
```

抽取脚本里直接改这两个值：

- `concurrency=5`：目录批处理线程数。
- `paddle_operator_max_inflight=4`：客户端 Paddle operator 最大同时在跑的 PDF 数量；填 `None` 表示不限制。

建议先用 `concurrency=5`、`paddle_operator_max_inflight=4` 跑；如果 8200 服务不稳定或显存压力过大，就把 `paddle_operator_max_inflight` 降到 `2`。
