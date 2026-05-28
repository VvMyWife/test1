# Docker 性能边界测试脚本

这个脚本用于在同一台服务器上控制变量测试 MinerU OCR 算子的性能边界：

- `table_engine`: `ocr` 或 `paddle`
- 调用方并发数：`CONCURRENCY`
- API 服务端最大并发数：
  - `MINERU_API_MAX_CONCURRENT_REQUESTS`: MinerU API 服务端最大并发
  - `PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS`: Paddle Table API 服务端最大并发，仅 Paddle 模式测试

脚本会在每个 API 服务端并发值变化时重启 Docker 服务并等待健康检查通过。服务启动和预热时间不会计入结果，最终以每次 `batch_report.json` 里的 `total_elapsed_seconds`、`page_count`、`pages_per_second` 为准。

## 基础运行

在项目根目录执行：

```bash
cd /root/mineru_workspace

python scripts/docker_benchmark_matrix.py \
  --engines ocr,paddle \
  --mineru-api-concurrency-values 1,2,4,8,12,16,24,32,64,128 \
  --paddle-api-concurrency-values 1,2,4,8,12,16,24,32 \
  --concurrency-values 1,2,4,8,12,16,24,32
```

默认输入目录是容器内 `/workspace/input`，也就是宿主机的 `data/input`。

默认输出：

```text
output/docker_benchmark/benchmark_state.json
output/docker_benchmark/docker_benchmark.xlsx
output/docker_benchmark/<engine>_api<api_max>_c<concurrency>/
```

## 指定输出路径

```bash
python scripts/docker_benchmark_matrix.py \
  --engines ocr,paddle \
  --mineru-api-concurrency-values 4,8,12,16,24,32 \
  --paddle-api-concurrency-values 2,4,8,12,16 \
  --concurrency-values 4,8,12,16,24,32 \
  --host-output-root output/server_benchmark_4090 \
  --excel-path output/server_benchmark_4090/result.xlsx
```

## 断点继续

脚本每跑完一个组合就会写入：

```text
benchmark_state.json
docker_benchmark.xlsx
```

如果中途断开，直接再次执行同一条命令即可。已成功的组合不会重复跑，已经记录失败边界的组合也不会继续测试更大的调用方并发。

如果要重新测试失败项：

```bash
python scripts/docker_benchmark_matrix.py \
  --rerun-failed \
  --mineru-api-concurrency-values 8,12,16 \
  --paddle-api-concurrency-values 2,4,8 \
  --concurrency-values 8,12,16,24
```

如果要完全重测：

```bash
python scripts/docker_benchmark_matrix.py \
  --reset-state \
  --mineru-api-concurrency-values 1,2,4,8,12,16,24,32 \
  --paddle-api-concurrency-values 1,2,4,8,12,16 \
  --concurrency-values 1,2,4,8,12,16,24,32
```

## Excel 结果说明

Excel 包含三个 sheet：

- `Summary`: OCR 和 Paddle 各自的最优参数
- `OCR`: OCR 全部测试组合
- `Paddle`: Paddle 全部测试组合

颜色含义：

- 绿色加粗：该模式下耗时最短的成功组合
- 红色：失败边界点，也就是该组合确实测试过并报错

核心字段：

- `mineru_api_max_concurrency`: MinerU API 服务端最大并发
- `paddle_api_max_concurrency`: Paddle Table API 服务端最大并发
- `concurrency`: 调用方批处理并发
- `total_elapsed_seconds`: 该批 PDF 总耗时，来自 `batch_report.json`
- `page_count`: 成功处理总页数，来自 `batch_report.json`
- `pages_per_second`: 每秒处理页数，来自 `batch_report.json`
- `failure_count`: 失败 PDF 数
- `error`: 失败原因摘要

## 注意

这个脚本会反复执行：

```bash
docker compose up -d --force-recreate --no-build mineru-operator
```

因此它适合在专门测试性能边界时运行，不建议和正式任务同时跑在同一台机器上。
