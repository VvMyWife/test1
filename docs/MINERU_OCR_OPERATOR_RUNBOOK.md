# MinerU OCR Operator Runbook

本文补充 OCR 算子的运行保护和排查说明，重点覆盖超时、失败文件记录、并发上限建议和日志定位。

## 1. 单 PDF 超时

纯 MinerU 入口只需要 PDF 路径；`archive_id` / `owner` / `triggered_by` 等业务字段由内部兼容层自动补齐，不出现在返回 JSON 中。算子调用时必须传入 `timeout_seconds`，避免异常 PDF 长时间卡住批处理：

```python
from platform_foundation.ocr import extract_pdf

result = extract_pdf(
    "$WORKSPACE/input/doc-001.pdf",
    output_dir="$WORKSPACE/output/doc-001",
    api_url="http://127.0.0.1:8000",
    timeout_seconds=1800,
)
```

Daft 批处理脚本也支持命令行超时：

```bash
python $WORKSPACE/platform-core-public-feature-foundation-base-operator/scripts/run-daft-batch-operate.py \
  --api-url http://127.0.0.1:8000 \
  --input-dir $WORKSPACE/input \
  --output-dir $WORKSPACE/output \
  --timeout-seconds 1800
```

建议起步值：

| 场景 | 建议超时 |
| --- | --- |
| 1-5 页普通 PDF | 600 秒 |
| 5-30 页或扫描质量较差 PDF | 1800 秒 |
| 大文件、复杂表格、首次冷启动 | 3600 秒 |

如果频繁超时，优先确认是否复用了常驻 `mineru-api`，不要先盲目把超时调大。

## 2. 失败文件记录

`scripts/run-daft-batch-operate.py` 会生成三层失败记录：

| 文件 | 说明 |
| --- | --- |
| `batch_report.json` | 总报告，包含成功数、失败数、每个 PDF 的耗时和产物路径 |
| `failed_files.json` | 失败 PDF 列表，适合人工查看 |
| `failed_files.jsonl` | 一行一个失败记录，适合二次重跑或脚本解析 |
| `<pdf_name>.error.json` | 单个失败 PDF 的错误详情 |

失败记录示例：

```json
{
  "pdf_path": "$WORKSPACE/input/bad.pdf",
  "success": false,
  "elapsed_seconds": 123.456,
  "error_type": "OperatorError",
  "error": "MinerU timed out while extracting layout",
  "timeout_seconds": 1800,
  "error_report": "$WORKSPACE/output/bad.error.json"
}
```

重跑失败文件时，建议根据 `failed_files.jsonl` 提取 `pdf_path`，复制到一个临时 input 目录，再单独执行批处理；不要直接覆盖原始报告目录。

## 3. 并发上限建议

当前演示服务器实测推荐：

| 能力 | 推荐并发 | 说明 |
| --- | --- | --- |
| 常驻 MinerU API + Daft 调用 | `--concurrency 2` | 当前最佳默认值，总耗时明显优于顺序处理 |
| 单机 CPU + Paddle 表格结构增强 | `1-2` | Paddle 模型常驻后可复用，但 CPU 仍然重 |
| Paddle 表格文本 OCR / PP-StructureV3 全流程 | 不建议 CPU 默认开启 | 单张表格图可达到分钟级，不适合几千文件批处理 |
| GPU 或独立 Paddle 常驻服务 | 需要重新压测 | 以显存、吞吐和失败率为准 |

原则：

1. `concurrency` 控制的是调用层并发，不写死在基础 OCR 算子内部。
2. 基础 OCR 算子保持单 PDF 粒度；Daft/Celery/Ray 负责批量调度。
3. MinerU API 应先启动为常驻服务，避免每个 PDF 冷启动模型。
4. Paddle 如果启用，必须在同一批处理进程或服务 worker 中复用模型；不要每个 PDF 启一个新 Python 进程。
5. 并发从 `2` 起步，只有当 CPU/GPU 内存、`mineru-api` 延迟和失败率都稳定时再上调。

## 4. Paddle 表格融合输出

启用 `LayoutExtractMinerUPaddleTableOperator` 或 `platform_foundation.ocr.mineru_layout_paddle_table.operate()` 后，算子会先跑 MinerU，再在 MinerU 识别出的表格裁剪图上跑 PaddleOCR 表格结构识别。默认参数会开启：

```python
{
    "enable_table_cell_refine": True,
    "table_cell_refine_when_tables_present": True,
    "table_cell_refine_fail_open": True,
    "emit_table_cells_as_text_blocks": True,
    "paddle_table_mode": "table_structure",
    "paddle_table_structure_init_kwargs": {"model_name": "SLANet_plus"},
}
```

最终结果仍然是统一的 `MinerUPdfResult` / `<pdf_name>.json`，不会变成 Paddle 私有格式。Paddle 的结果会被合并到这些位置：

| 位置 | 说明 |
| --- | --- |
| `parsed_pdf.pages[].table_blocks[]` | 结构化表格结果，按页保存 |
| `pages[].table_blocks[]` | 与页面输出一起给下游算子消费 |
| `pages[].text_blocks[]` | 默认额外追加 `block_type="table_cell"` 的单元格文本块，方便后续规则/检索直接按文本块处理 |
| `pages[].page_meta.table_cell_refine` | 本页表格增强摘要，如 provider、table_count、cell_count、artifact |
| `artifacts[]` | 新增 `kind="paddle_table_json"`，指向 `<output_dir>/paddle_table_structure.json` |

`table_blocks[]` 的结构大致如下：

```json
{
  "table_id": "p0-t0",
  "page_index": 0,
  "provider": "paddleocr_table_structure",
  "bounding_box": {"x": 10, "y": 20, "w": 300, "h": 120},
  "coord_space": "mineru_layout",
  "html": "<table>...</table>",
  "cells": [
    {
      "cell_id": "p0-t0-c0",
      "text": "单元格文本",
      "bounding_box": {"x": 12, "y": 24, "w": 80, "h": 24},
      "row_index": 0,
      "col_index": 0,
      "row_span": 1,
      "col_span": 1,
      "confidence": 0.95,
      "meta": {
        "source": "paddleocr_table_structure",
        "bbox_source": "table_structure.bbox"
      }
    }
  ],
  "meta": {
    "source": "paddleocr_table_structure",
    "image_uri": "/path/to/mineru/table_crop.jpg",
    "text_fill_source": "mineru_text_blocks"
  }
}
```

融合规则要点：

1. 坐标统一成 `mineru_layout`，`bounding_box` 使用 `{x, y, w, h}`。
2. `table_structure` 模式主要从 Paddle 取得单元格框，再用 MinerU 已识别的文本块中心点落入单元格的方式回填 `cell.text`。
3. 如果没有 MinerU 表格候选，默认跳过 Paddle，不影响基础 OCR 输出。
4. 如果 Paddle 报错且 `table_cell_refine_fail_open=True`，基础 MinerU 结果仍返回，并在 `page_meta.table_cell_refine` 写入失败摘要。

## 5. 日志和产物排查

常用路径：

| 路径 | 用途 |
| --- | --- |
| `$WORKSPACE/logs/mineru-api-8000.log` | 常驻 MinerU API 日志，具体以部署脚本为准 |
| `<output_dir>/batch_report.json` | Daft 批处理总报告 |
| `<output_dir>/failed_files.jsonl` | 失败文件清单 |
| `<output_dir>/<pdf_name>/**/_middle.json` | MinerU 中间结构 |
| `<output_dir>/<pdf_name>/**/_content_list.json` | 块级内容结构 |
| `<output_dir>/<pdf_name>.json` | 算子统一结构化输出 |
| `<output_dir>/<pdf_name>.error.json` | 单文件失败详情 |

排查顺序：

1. 看 `batch_report.json` 的 `failure_count`、`total_elapsed_seconds` 和每个 item 的 `elapsed_seconds`。
2. 如果某个 PDF 失败，看对应 `<pdf_name>.error.json`。
3. 如果大量 PDF 同时报错，看 MinerU API 日志是否有模型加载、端口、显存/内存或 HTTP 500 问题。
4. 如果所有 PDF 耗时接近且都很慢，确认命令里是否传了 `--api-url http://127.0.0.1:8000`。
5. 如果只有含表格的 PDF 很慢，确认是否开启了 Paddle 表格 OCR；CPU 环境默认不建议开启 Paddle 文本 OCR。
6. 如果 `middle_json` 存在但 `<pdf_name>.json` 不存在，说明 MinerU 已产出但算子转换阶段失败，优先看 `<pdf_name>.error.json`。

常用检查命令：

```bash
tail -n 100 $WORKSPACE/logs/mineru-api-8000.log
cat $WORKSPACE/output/failed_files.jsonl
find $WORKSPACE/output -name "*.error.json" -print
```
