# MinerU OCR 算子使用文档

目标：对调用方暴露“纯 MinerU”能力。调用方不需要传 `archive_id`、`archive_owner_user_id`、`triggered_by_user_id`、`doc_id` 这类平台业务字段。

## 1. Python 单 PDF

```python
from pathlib import Path

from platform_foundation.ocr import extract_pdf

result = extract_pdf(
    "$WORKSPACE/input/example.pdf",
    output_dir="$WORKSPACE/output/example",
    api_url="http://127.0.0.1:8000",
    timeout_seconds=1800,
)

Path("$WORKSPACE/output/example.json").write_text(
    result.model_dump_json(indent=2),
    encoding="utf-8",
)
```

返回结构是 `MinerUPdfResult`，核心字段：

| 字段 | 说明 |
| --- | --- |
| `source_pdf` | 输入 PDF 路径 |
| `source_file_name` | 输入文件名 |
| `page_count` | 页数 |
| `coord_space` | 坐标系，默认 `mineru_layout` |
| `parsed_pdf.pages[].text_blocks[]` | 页级文本块 |
| `parsed_pdf.pages[].table_blocks[]` | 页级表格块 |
| `pages[]` | 带 `text`、`text_blocks`、`table_blocks`、`page_meta`、`layout_ref` 的页面结果 |
| `artifacts[]` | MinerU 原始产物引用，如 `middle_json`、`content_list_json`、`markdown` |

## 2. 文件夹批处理

目录模式默认使用 Daft。输入目录里放 PDF，输出目录会生成一对一 JSON：

```bash
cd $WORKSPACE/platform-core-public-feature-foundation-base-operator
export MINERU_API_URL=http://127.0.0.1:8000

python scripts/run-daft-batch-operate.py \
  input \
  --output-dir output \
  --concurrency 2 \
  --timeout-seconds 1800
```

输出示例：

```text
output/
├── example.json
├── example/                  # MinerU 原始产物目录
├── batch_report.json
├── failed_files.json
└── failed_files.jsonl
```

单个 PDF 也走同一个脚本：

```bash
python scripts/run-daft-batch-operate.py input/example.pdf --output-dir output
```

## 3. HTTP API

HTTP 调用推荐传服务器本地 PDF 路径和输出目录。响应只返回生成路径和摘要，完整 OCR JSON 写入 `output_dir`：

```bash
curl -X POST http://127.0.0.1:8001/api/v1/operators/mineru/layout-extract-path \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "$WORKSPACE/input/example.pdf",
    "output_dir": "$WORKSPACE/output"
  }'
```

也支持上传文件，同时指定输出目录：

```bash
curl \
  -F "files=@input/example.pdf" \
  -F "output_dir=$WORKSPACE/output" \
  http://127.0.0.1:8001/api/v1/operators/mineru/layout-extract
```

多文件上传：

```bash
curl \
  -F "files=@input/a.pdf" \
  -F "files=@input/b.pdf" \
  -F "output_dir=$WORKSPACE/output" \
  http://127.0.0.1:8001/api/v1/operators/mineru/layout-extract
```

响应示例：

```json
{
  "success": true,
  "data": {
    "source_pdf": "$WORKSPACE/input/example.pdf",
    "source_file_name": "example.pdf",
    "success": true,
    "json_path": "$WORKSPACE/output/example.json",
    "artifact_dir": "$WORKSPACE/output/example",
    "page_count": 10,
    "text_block_count": 81,
    "table_block_count": 0,
    "elapsed_seconds": 7.03
  },
  "error": null
}
```

## 4. 服务约定

- 默认 MinerU 原生 API 地址是 `http://127.0.0.1:8000`。
- 当前封装 HTTP API 地址是 `http://127.0.0.1:8001`。
- 脚本和 Python 入口默认复用常驻 MinerU API，不再为每个 PDF 临时启动本地服务。
- 如果服务部署在其他地址，设置 `MINERU_API_URL` 或传 `--api-url`。
- Daft 只负责目录批处理调度；单 PDF 抽取仍是 `extract_pdf()`。
