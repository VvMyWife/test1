# Internal Python Package Install

Docker 是推荐交付方式。只有在裸机 Python 环境里直接开发或调试时，才需要安装内部包。

## Package Root

```bash
cd "$WORKSPACE/platform-core-public-feature-foundation-base-operator/backend/foundation"
```

其中 `WORKSPACE` 是你自己的工作目录，例如：

```bash
export WORKSPACE="$(pwd)"
```

## Development Install

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mineru312
cd "$WORKSPACE/platform-core-public-feature-foundation-base-operator/backend/foundation"
python -m pip install -e .
```

If build tools are missing:

```bash
python -m pip install -U build setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Verify Import

```bash
python -c "from platform_foundation.ocr import extract_pdf_file, extract_pdf_dir; print('import ok')"
```

## Verify Function Signatures

```bash
python - <<'PY'
from platform_foundation.ocr import extract_pdf_file, extract_pdf_dir
import inspect
print(inspect.signature(extract_pdf_file))
print(inspect.signature(extract_pdf_dir))
PY
```

## Minimal Single-PDF Call

```python
from platform_foundation.ocr import extract_pdf_file

result = extract_pdf_file(
    "data/input/5.pdf",
    output_dir="output/single",
    table_engine="paddle",
)
print(result.json_path)
```

## Minimal Directory Call

```python
from platform_foundation.ocr import extract_pdf_dir

report = extract_pdf_dir(
    "data/input",
    output_dir="output/batch",
    table_engine="paddle",
    concurrency=2,
)
print(report.batch_report_path)
print(report.batch_report_csv_path)
```

## Minimal Runnable Scripts

```bash
python scripts/extract_single_pdf_minimal.py
python scripts/extract_pdf_dir_minimal.py
```
