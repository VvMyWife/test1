# Internal Python Package Install

This project exposes `platform_foundation` as an editable internal Python package.

## Package Root

```bash
/home/kaifang/mineru_workspace/platform-core-public-feature-foundation-base-operator/backend/foundation
```

## Development Install

```bash
cd /home/kaifang/mineru_workspace
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mineru312
cd /home/kaifang/mineru_workspace/platform-core-public-feature-foundation-base-operator/backend/foundation
python -m pip install -e .
```

If `conda activate` already works in your shell, this is also fine:

```bash
cd /home/kaifang/mineru_workspace
conda activate mineru312
cd /home/kaifang/mineru_workspace/platform-core-public-feature-foundation-base-operator/backend/foundation
python -m pip install -e .
```

## Build Tool Bootstrap

If build tools are missing, install or upgrade them with the Tsinghua mirror:

```bash
python -m pip install -U build setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Verify Import

This should work from any directory after the editable install:

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
    "/home/kaifang/mineru_workspace/data/input/5.pdf",
    output_dir="/home/kaifang/mineru_workspace/output",
    table_engine="paddle",
)
print(result.json_path)
```

## Minimal Directory Call

```python
from platform_foundation.ocr import extract_pdf_dir

report = extract_pdf_dir(
    "/home/kaifang/mineru_workspace/data/input",
    output_dir="/home/kaifang/mineru_workspace/output_batch",
    table_engine="paddle",
    concurrency=2,
)
print(report.batch_report_path)
print(report.batch_report_csv_path)
```

## Resident Paddle Table API

Start the resident Paddle table service once:

```bash
cd /home/kaifang/mineru_workspace
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mineru312
bash scripts/start-paddle-table-api.sh
export PADDLE_TABLE_API_URL=http://127.0.0.1:8200
```

Check service health:

```bash
curl http://127.0.0.1:8200/health
```

After `PADDLE_TABLE_API_URL` is exported, normal calls keep the same API:

```python
from platform_foundation.ocr import extract_pdf_file

result = extract_pdf_file(
    "/home/kaifang/mineru_workspace/data/input/5.pdf",
    output_dir="/home/kaifang/mineru_workspace/output",
    table_engine="paddle",
)
print(result.json_path)
```

Stop the resident service:

```bash
bash scripts/stop-paddle-table-api.sh
```

Minimal runnable scripts:

```bash
python scripts/extract_single_pdf_minimal.py
python scripts/extract_pdf_dir_minimal.py
```
