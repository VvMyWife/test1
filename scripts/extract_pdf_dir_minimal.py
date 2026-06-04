import os
from pathlib import Path

from platform_foundation.ocr import extract_pdf_dir


WORKSPACE = Path(
    os.environ.get("MINERU_WORKSPACE")
    or os.environ.get("WORKSPACE")
    or Path(__file__).resolve().parents[1]
).expanduser().resolve()
TABLE_ENGINE = os.environ.get("TABLE_ENGINE", "ocr").strip().lower()
CONCURRENCY = int(os.environ.get("CONCURRENCY", "12"))
DEFAULT_INPUT_DIR = WORKSPACE / "input"
if not DEFAULT_INPUT_DIR.exists():
    DEFAULT_INPUT_DIR = WORKSPACE / "data" / "input"
INPUT_DIR = Path(os.environ.get("INPUT_DIR", str(DEFAULT_INPUT_DIR))).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(WORKSPACE / "output" / f"{TABLE_ENGINE}_{CONCURRENCY}"),
    )
).expanduser().resolve()

os.environ.setdefault("MINERU_API_URL", "http://127.0.0.1:18000")
if TABLE_ENGINE == "paddle":
    os.environ.setdefault("PADDLE_TABLE_API_URL", "http://127.0.0.1:18200")

report = extract_pdf_dir(
    INPUT_DIR,
    output_dir=OUTPUT_DIR,
    table_engine=TABLE_ENGINE,
    concurrency=CONCURRENCY,
    recursive=False,
    limit=None,
    overwrite=True,
    resume=False,
    paddle_operator_max_inflight=None,
    ocr_operator_max_inflight=None,
)

print("batch_report_path:", report.batch_report_path)
print("batch_report_csv_path:", report.batch_report_csv_path)
print("success_count:", report.success_count)
print("failure_count:", report.failure_count)
print("skipped_count:", report.skipped_count)
print("pdf_count:", report.pdf_count)
print("page_count:", report.page_count)
print("pages_per_second:", report.pages_per_second)
print("total_elapsed_seconds:", report.total_elapsed_seconds)
print("engine:", report.engine)
print("table_engine:", report.table_engine)
print("concurrency:", report.concurrency)
print("mineru_extra_args:", report.mineru_extra_args)
