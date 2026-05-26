import os
from pathlib import Path

from platform_foundation.ocr import extract_pdf_dir


WORKSPACE = Path(__file__).resolve().parents[1]

os.environ.setdefault("MINERU_API_URL", "http://127.0.0.1:8000")
os.environ.setdefault("PADDLE_TABLE_API_URL", "http://127.0.0.1:8200")

# Fast directory smoke test. Change table_engine to "paddle" after starting the Paddle API.
report = extract_pdf_dir(
    WORKSPACE / "data" / "input",
    output_dir=WORKSPACE / "output_dir_ocr_minimal",
    table_engine="ocr",
    concurrency=2,
    recursive=False,
    limit=2,
    overwrite=True,
    resume=False,
)

print("batch_report_path:", report.batch_report_path)
print("batch_report_csv_path:", report.batch_report_csv_path)
print("success_count:", report.success_count)
print("failure_count:", report.failure_count)
print("pdf_count:", report.pdf_count)
print("page_count:", report.page_count)
print("pages_per_second:", report.pages_per_second)
print("engine:", report.engine)
print("table_engine:", report.table_engine)
print("mineru_extra_args:", report.mineru_extra_args)
