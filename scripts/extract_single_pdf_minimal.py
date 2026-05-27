import os
from pathlib import Path

from platform_foundation.ocr import extract_pdf_file


WORKSPACE = Path(
    os.environ.get("MINERU_WORKSPACE")
    or os.environ.get("WORKSPACE")
    or Path(__file__).resolve().parents[1]
).expanduser().resolve()
TABLE_ENGINE = os.environ.get("TABLE_ENGINE", "ocr").strip().lower()
DEFAULT_INPUT_PDF = WORKSPACE / "input" / "5.pdf"
if not DEFAULT_INPUT_PDF.exists():
    DEFAULT_INPUT_PDF = WORKSPACE / "data" / "input" / "5.pdf"
INPUT_PDF = Path(os.environ.get("INPUT_PDF", str(DEFAULT_INPUT_PDF))).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", str(WORKSPACE / "output" / f"single_{TABLE_ENGINE}"))
).expanduser().resolve()

os.environ.setdefault("MINERU_API_URL", "http://127.0.0.1:8000")
if TABLE_ENGINE == "paddle":
    os.environ.setdefault("PADDLE_TABLE_API_URL", "http://127.0.0.1:8200")

result = extract_pdf_file(
    INPUT_PDF,
    output_dir=OUTPUT_DIR,
    table_engine=TABLE_ENGINE,
    overwrite=True,
)

print("success:", result.success)
print("json_path:", result.json_path)
print("error_report:", result.error_report)
print("table_engine:", result.table_engine)
print("mineru_extra_args:", result.mineru_extra_args)
print("table_cell_count:", result.table_cell_count)
