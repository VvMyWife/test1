import os
from pathlib import Path

from platform_foundation.ocr import extract_pdf_file


WORKSPACE = Path(
    os.environ.get("MINERU_WORKSPACE")
    or os.environ.get("WORKSPACE")
    or Path(__file__).resolve().parents[1]
).expanduser().resolve()
TABLE_ENGINE = os.environ.get("TABLE_ENGINE", "ocr").strip().lower()
ENABLE_PAGE_SCREENSHOTS = os.environ.get("ENABLE_PAGE_SCREENSHOTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
PAGE_SCREENSHOT_DPI = int(os.environ.get("PAGE_SCREENSHOT_DPI", "144"))
FIELD_KEYWORDS = [
    item.strip()
    for item in os.environ.get("FIELD_KEYWORDS", "").replace("，", ",").replace("；", ",").split(",")
    if item.strip()
]
DEFAULT_INPUT_PDF = WORKSPACE / "input" / "5.pdf"
if not DEFAULT_INPUT_PDF.exists():
    DEFAULT_INPUT_PDF = WORKSPACE / "data" / "input" / "5.pdf"
INPUT_PDF = Path(os.environ.get("INPUT_PDF", str(DEFAULT_INPUT_PDF))).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", str(WORKSPACE / "output" / f"single_{TABLE_ENGINE}"))
).expanduser().resolve()

os.environ.setdefault("MINERU_API_URL", "http://127.0.0.1:18000")
if TABLE_ENGINE == "paddle":
    os.environ.setdefault("PADDLE_TABLE_API_URL", "http://127.0.0.1:18200")

result = extract_pdf_file(
    INPUT_PDF,
    output_dir=OUTPUT_DIR,
    table_engine=TABLE_ENGINE,
    overwrite=True,
    enable_page_screenshots=ENABLE_PAGE_SCREENSHOTS,
    page_screenshot_dpi=PAGE_SCREENSHOT_DPI,
    field_keywords=FIELD_KEYWORDS,
)

print("success:", result.success)
print("json_path:", result.json_path)
print("error_report:", result.error_report)
print("table_engine:", result.table_engine)
print("mineru_extra_args:", result.mineru_extra_args)
print("table_cell_count:", result.table_cell_count)
print("input_type:", result.input_type)
print("converted_pdf_path:", result.converted_pdf_path)
print("page_screenshots_manifest:", result.page_screenshots_manifest)
print("field_keywords:", result.field_keywords)
print("field_match_count:", result.field_match_count)
print("field_coordinates_path:", result.field_coordinates_path)
print("field_annotation_pdf_path:", result.field_annotation_pdf_path)
