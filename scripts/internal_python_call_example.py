from __future__ import annotations

import os
from pathlib import Path

from platform_foundation.ocr import PdfDirExtractReport, PdfFileExtractResult
from platform_foundation.ocr import extract_pdf_dir, extract_pdf_file


# This file intentionally has no argparse and no sys.path modification.
# It shows how another Python project should call the installed internal package.

os.environ.setdefault("MINERU_API_URL", "http://127.0.0.1:18000")
os.environ.setdefault("PADDLE_TABLE_API_URL", "http://127.0.0.1:18200")

WORKSPACE = Path(__file__).resolve().parents[1]
INPUT_PDF = WORKSPACE / "data" / "input" / "5.pdf"
INPUT_DIR = WORKSPACE / "data" / "input"
SINGLE_OUTPUT_DIR = WORKSPACE / "output_python_single_example"
BATCH_OUTPUT_DIR = WORKSPACE / "output_python_batch_example"


def extract_one_pdf() -> PdfFileExtractResult:
    result = extract_pdf_file(
        INPUT_PDF,
        output_dir=SINGLE_OUTPUT_DIR,
        table_engine="paddle",
        overwrite=True,
        field_keywords=["身份证号"],
    )

    print("single.success:", result.success)
    print("single.json_path:", result.json_path)
    print("single.error_report:", result.error_report)
    print("single.table_cell_count:", result.table_cell_count)
    print("single.field_coordinates_path:", result.field_coordinates_path)
    print("single.field_annotation_pdf_path:", result.field_annotation_pdf_path)
    return result


def extract_pdf_folder() -> PdfDirExtractReport:
    report = extract_pdf_dir(
        INPUT_DIR,
        output_dir=BATCH_OUTPUT_DIR,
        table_engine="ocr",
        concurrency=2,
        recursive=False,
        limit=2,
        overwrite=True,
        resume=False,
    )

    print("batch.batch_report_path:", report.batch_report_path)
    print("batch.batch_report_csv_path:", report.batch_report_csv_path)
    print("batch.success_count:", report.success_count)
    print("batch.failure_count:", report.failure_count)
    print("batch.pdf_count:", report.pdf_count)
    print("batch.page_count:", report.page_count)
    print("batch.seconds_per_page:", report.seconds_per_page)
    print("batch.engine:", report.engine)
    print("batch.output_dir:", report.output_dir)
    return report


if __name__ == "__main__":
    extract_one_pdf()
    extract_pdf_folder()
