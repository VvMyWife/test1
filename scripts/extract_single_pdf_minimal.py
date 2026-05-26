from pathlib import Path

from platform_foundation.ocr import extract_pdf_file


WORKSPACE = Path(__file__).resolve().parents[1]

# Fast path: pure MinerU OCR. No Paddle table API or operator max_inflight cap is required.
result = extract_pdf_file(
    WORKSPACE / "data" / "input" / "5.pdf",
    output_dir=WORKSPACE / "output_single_ocr_minimal",
    table_engine="ocr",
    overwrite=True,
)

print("success:", result.success)
print("json_path:", result.json_path)
print("error_report:", result.error_report)
print("table_engine:", result.table_engine)
print("mineru_extra_args:", result.mineru_extra_args)
print("table_cell_count:", result.table_cell_count)
