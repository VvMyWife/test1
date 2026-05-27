import os
from pathlib import Path

from platform_foundation.ocr import extract_pdf_dir, extract_pdf_file


workspace = Path(os.environ.get("MINERU_WORKSPACE", Path.cwd())).expanduser().resolve()
input_dir = Path(os.environ.get("INPUT_DIR", workspace / "input")).expanduser().resolve()
input_pdf = Path(os.environ.get("INPUT_PDF", input_dir / "5.pdf")).expanduser().resolve()
output_dir = Path(os.environ.get("OUTPUT_DIR", workspace / "output")).expanduser().resolve()

os.environ.setdefault("MINERU_API_URL", "http://127.0.0.1:8000")
os.environ.setdefault("PADDLE_TABLE_API_URL", "http://127.0.0.1:8200")


pdf_result = extract_pdf_file(
    input_pdf,
    output_dir=output_dir / "single_paddle_example",
    table_engine="paddle",
    overwrite=True,
)
print(pdf_result.json_path)

batch_report = extract_pdf_dir(
    input_dir,
    output_dir=output_dir / "batch_paddle_example",
    table_engine="paddle",
    concurrency=2,
    recursive=False,
    limit=None,
    overwrite=False,
    resume=True,
)
print(batch_report.batch_report_path)
print(batch_report.batch_report_csv_path)
