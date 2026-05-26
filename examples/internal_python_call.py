from platform_foundation.ocr import extract_pdf_dir, extract_pdf_file


pdf_result = extract_pdf_file(
    "/home/kaifang/mineru_workspace/data/input/5.pdf",
    output_dir="/home/kaifang/mineru_workspace/output_single_example",
    table_engine="paddle",
    overwrite=True,
)
print(pdf_result.json_path)

batch_report = extract_pdf_dir(
    "/home/kaifang/mineru_workspace/data/input",
    output_dir="/home/kaifang/mineru_workspace/output_batch_example",
    table_engine="paddle",
    concurrency=2,
    recursive=False,
    limit=None,
    overwrite=False,
    resume=True,
)
print(batch_report.batch_report_path)
print(batch_report.batch_report_csv_path)
