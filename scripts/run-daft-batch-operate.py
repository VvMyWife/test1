from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from platform_foundation.ocr import extract_pdf_dir, extract_pdf_file

WORKSPACE = Path(os.environ.get("WORKSPACE", str(Path.cwd()))).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one-to-one MinerU JSON files from a PDF/image or a directory."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="PDF/image file or directory. Defaults to input/ under the workspace when present.",
    )
    parser.add_argument("--input-dir", default=None, help="Directory containing PDF/image files.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated JSON files.")
    parser.add_argument(
        "--table-engine",
        default=os.environ.get("MINERU_TABLE_ENGINE", "ocr"),
        choices=("ocr", "paddle"),
        help="Use ocr for pure MinerU, or paddle for PaddleOCR table refinement.",
    )
    parser.add_argument(
        "--use-paddle-tables",
        action="store_true",
        help="Shortcut for --table-engine paddle.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("MINERU_CONCURRENCY", "1")),
        help="Maximum input files processed at the same time in directory mode.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for input files.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max input file count in directory mode.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even when the target JSON already exists.",
    )
    parser.add_argument(
        "--enable-page-screenshots",
        action="store_true",
        default=_env_bool("ENABLE_PAGE_SCREENSHOTS", default=False),
        help="Export full-page PNG screenshots under each artifact directory.",
    )
    parser.add_argument(
        "--page-screenshot-dpi",
        type=int,
        default=int(os.environ.get("PAGE_SCREENSHOT_DPI", "144")),
        help="DPI for --enable-page-screenshots.",
    )
    parser.add_argument(
        "--field-keywords",
        action="append",
        default=None,
        help="Comma-separated or repeated keywords to locate in Paddle table cells and annotate in a PDF.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input, args.input_dir)
    output_dir = _resolve_output_dir(args.output_dir)
    table_engine = "paddle" if args.use_paddle_tables else args.table_engine
    field_keywords = _parse_field_keywords(args.field_keywords)

    if input_path.is_file():
        result = extract_pdf_file(
            input_path,
            output_dir=output_dir,
            table_engine=table_engine,
            overwrite=True,
            enable_page_screenshots=args.enable_page_screenshots,
            page_screenshot_dpi=args.page_screenshot_dpi,
            field_keywords=field_keywords,
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), flush=True)
        if result.success:
            print(f"\nMINERU_JSON={result.json_path}", flush=True)
        else:
            print(f"\nMINERU_ERROR={result.error_report}", flush=True)
            raise SystemExit(1)
        return

    report = extract_pdf_dir(
        input_path,
        output_dir=output_dir,
        table_engine=table_engine,
        concurrency=args.concurrency,
        recursive=args.recursive,
        limit=args.limit if args.limit > 0 else None,
        overwrite=args.overwrite,
        resume=not args.overwrite,
        enable_page_screenshots=args.enable_page_screenshots,
        page_screenshot_dpi=args.page_screenshot_dpi,
        field_keywords=field_keywords,
    )
    print(f"MINERU_BATCH_REPORT={report.batch_report_path}", flush=True)
    print(f"MINERU_BATCH_CSV={report.batch_report_csv_path}", flush=True)
    print(
        json.dumps(
            {
                "success_count": report.success_count,
                "failure_count": report.failure_count,
                "skipped_count": report.skipped_count,
                "pdf_count": report.pdf_count,
                "page_count": report.page_count,
                "seconds_per_page": report.seconds_per_page,
                "field_match_count": report.field_match_count,
                "engine": report.engine,
                "output_dir": report.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if report.failure_count:
        raise SystemExit(1)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_field_keywords(raw_values: list[str] | None) -> list[str]:
    values = list(raw_values or [])
    env_value = os.environ.get("FIELD_KEYWORDS")
    if env_value:
        values.append(env_value)
    keywords: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in value.replace("，", ",").replace("；", ",").replace(";", ",").split(","):
            keyword = part.strip()
            if not keyword:
                continue
            dedupe_key = "".join(keyword.split()).casefold()
            if dedupe_key not in seen:
                keywords.append(keyword)
                seen.add(dedupe_key)
    return keywords


def _resolve_input_path(raw_input: str | None, raw_input_dir: str | None) -> Path:
    if raw_input_dir:
        return Path(raw_input_dir).expanduser().resolve()
    if raw_input:
        return Path(raw_input).expanduser().resolve()

    for candidate in (
        Path(os.environ["MINERU_INPUT_DIR"]).expanduser() if os.environ.get("MINERU_INPUT_DIR") else None,
        Path.cwd() / "input",
        WORKSPACE / "input",
        WORKSPACE / "data" / "input",
    ):
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / "input").resolve()


def _resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir).expanduser().resolve()
    if os.environ.get("MINERU_OUTPUT_DIR"):
        return Path(os.environ["MINERU_OUTPUT_DIR"]).expanduser().resolve()
    return (WORKSPACE / "output").resolve()


if __name__ == "__main__":
    main()
