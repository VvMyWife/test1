from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import asyncio
import csv
import difflib
import html as html_lib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import ArtifactRef, BoundingBox
from ..operators import LayoutExtractMinerUOperator, LayoutExtractMinerUPaddleTableOperator
from .pure_mineru import (
    DEFAULT_MINERU_API_URL,
    DEFAULT_TIMEOUT_SECONDS,
    dump_pure_mineru_json,
    extract_pdf,
)

JsonDict = dict[str, Any]
DEFAULT_OCR_OPERATOR_MAX_INFLIGHT: int | None = None
DEFAULT_PADDLE_OPERATOR_MAX_INFLIGHT: int | None = None
SUPPORTED_PDF_SUFFIXES = {".pdf"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_PDF_SUFFIXES | SUPPORTED_IMAGE_SUFFIXES
DEFAULT_PAGE_SCREENSHOT_DPI = 144
GENERIC_FLAT_INPUT_DIR_NAMES = {"input", "inputs"}
INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


class PdfFileExtractResult(BaseModel):
    """Single-PDF extraction result with persisted artifact locations and metrics."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: str
    source_file_name: str
    success: bool
    json_path: str | None = None
    artifact_dir: str
    error_report: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    error: str | None = None
    error_meta: dict[str, Any] | None = None
    skipped: bool = False
    elapsed_seconds: float | None = Field(default=None, ge=0)
    processing_seconds: float | None = Field(default=None, ge=0)
    queue_wait_seconds: float = Field(default=0.0, ge=0)
    wall_elapsed_seconds: float | None = Field(default=None, ge=0)
    page_count: int = Field(default=0, ge=0)
    text_block_count: int = Field(default=0, ge=0)
    table_block_count: int = Field(default=0, ge=0)
    table_cell_count: int = Field(default=0, ge=0)
    table_engine: str = "ocr"
    paddle_table_mode: str | None = None
    paddle_table_artifact: str | None = None
    mineru_backend: str = "pipeline"
    mineru_parse_method: str = "auto"
    mineru_lang: str = "ch"
    mineru_extra_args: list[str] = Field(default_factory=list)
    api_url: str = DEFAULT_MINERU_API_URL
    input_type: str = "pdf"
    converted_pdf_path: str | None = None
    page_screenshots_manifest: str | None = None
    field_keywords: list[str] = Field(default_factory=list)
    field_match_count: int = Field(default=0, ge=0)
    field_coordinates_path: str | None = None
    field_annotation_pdf_path: str | None = None
    relative_input_path: str | None = None
    output_relative_dir: str | None = None


class PdfDirExtractReport(BaseModel):
    """Directory batch extraction report persisted beside the output JSON files."""

    model_config = ConfigDict(extra="forbid")

    engine: str
    input: str
    output_dir: str
    pdf_count: int = Field(ge=0)
    api_url: str
    concurrency: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    mineru_backend: str
    mineru_parse_method: str
    mineru_lang: str
    mineru_extra_args: list[str] = Field(default_factory=list)
    table_engine: str
    paddle_table_mode: str | None = None
    paddle_device: str | None = None
    page_count: int = Field(default=0, ge=0)
    seconds_per_page: float = Field(default=0.0, ge=0)
    field_match_count: int = Field(default=0, ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    skipped_count: int = Field(default=0, ge=0)
    batch_report_path: str
    batch_report_csv_path: str
    failed_files: str
    failed_files_jsonl: str
    total_elapsed_seconds: float = Field(ge=0)
    items: list[PdfFileExtractResult] = Field(default_factory=list)
    enable_page_screenshots: bool = False


class MinerUPdfFileOperator:
    """High-level single-PDF operator: validate, extract, persist JSON/error, return metrics."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        parse_method: str = "auto",
        backend: str = "pipeline",
        lang: str = "ch",
        formula_enable: bool = False,
        mineru_table_enable: bool | None = None,
        operator_factory: Callable[[], LayoutExtractMinerUOperator] | None = None,
        ocr_operator_max_inflight: int | None = DEFAULT_OCR_OPERATOR_MAX_INFLIGHT,
        paddle_operator_max_inflight: int | None = DEFAULT_PADDLE_OPERATOR_MAX_INFLIGHT,
    ) -> None:
        self.api_url = _resolve_api_url(api_url)
        self.timeout_seconds = timeout_seconds
        self.parse_method = parse_method
        self.backend = backend
        self.lang = lang
        self.formula_enable = formula_enable
        self.mineru_table_enable = mineru_table_enable
        self.operator_factory = operator_factory
        self.ocr_operator_max_inflight = _validate_positive_int_or_none(
            ocr_operator_max_inflight,
            option_name="ocr_operator_max_inflight",
        )
        self.paddle_operator_max_inflight = _validate_positive_int_or_none(
            paddle_operator_max_inflight,
            option_name="paddle_operator_max_inflight",
        )

    def extract_file(
        self,
        pdf_path: str | Path,
        *,
        output_dir: str | Path,
        use_paddle_tables: bool | None = None,
        table_engine: str | None = None,
        paddle_table_mode: str = "ppstructurev3",
        paddle_device: str | None = None,
        source_file_name: str | None = None,
        overwrite: bool = True,
        enable_page_screenshots: bool = False,
        page_screenshot_dpi: int = DEFAULT_PAGE_SCREENSHOT_DPI,
        field_keywords: Sequence[str] | str | None = None,
        queue_wait_seconds: float = 0.0,
        extra_args: Sequence[str] | None = None,
        mineru_options: dict[str, Any] | None = None,
    ) -> PdfFileExtractResult:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_input = Path(pdf_path).expanduser().resolve()
        resolved_source_name = source_file_name or resolved_input.name
        output_stem = _safe_stem(resolved_source_name or resolved_input.name)
        artifact_dir = output_root / output_stem
        json_path = artifact_dir / f"{output_stem}.json"
        error_path = artifact_dir / f"{output_stem}.error.json"

        resolved_table_engine = _resolve_table_engine(
            use_paddle_tables=use_paddle_tables,
            table_engine=table_engine,
        )
        resolved_extra_args = (
            list(extra_args) if extra_args is not None else self._default_extra_args(resolved_table_engine)
        )

        if overwrite:
            _unlink_if_exists(json_path)
            _unlink_if_exists(error_path)
            _remove_path_if_exists(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        input_type = "pdf"
        converted_pdf_path: Path | None = None
        page_screenshots_manifest: str | None = None
        resolved_field_keywords = _normalize_field_keywords(field_keywords)
        field_coordinates_path: str | None = None
        field_annotation_pdf_path: str | None = None
        field_match_count = 0
        try:
            self._validate_input_path(resolved_input)
            pdf_for_mineru = resolved_input
            if _is_image_file(resolved_input):
                input_type = "image"

            options = dict(mineru_options or {})
            options.update(
                {
                    "table_engine": resolved_table_engine,
                    "backend": self.backend,
                    "parse_method": self.parse_method,
                    "lang": self.lang,
                    "extra_args": resolved_extra_args,
                }
            )
            if resolved_table_engine == "paddle":
                options["paddle_table_mode"] = paddle_table_mode
            if paddle_device:
                options["paddle_device"] = paddle_device

            parsed = extract_pdf(
                pdf_for_mineru,
                output_dir=artifact_dir,
                api_url=self.api_url,
                timeout_seconds=self.timeout_seconds,
                parse_method=self.parse_method,
                backend=self.backend,
                lang=self.lang,
                extra_args=resolved_extra_args,
                table_engine=resolved_table_engine,
                paddle_table_mode=paddle_table_mode,
                paddle_device=paddle_device,
                mineru_options=options,
                operator_factory=self._operator_factory_for(resolved_table_engine),
            )
            parsed = _flatten_mineru_output_dir(
                parsed,
                artifact_dir=artifact_dir,
                mineru_pdf_path=pdf_for_mineru,
            )
            if resolved_table_engine == "paddle" and _result_layout_provider(parsed) != "paddle":
                parsed = _merge_paddle_artifact_text_blocks(parsed)
            parsed = _apply_ocr_text_postprocessing(parsed)
            parsed = _sync_page_text_from_structured_content(parsed)
            if resolved_table_engine == "paddle":
                parsed = _rewrite_markdown_artifacts_from_parsed(
                    parsed,
                    artifact_dir=artifact_dir,
                    output_stem=output_stem,
                )
            _rewrite_readable_artifacts_with_text_postprocessing(artifact_dir=artifact_dir)
            artifacts = list(parsed.artifacts)
            if converted_pdf_path is not None:
                artifacts.append(
                    ArtifactRef(
                        kind="converted_pdf",
                        uri=str(converted_pdf_path),
                        meta={
                            "source_path": str(resolved_input),
                            "source_file_name": resolved_source_name,
                            "source_type": input_type,
                        },
                    )
                )
            if enable_page_screenshots:
                screenshot_artifact = _render_pdf_page_screenshots(
                    pdf_for_mineru,
                    output_dir=artifact_dir / "page_screenshots",
                    page_count=parsed.page_count,
                    dpi=page_screenshot_dpi,
                )
                artifacts.append(screenshot_artifact)
                page_screenshots_manifest = screenshot_artifact.uri
            if resolved_field_keywords:
                field_artifacts = _write_field_coordinate_artifacts(
                    parsed,
                    keywords=resolved_field_keywords,
                    source_pdf_path=resolved_input,
                    annotation_pdf_path=pdf_for_mineru,
                    output_dir=artifact_dir,
                    output_stem=output_stem,
                    table_engine=resolved_table_engine,
                )
                artifacts.extend(field_artifacts["artifacts"])
                field_coordinates_path = field_artifacts["coordinates_path"]
                field_annotation_pdf_path = field_artifacts["annotation_pdf_path"]
                field_match_count = int(field_artifacts["match_count"])

            parsed = parsed.model_copy(
                update={
                    "source_pdf": str(resolved_input),
                    "source_file_name": resolved_source_name,
                    "artifacts": artifacts,
                }
            )
            json_path.write_text(dump_pure_mineru_json(parsed, indent=2), encoding="utf-8")
            processing_seconds = time.perf_counter() - started
            return PdfFileExtractResult(
                pdf_path=str(resolved_input),
                source_file_name=resolved_source_name,
                success=True,
                json_path=str(json_path),
                artifact_dir=str(artifact_dir),
                elapsed_seconds=round(processing_seconds, 3),
                processing_seconds=round(processing_seconds, 3),
                queue_wait_seconds=round(queue_wait_seconds, 3),
                wall_elapsed_seconds=round(processing_seconds + queue_wait_seconds, 3),
                page_count=parsed.page_count,
                text_block_count=sum(len(page.text_blocks) for page in parsed.parsed_pdf.pages),
                table_block_count=sum(len(page.table_blocks) for page in parsed.parsed_pdf.pages),
                table_cell_count=sum(
                    len(table.cells)
                    for page in parsed.parsed_pdf.pages
                    for table in page.table_blocks
                ),
                table_engine=resolved_table_engine,
                paddle_table_mode=paddle_table_mode if resolved_table_engine == "paddle" else None,
                paddle_table_artifact=_find_artifact_uri(parsed.artifacts, "paddle_table_json"),
                mineru_backend=self.backend,
                mineru_parse_method=self.parse_method,
                mineru_lang=self.lang,
                mineru_extra_args=resolved_extra_args,
                api_url=self.api_url,
                input_type=input_type,
                converted_pdf_path=str(converted_pdf_path) if converted_pdf_path is not None else None,
                page_screenshots_manifest=page_screenshots_manifest,
                field_keywords=resolved_field_keywords,
                field_match_count=field_match_count,
                field_coordinates_path=field_coordinates_path,
                field_annotation_pdf_path=field_annotation_pdf_path,
            )
        except Exception as exc:
            processing_seconds = time.perf_counter() - started
            result = PdfFileExtractResult(
                pdf_path=str(resolved_input),
                source_file_name=resolved_source_name,
                success=False,
                json_path=None,
                artifact_dir=str(artifact_dir),
                error_report=str(error_path),
                error_type=type(exc).__name__,
                error_code=getattr(exc, "code", None),
                error=str(exc),
                error_meta=getattr(exc, "detail", None),
                elapsed_seconds=round(processing_seconds, 3),
                processing_seconds=round(processing_seconds, 3),
                queue_wait_seconds=round(queue_wait_seconds, 3),
                wall_elapsed_seconds=round(processing_seconds + queue_wait_seconds, 3),
                table_engine=resolved_table_engine,
                paddle_table_mode=paddle_table_mode if resolved_table_engine == "paddle" else None,
                mineru_backend=self.backend,
                mineru_parse_method=self.parse_method,
                mineru_lang=self.lang,
                mineru_extra_args=resolved_extra_args,
                api_url=self.api_url,
                input_type=input_type,
                converted_pdf_path=str(converted_pdf_path) if converted_pdf_path is not None else None,
                page_screenshots_manifest=page_screenshots_manifest,
                field_keywords=resolved_field_keywords,
                field_match_count=field_match_count,
                field_coordinates_path=field_coordinates_path,
                field_annotation_pdf_path=field_annotation_pdf_path,
            )
            error_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result

    def _default_extra_args(self, table_engine: str) -> list[str]:
        mineru_table_enable = self._resolve_mineru_table_enable(table_engine)
        return [
            "--formula",
            _bool_cli_value(self.formula_enable),
            "--table",
            _bool_cli_value(mineru_table_enable),
        ]

    def _resolve_mineru_table_enable(self, table_engine: str) -> bool:
        if self.mineru_table_enable is not None:
            return self.mineru_table_enable
        return table_engine in {"ocr", "paddle"}

    def _operator_factory_for(self, table_engine: str) -> Callable[[], LayoutExtractMinerUOperator]:
        if self.operator_factory is not None:
            return self.operator_factory
        if table_engine == "paddle":
            return lambda: LayoutExtractMinerUPaddleTableOperator(
                max_inflight=self.paddle_operator_max_inflight,
                recommended_max_concurrency=self.paddle_operator_max_inflight,
            )
        return lambda: LayoutExtractMinerUOperator(
            max_inflight=self.ocr_operator_max_inflight,
            recommended_max_concurrency=self.ocr_operator_max_inflight,
        )

    @staticmethod
    def _validate_input_path(input_path: Path) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")
        if input_path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
            raise ValueError(f"Input file must be one of {supported}: {input_path}")


class MinerUPdfDirBatchOperator:
    """High-level directory operator: scan, resume, run concurrently, persist batch reports."""

    def __init__(
        self,
        *,
        file_operator: MinerUPdfFileOperator | None = None,
        api_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        parse_method: str = "auto",
        backend: str = "pipeline",
        lang: str = "ch",
        formula_enable: bool = False,
        mineru_table_enable: bool | None = None,
        operator_factory: Callable[[], LayoutExtractMinerUOperator] | None = None,
        ocr_operator_max_inflight: int | None = DEFAULT_OCR_OPERATOR_MAX_INFLIGHT,
        paddle_operator_max_inflight: int | None = DEFAULT_PADDLE_OPERATOR_MAX_INFLIGHT,
    ) -> None:
        self.file_operator = file_operator or MinerUPdfFileOperator(
            api_url=api_url,
            timeout_seconds=timeout_seconds,
            parse_method=parse_method,
            backend=backend,
            lang=lang,
            formula_enable=formula_enable,
            mineru_table_enable=mineru_table_enable,
            operator_factory=operator_factory,
            ocr_operator_max_inflight=ocr_operator_max_inflight,
            paddle_operator_max_inflight=paddle_operator_max_inflight,
        )

    def extract_dir(
        self,
        input_dir: str | Path,
        *,
        output_dir: str | Path,
        use_paddle_tables: bool | None = None,
        table_engine: str | None = None,
        concurrency: int = 1,
        recursive: bool = False,
        limit: int | None = None,
        resume: bool = True,
        overwrite: bool = False,
        enable_page_screenshots: bool = False,
        page_screenshot_dpi: int = DEFAULT_PAGE_SCREENSHOT_DPI,
        field_keywords: Sequence[str] | str | None = None,
        paddle_table_mode: str = "ppstructurev3",
        paddle_device: str | None = None,
        engine: str = "auto",
        lock_acquire_timeout_seconds: float = 3600.0,
    ) -> PdfDirExtractReport:
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")

        input_root = Path(input_dir).expanduser().resolve()
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        pdfs = _list_input_files(input_root, recursive=recursive)
        if limit is not None and limit > 0:
            pdfs = pdfs[:limit]
        if not pdfs:
            supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
            raise ValueError(f"No supported input files found in {input_root}; supported: {supported}")

        started = time.perf_counter()
        resolved_table_engine = _resolve_table_engine(
            use_paddle_tables=use_paddle_tables,
            table_engine=table_engine,
        )
        selected_engine = _resolve_batch_engine(engine=engine, concurrency=concurrency)
        resolved_field_keywords = _normalize_field_keywords(field_keywords)
        tasks = [
            self._build_task(
                pdf,
                input_root=input_root,
                output_root=output_root,
                table_engine=resolved_table_engine,
                paddle_table_mode=paddle_table_mode,
                paddle_device=paddle_device,
                resume=resume,
                overwrite=overwrite,
                enable_page_screenshots=enable_page_screenshots,
                page_screenshot_dpi=page_screenshot_dpi,
                field_keywords=resolved_field_keywords,
            )
            for pdf in pdfs
        ]

        if selected_engine == "sequential":
            items = [self._run_task(task, None, 1, lock_acquire_timeout_seconds) for task in tasks]
        elif selected_engine == "thread":
            items = self._run_threaded(tasks, output_root, concurrency, lock_acquire_timeout_seconds)
        else:
            items = self._run_daft(tasks, output_root, concurrency, lock_acquire_timeout_seconds)

        return self._write_report(
            input_root=input_root,
            output_root=output_root,
            engine=selected_engine,
            concurrency=concurrency,
            table_engine=resolved_table_engine,
            paddle_table_mode=paddle_table_mode,
            paddle_device=paddle_device,
            enable_page_screenshots=enable_page_screenshots,
            started_at=started,
            items=items,
        )

    def _build_task(
        self,
        pdf: Path,
        *,
        input_root: Path,
        output_root: Path,
        table_engine: str,
        paddle_table_mode: str,
        paddle_device: str | None,
        resume: bool,
        overwrite: bool,
        enable_page_screenshots: bool,
        page_screenshot_dpi: int,
        field_keywords: Sequence[str],
    ) -> "_BatchTask":
        task_output_dir, output_relative_dir = _output_dir_for_input_file(
            pdf,
            input_root=input_root,
            output_root=output_root,
        )
        return _BatchTask(
            pdf_path=pdf,
            source_file_name=pdf.name,
            relative_input_path=_relative_input_path(pdf, input_root=input_root),
            output_dir=task_output_dir,
            output_relative_dir=output_relative_dir,
            table_engine=table_engine,
            paddle_table_mode=paddle_table_mode,
            paddle_device=paddle_device,
            resume=resume,
            overwrite=overwrite,
            enable_page_screenshots=enable_page_screenshots,
            page_screenshot_dpi=page_screenshot_dpi,
            field_keywords=list(field_keywords),
        )

    def _run_task(
        self,
        task: "_BatchTask",
        lock_dir: Path | None,
        lock_limit: int,
        lock_acquire_timeout_seconds: float,
    ) -> PdfFileExtractResult:
        if task.resume and not task.overwrite:
            existing = _result_from_existing_json(
                pdf_path=task.pdf_path,
                source_file_name=task.source_file_name,
                output_dir=task.output_dir,
                table_engine=task.table_engine,
                paddle_table_mode=task.paddle_table_mode,
                file_operator=self.file_operator,
                field_keywords=task.field_keywords,
                relative_input_path=task.relative_input_path,
                output_relative_dir=task.output_relative_dir,
            )
            if existing is not None:
                return existing

        wait_started = time.perf_counter()
        with _filesystem_concurrency_slot(
            lock_dir=lock_dir,
            limit=lock_limit,
            timeout_seconds=lock_acquire_timeout_seconds,
        ):
            queue_wait_seconds = time.perf_counter() - wait_started
            result = self.file_operator.extract_file(
                task.pdf_path,
                output_dir=task.output_dir,
                table_engine=task.table_engine,
                paddle_table_mode=task.paddle_table_mode,
                paddle_device=task.paddle_device,
                source_file_name=task.source_file_name,
                overwrite=True,
                enable_page_screenshots=task.enable_page_screenshots,
                page_screenshot_dpi=task.page_screenshot_dpi,
                field_keywords=task.field_keywords,
                queue_wait_seconds=queue_wait_seconds,
            )
            return result.model_copy(
                update={
                    "relative_input_path": task.relative_input_path,
                    "output_relative_dir": task.output_relative_dir,
                }
            )

    def _run_threaded(
        self,
        tasks: list["_BatchTask"],
        output_root: Path,
        concurrency: int,
        lock_acquire_timeout_seconds: float,
    ) -> list[PdfFileExtractResult]:
        lock_dir = output_root / ".mineru_concurrency_locks"
        results: list[PdfFileExtractResult | None] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {
                executor.submit(self._run_task, task, lock_dir, concurrency, lock_acquire_timeout_seconds): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
        return [item for item in results if item is not None]

    def _run_daft(
        self,
        tasks: list["_BatchTask"],
        output_root: Path,
        concurrency: int,
        lock_acquire_timeout_seconds: float,
    ) -> list[PdfFileExtractResult]:
        daft = _import_daft()
        extract_udf = _build_daft_udf(daft, concurrency)
        lock_dir = output_root / ".mineru_concurrency_locks"
        df = daft.from_pydict(
            {
                "pdf_path": [str(task.pdf_path) for task in tasks],
                "source_file_name": [task.source_file_name for task in tasks],
                "relative_input_path": [task.relative_input_path for task in tasks],
                "output_dir": [str(task.output_dir) for task in tasks],
                "output_relative_dir": [task.output_relative_dir or "" for task in tasks],
                "table_engine": [task.table_engine for task in tasks],
                "paddle_table_mode": [task.paddle_table_mode for task in tasks],
                "paddle_device": [task.paddle_device or "" for task in tasks],
                "enable_page_screenshots": [task.enable_page_screenshots for task in tasks],
                "page_screenshot_dpi": [task.page_screenshot_dpi for task in tasks],
                "field_keywords_json": [
                    json.dumps(task.field_keywords, ensure_ascii=False) for task in tasks
                ],
                "resume": [task.resume for task in tasks],
                "overwrite": [task.overwrite for task in tasks],
                "api_url": [self.file_operator.api_url for _ in tasks],
                "timeout_seconds": [float(self.file_operator.timeout_seconds) for _ in tasks],
                "parse_method": [self.file_operator.parse_method for _ in tasks],
                "backend": [self.file_operator.backend for _ in tasks],
                "lang": [self.file_operator.lang for _ in tasks],
                "formula_enable": [self.file_operator.formula_enable for _ in tasks],
                "mineru_table_enable": [
                    self.file_operator._resolve_mineru_table_enable(task.table_engine) for task in tasks
                ],
                "ocr_operator_max_inflight": [
                    _int_or_zero(self.file_operator.ocr_operator_max_inflight) for _ in tasks
                ],
                "paddle_operator_max_inflight": [
                    _int_or_zero(self.file_operator.paddle_operator_max_inflight) for _ in tasks
                ],
                "lock_dir": [str(lock_dir) for _ in tasks],
                "lock_limit": [int(concurrency) for _ in tasks],
                "lock_acquire_timeout_seconds": [float(lock_acquire_timeout_seconds) for _ in tasks],
            }
        )
        result_df = df.with_column(
            "result_json",
            extract_udf(
                daft.col("pdf_path"),
                daft.col("source_file_name"),
                daft.col("relative_input_path"),
                daft.col("output_dir"),
                daft.col("output_relative_dir"),
                daft.col("table_engine"),
                daft.col("paddle_table_mode"),
                daft.col("paddle_device"),
                daft.col("enable_page_screenshots"),
                daft.col("page_screenshot_dpi"),
                daft.col("field_keywords_json"),
                daft.col("resume"),
                daft.col("overwrite"),
                daft.col("api_url"),
                daft.col("timeout_seconds"),
                daft.col("parse_method"),
                daft.col("backend"),
                daft.col("lang"),
                daft.col("formula_enable"),
                daft.col("mineru_table_enable"),
                daft.col("ocr_operator_max_inflight"),
                daft.col("paddle_operator_max_inflight"),
                daft.col("lock_dir"),
                daft.col("lock_limit"),
                daft.col("lock_acquire_timeout_seconds"),
            ),
        ).collect()
        rows = result_df.to_pydict()
        return [PdfFileExtractResult.model_validate_json(raw) for raw in rows["result_json"]]

    def _write_report(
        self,
        *,
        input_root: Path,
        output_root: Path,
        engine: str,
        concurrency: int,
        table_engine: str,
        paddle_table_mode: str,
        paddle_device: str | None,
        enable_page_screenshots: bool,
        started_at: float,
        items: list[PdfFileExtractResult],
    ) -> PdfDirExtractReport:
        failed_items = [item for item in items if not item.success]
        batch_report_path = output_root / "batch_report.json"
        batch_report_csv_path = output_root / "batch_report.csv"
        failed_files_path = output_root / "failed_files.json"
        failed_files_jsonl_path = output_root / "failed_files.jsonl"
        failed_files_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in failed_items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        failed_files_jsonl_path.write_text(
            "".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n" for item in failed_items),
            encoding="utf-8",
        )
        _write_batch_csv(batch_report_csv_path, items)
        total_elapsed_seconds = round(time.perf_counter() - started_at, 3)
        page_count = sum(item.page_count for item in items if item.success)
        seconds_per_page = round(total_elapsed_seconds / page_count, 3) if page_count > 0 else 0.0
        report = PdfDirExtractReport(
            engine=engine,
            input=str(input_root),
            output_dir=str(output_root),
            pdf_count=len(items),
            api_url=self.file_operator.api_url,
            concurrency=concurrency,
            timeout_seconds=self.file_operator.timeout_seconds,
            mineru_backend=self.file_operator.backend,
            mineru_parse_method=self.file_operator.parse_method,
            mineru_lang=self.file_operator.lang,
            mineru_extra_args=self.file_operator._default_extra_args(table_engine),
            table_engine=table_engine,
            paddle_table_mode=paddle_table_mode if table_engine == "paddle" else None,
            paddle_device=paddle_device,
            page_count=page_count,
            seconds_per_page=seconds_per_page,
            field_match_count=sum(item.field_match_count for item in items if item.success),
            success_count=sum(1 for item in items if item.success),
            failure_count=len(failed_items),
            skipped_count=sum(1 for item in items if item.skipped),
            batch_report_path=str(batch_report_path),
            batch_report_csv_path=str(batch_report_csv_path),
            failed_files=str(failed_files_path),
            failed_files_jsonl=str(failed_files_jsonl_path),
            total_elapsed_seconds=total_elapsed_seconds,
            items=items,
            enable_page_screenshots=enable_page_screenshots,
        )
        batch_report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report


class _BatchTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pdf_path: Path
    source_file_name: str
    relative_input_path: str | None = None
    output_dir: Path
    output_relative_dir: str | None = None
    table_engine: str
    paddle_table_mode: str
    paddle_device: str | None
    enable_page_screenshots: bool = False
    page_screenshot_dpi: int = DEFAULT_PAGE_SCREENSHOT_DPI
    field_keywords: list[str] = Field(default_factory=list)
    resume: bool
    overwrite: bool


def extract_pdf_file(
    pdf_path: str | Path,
    *,
    output_dir: str | Path,
    use_paddle_tables: bool | None = None,
    table_engine: str | None = None,
    concurrency: int | None = None,
    **kwargs: Any,
) -> PdfFileExtractResult:
    """Extract one PDF to one JSON file. Public callers should not need lower-level MinerU details."""

    del concurrency
    operator_kwargs = _pop_operator_kwargs(kwargs)
    return MinerUPdfFileOperator(**operator_kwargs).extract_file(
        pdf_path,
        output_dir=output_dir,
        use_paddle_tables=use_paddle_tables,
        table_engine=table_engine,
        **kwargs,
    )


def extract_pdf_dir(
    input_dir: str | Path,
    *,
    output_dir: str | Path,
    use_paddle_tables: bool | None = None,
    table_engine: str | None = None,
    concurrency: int = 1,
    **kwargs: Any,
) -> PdfDirExtractReport:
    """Extract every PDF in a directory and write batch reports."""

    operator_kwargs = _pop_operator_kwargs(kwargs)
    return MinerUPdfDirBatchOperator(**operator_kwargs).extract_dir(
        input_dir,
        output_dir=output_dir,
        use_paddle_tables=use_paddle_tables,
        table_engine=table_engine,
        concurrency=concurrency,
        **kwargs,
    )


def _pop_operator_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    operator_keys = {
        "api_url",
        "timeout_seconds",
        "parse_method",
        "backend",
        "lang",
        "formula_enable",
        "mineru_table_enable",
        "operator_factory",
        "ocr_operator_max_inflight",
        "paddle_operator_max_inflight",
    }
    return {key: kwargs.pop(key) for key in list(kwargs) if key in operator_keys}


def _extract_file_for_daft(
    pdf_path: str,
    source_file_name: str,
    relative_input_path: str | None,
    output_dir: str,
    output_relative_dir: str,
    table_engine: str,
    paddle_table_mode: str,
    paddle_device: str,
    enable_page_screenshots: bool,
    page_screenshot_dpi: int,
    field_keywords_json: str,
    resume: bool,
    overwrite: bool,
    api_url: str,
    timeout_seconds: float,
    parse_method: str,
    backend: str,
    lang: str,
    formula_enable: bool,
    mineru_table_enable: bool,
    ocr_operator_max_inflight: int,
    paddle_operator_max_inflight: int,
    lock_dir: str,
    lock_limit: int,
    lock_acquire_timeout_seconds: float,
) -> str:
    operator = MinerUPdfDirBatchOperator(
        file_operator=MinerUPdfFileOperator(
            api_url=api_url,
            timeout_seconds=timeout_seconds,
            parse_method=parse_method,
            backend=backend,
            lang=lang,
            formula_enable=formula_enable,
            mineru_table_enable=mineru_table_enable,
            ocr_operator_max_inflight=_zero_to_none(ocr_operator_max_inflight),
            paddle_operator_max_inflight=_zero_to_none(paddle_operator_max_inflight),
        )
    )
    task = _BatchTask(
        pdf_path=Path(pdf_path),
        source_file_name=source_file_name,
        relative_input_path=relative_input_path or None,
        output_dir=Path(output_dir),
        output_relative_dir=output_relative_dir or None,
        table_engine=table_engine,
        paddle_table_mode=paddle_table_mode,
        paddle_device=paddle_device or None,
        enable_page_screenshots=bool(enable_page_screenshots),
        page_screenshot_dpi=int(page_screenshot_dpi),
        field_keywords=_load_field_keywords_json(field_keywords_json),
        resume=resume,
        overwrite=overwrite,
    )
    result = operator._run_task(task, Path(lock_dir), lock_limit, lock_acquire_timeout_seconds)
    return result.model_dump_json()


def _build_daft_udf(daft: Any, max_concurrency: int) -> Any:
    @daft.func(return_dtype=daft.DataType.string(), max_concurrency=max_concurrency)
    async def extract_file_udf(  # noqa: PLR0913
        pdf_path: str,
        source_file_name: str,
        relative_input_path: str,
        output_dir: str,
        output_relative_dir: str,
        table_engine: str,
        paddle_table_mode: str,
        paddle_device: str,
        enable_page_screenshots: bool,
        page_screenshot_dpi: int,
        field_keywords_json: str,
        resume: bool,
        overwrite: bool,
        api_url: str,
        timeout_seconds: float,
        parse_method: str,
        backend: str,
        lang: str,
        formula_enable: bool,
        mineru_table_enable: bool,
        ocr_operator_max_inflight: int,
        paddle_operator_max_inflight: int,
        lock_dir: str,
        lock_limit: int,
        lock_acquire_timeout_seconds: float,
    ) -> str:
        return await asyncio.to_thread(
            _extract_file_for_daft,
            pdf_path,
            source_file_name,
            relative_input_path,
            output_dir,
            output_relative_dir,
            table_engine,
            paddle_table_mode,
            paddle_device,
            enable_page_screenshots,
            page_screenshot_dpi,
            field_keywords_json,
            resume,
            overwrite,
            api_url,
            timeout_seconds,
            parse_method,
            backend,
            lang,
            formula_enable,
            mineru_table_enable,
            ocr_operator_max_inflight,
            paddle_operator_max_inflight,
            lock_dir,
            lock_limit,
            lock_acquire_timeout_seconds,
        )

    return extract_file_udf


def _resolve_table_engine(*, use_paddle_tables: bool | None, table_engine: str | None) -> str:
    if table_engine is None:
        resolved = "paddle" if use_paddle_tables else "ocr"
    else:
        resolved = table_engine.strip().lower()
    if resolved not in {"ocr", "paddle"}:
        raise ValueError("table_engine must be 'ocr' or 'paddle'")
    return resolved


def _resolve_batch_engine(*, engine: str, concurrency: int) -> str:
    normalized = engine.strip().lower()
    if normalized == "auto":
        if concurrency <= 1:
            return "sequential"
        return "daft" if _daft_available() else "thread"
    if normalized not in {"sequential", "thread", "daft"}:
        raise ValueError("engine must be 'auto', 'sequential', 'thread', or 'daft'")
    return normalized


def _result_from_existing_json(
    *,
    pdf_path: Path,
    source_file_name: str,
    output_dir: Path,
    table_engine: str,
    paddle_table_mode: str,
    file_operator: MinerUPdfFileOperator,
    field_keywords: Sequence[str] | None = None,
    relative_input_path: str | None = None,
    output_relative_dir: str | None = None,
) -> PdfFileExtractResult | None:
    output_stem = _safe_stem(source_file_name or pdf_path.name)
    artifact_dir = output_dir / output_stem
    json_path = artifact_dir / f"{output_stem}.json"
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return None
    text_block_count = sum(len(page.get("text_blocks") or []) for page in pages if isinstance(page, dict))
    table_blocks = [
        table
        for page in pages
        if isinstance(page, dict)
        for table in (page.get("table_blocks") or [])
        if isinstance(table, dict)
    ]
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    converted_pdf_path = _artifact_uri(artifacts, "converted_pdf")
    page_screenshots_manifest = _artifact_uri(artifacts, "page_screenshots_manifest")
    field_coordinates_path = _artifact_uri(artifacts, "field_coordinates_json")
    field_annotation_pdf_path = _artifact_uri(artifacts, "field_annotation_pdf")
    resolved_field_keywords = _normalize_field_keywords(field_keywords)
    field_summary = _read_field_coordinate_summary(field_coordinates_path)
    if resolved_field_keywords:
        existing_keywords = _normalize_field_keywords(field_summary.get("field_keywords"))
        if (
            existing_keywords != resolved_field_keywords
            or not field_coordinates_path
            or not Path(field_coordinates_path).exists()
            or not field_annotation_pdf_path
            or not Path(field_annotation_pdf_path).exists()
        ):
            return None
    return PdfFileExtractResult(
        pdf_path=str(pdf_path),
        source_file_name=source_file_name,
        success=True,
        json_path=str(json_path),
        artifact_dir=str(artifact_dir),
        skipped=True,
        elapsed_seconds=0.0,
        processing_seconds=0.0,
        queue_wait_seconds=0.0,
        wall_elapsed_seconds=0.0,
        page_count=len(pages),
        text_block_count=text_block_count,
        table_block_count=len(table_blocks),
        table_cell_count=sum(len(table.get("cells") or []) for table in table_blocks),
        table_engine=table_engine,
        paddle_table_mode=paddle_table_mode if table_engine == "paddle" else None,
        mineru_backend=file_operator.backend,
        mineru_parse_method=file_operator.parse_method,
        mineru_lang=file_operator.lang,
        mineru_extra_args=file_operator._default_extra_args(table_engine),
        api_url=file_operator.api_url,
        input_type="image" if _is_image_file(pdf_path) else "pdf",
        converted_pdf_path=converted_pdf_path,
        page_screenshots_manifest=page_screenshots_manifest,
        field_keywords=resolved_field_keywords,
        field_match_count=int(field_summary.get("match_count") or 0),
        field_coordinates_path=field_coordinates_path,
        field_annotation_pdf_path=field_annotation_pdf_path,
        relative_input_path=relative_input_path,
        output_relative_dir=output_relative_dir,
    )


def _list_input_files(input_root: Path, *, recursive: bool) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
            raise ValueError(f"Input file must be one of {supported}: {input_root}")
        return [input_root]
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"Input path is not a directory: {input_root}")
    seen: set[str] = set()
    inputs: list[Path] = []
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_root.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            continue
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        inputs.append(resolved)
    return inputs


def _relative_input_path(input_path: Path, *, input_root: Path) -> str:
    if input_root.is_file():
        return input_path.name
    try:
        return input_path.relative_to(input_root).as_posix()
    except ValueError:
        return input_path.name


def _output_dir_for_input_file(
    input_path: Path,
    *,
    input_root: Path,
    output_root: Path,
) -> tuple[Path, str | None]:
    if input_root.is_file():
        return output_root, None

    try:
        relative_parent = input_path.relative_to(input_root).parent
    except ValueError:
        relative_parent = Path()
    output_stem = _safe_stem(input_path.name)

    if relative_parent == Path("."):
        if input_root.name.strip().lower() in GENERIC_FLAT_INPUT_DIR_NAMES:
            return output_root, None
        if _safe_path_part(input_root.name) == output_stem:
            return output_root, None
        output_relative_dir = Path(_safe_path_part(input_root.name))
    else:
        output_relative_dir = _safe_relative_dir(relative_parent)
        if output_relative_dir.name == output_stem:
            output_relative_dir = output_relative_dir.parent

    if not output_relative_dir.parts:
        return output_root, None
    return output_root / output_relative_dir, output_relative_dir.as_posix()


def _safe_relative_dir(relative_dir: Path) -> Path:
    safe_parts = [
        _safe_path_part(part)
        for part in relative_dir.parts
        if part not in {"", "."}
    ]
    safe_parts = [part for part in safe_parts if part]
    if not safe_parts:
        return Path()
    return Path(*safe_parts)


def _safe_path_part(value: str) -> str:
    safe = INVALID_PATH_CHARS_RE.sub("_", value).strip(" ._")
    return safe or "unnamed"


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _convert_image_to_pdf(image_path: Path, output_pdf_path: Path) -> None:
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to convert image inputs to PDF") from exc

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        converted = ImageOps.exif_transpose(image)
        if converted.mode in {"RGBA", "LA"} or (
            converted.mode == "P" and "transparency" in converted.info
        ):
            converted = converted.convert("RGBA")
            background = Image.new("RGB", converted.size, (255, 255, 255))
            background.paste(converted, mask=converted.getchannel("A"))
            converted = background
        elif converted.mode != "RGB":
            converted = converted.convert("RGB")
        converted.save(output_pdf_path, "PDF", resolution=300.0)


def _merge_paddle_artifact_text_blocks(result: Any) -> Any:
    """Promote Paddle full-page text/table blocks into the final structured result.

    Paddle PPStructureV3 often recognizes useful non-table text in
    ``parsing_res_list`` even when the MinerU/Paddle table-cell merge has no
    corresponding table cell. Treat those Paddle blocks as additional text
    candidates so JSON, Markdown, and field-coordinate lookup all see them.
    """

    artifact_uri = _find_artifact_uri(list(getattr(result, "artifacts", []) or []), "paddle_table_json")
    if not artifact_uri:
        return result
    artifact_path = Path(str(artifact_uri))
    if not artifact_path.exists():
        return result
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result

    candidates = _extract_paddle_artifact_text_candidates(payload, result)
    if not candidates or not hasattr(result, "model_dump"):
        return result
    seal_regions_by_page = _extract_paddle_artifact_layout_regions(
        payload,
        result,
        labels={"seal", "stamp"},
    )

    data = result.model_dump(mode="python")
    pages = data.get("pages")
    if not isinstance(pages, list):
        return result

    candidates_by_page: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        page_index = _to_int(candidate.get("page_index"))
        if page_index is None:
            continue
        candidates_by_page.setdefault(page_index, []).append(candidate)

    changed = False
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_index = _to_int(page.get("page_index")) or 0
        page_candidates = candidates_by_page.get(page_index, [])
        if not page_candidates:
            continue

        text_blocks = [
            block
            for block in (page.get("text_blocks") or [])
            if isinstance(block, dict)
        ]
        page_changed = False
        if _annotate_paddle_seal_blocks(
            text_blocks,
            seal_regions=seal_regions_by_page.get(page_index, []),
            paddle_candidates=page_candidates,
        ):
            changed = True
            page_changed = True
        existing_texts = [
            str(block.get("text") or "")
            for block in text_blocks
            if str(block.get("text") or "").strip()
        ]
        if str(page.get("text") or "").strip():
            existing_texts.append(str(page.get("text") or ""))

        for candidate in page_candidates:
            text = str(candidate.get("text") or "").strip()
            if not text:
                continue
            merge_action = _merge_paddle_candidate_by_layout(candidate, text_blocks)
            if merge_action == "updated":
                existing_texts.append(text)
                changed = True
                page_changed = True
                continue
            if merge_action == "skip":
                continue
            if _is_redundant_text(text, existing_texts):
                continue
            if _is_paddle_overall_ocr_candidate(candidate):
                continue
            text_blocks.append(
                {
                    "text": text,
                    "bounding_box": candidate["bounding_box"],
                    "block_type": candidate.get("block_type") or "paddle_text",
                    "confidence": candidate.get("confidence"),
                    "meta": candidate.get("meta") or {},
                }
            )
            existing_texts.append(text)
            changed = True
            page_changed = True

        if page_changed:
            text_blocks = _sort_text_block_dicts(text_blocks)
            page["text_blocks"] = text_blocks
            page["text"] = _compose_page_text_from_block_dicts(text_blocks)

    if not changed:
        return result

    parsed_pdf = data.get("parsed_pdf")
    if isinstance(parsed_pdf, dict) and isinstance(parsed_pdf.get("pages"), list):
        page_block_map = {
            int(page.get("page_index") or 0): list(page.get("text_blocks") or [])
            for page in pages
            if isinstance(page, dict)
        }
        for parsed_page in parsed_pdf["pages"]:
            if not isinstance(parsed_page, dict):
                continue
            page_index = _to_int(parsed_page.get("page_index")) or 0
            if page_index in page_block_map:
                parsed_page["text_blocks"] = page_block_map[page_index]

    return result.__class__.model_validate(data)


def _extract_paddle_artifact_text_candidates(payload: Mapping[str, Any], result: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_page_sizes = _result_page_sizes(result)

    pages = payload.get("pages")
    if isinstance(pages, list):
        for fallback_page_index, raw_page in enumerate(pages):
            page_payload = _unwrap_paddle_page_payload(raw_page)
            if not isinstance(page_payload, Mapping):
                continue
            page_index = _to_int(page_payload.get("page_index"))
            if page_index is None:
                page_index = fallback_page_index
            scale_x, scale_y = _paddle_page_scale(page_payload, target_page_sizes.get(page_index))
            for block_index, block in enumerate(_as_mapping_list(page_payload.get("parsing_res_list"))):
                text = _paddle_block_text(block)
                bbox = _coerce_paddle_bbox(
                    block.get("block_bbox") or block.get("bbox"),
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                if not text or bbox is None:
                    continue
                label = str(block.get("block_label") or "text").strip() or "text"
                candidates.append(
                    {
                        "page_index": page_index,
                        "text": text,
                        "bounding_box": bbox,
                        "block_type": f"paddle_{label}",
                        "confidence": _to_float(block.get("score") or block.get("confidence")),
                        "meta": {
                            "source": "paddleocr_ppstructurev3",
                            "paddle_artifact_source": "parsing_res_list",
                            "block_label": label,
                            "block_id": block.get("block_id"),
                            "block_order": block.get("block_order"),
                            "paddle_index": block_index,
                            "coord_space": "mineru_layout",
                        },
                    }
                )
            candidates.extend(
                _extract_paddle_overall_ocr_candidates(
                    page_payload,
                    page_index=page_index,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            )

    for raw_index, item in enumerate(_as_mapping_list(payload.get("raw_results"))):
        candidate = item.get("candidate")
        raw_result = item.get("result")
        if not isinstance(candidate, Mapping) or not isinstance(raw_result, Mapping):
            continue
        page_index = _to_int(candidate.get("page_index"))
        if page_index is None:
            continue
        text = _paddle_block_text(_unwrap_paddle_page_payload(raw_result))
        bbox = _coerce_paddle_bbox(candidate.get("bbox"), scale_x=1.0, scale_y=1.0)
        if not text or bbox is None:
            continue
        candidates.append(
            {
                "page_index": page_index,
                "text": text,
                "bounding_box": bbox,
                "block_type": "paddle_table",
                "confidence": _to_float(raw_result.get("structure_score")),
                "meta": {
                    "source": "paddleocr_table_structure",
                    "paddle_artifact_source": "raw_results",
                    "paddle_index": raw_index,
                    "image_uri": candidate.get("image_uri"),
                    "coord_space": "mineru_layout",
                },
            }
        )

    return candidates


def _extract_paddle_overall_ocr_candidates(
    page_payload: Mapping[str, Any],
    *,
    page_index: int,
    scale_x: float,
    scale_y: float,
) -> list[dict[str, Any]]:
    overall = page_payload.get("overall_ocr_res")
    if not isinstance(overall, Mapping):
        return []
    texts = overall.get("rec_texts")
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        return []
    scores = overall.get("rec_scores")
    boxes = overall.get("rec_boxes")
    if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
        boxes = overall.get("dt_polys") or overall.get("rec_polys")
    if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
        return []

    candidates: list[dict[str, Any]] = []
    page_height = _to_float(page_payload.get("height") or page_payload.get("source_height"))
    for index, raw_text in enumerate(texts):
        text = _normalize_ocr_output_text(str(raw_text or "")).strip()
        if not text:
            continue
        raw_box = boxes[index] if index < len(boxes) else None
        bbox = _coerce_paddle_bbox(raw_box, scale_x=scale_x, scale_y=scale_y)
        if bbox is None:
            continue
        score = None
        if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)) and index < len(scores):
            score = _to_float(scores[index])
        block_type = "page_number" if _looks_like_page_number_candidate(text, bbox, page_height) else "paddle_ocr_text"
        candidates.append(
            {
                "page_index": page_index,
                "text": text,
                "bounding_box": bbox,
                "block_type": block_type,
                "confidence": score,
                "meta": {
                    "source": "paddleocr_ppstructurev3",
                    "paddle_artifact_source": "overall_ocr_res",
                    "paddle_index": index,
                    "coord_space": "mineru_layout",
                },
            }
        )
    return candidates


def _extract_paddle_artifact_layout_regions(
    payload: Mapping[str, Any],
    result: Any,
    *,
    labels: set[str],
) -> dict[int, list[dict[str, Any]]]:
    regions_by_page: dict[int, list[dict[str, Any]]] = {}
    target_page_sizes = _result_page_sizes(result)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return regions_by_page
    normalized_labels = {label.strip().lower() for label in labels}
    for fallback_page_index, raw_page in enumerate(pages):
        page_payload = _unwrap_paddle_page_payload(raw_page)
        if not isinstance(page_payload, Mapping):
            continue
        page_index = _to_int(page_payload.get("page_index"))
        if page_index is None:
            page_index = fallback_page_index
        scale_x, scale_y = _paddle_page_scale(page_payload, target_page_sizes.get(page_index))
        for block_index, block in enumerate(_as_mapping_list(page_payload.get("parsing_res_list"))):
            label = str(block.get("block_label") or "").strip().lower()
            if label not in normalized_labels:
                continue
            bbox = _coerce_paddle_bbox(
                block.get("block_bbox") or block.get("bbox"),
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if bbox is None:
                continue
            text = _paddle_block_text(block)
            regions_by_page.setdefault(page_index, []).append(
                {
                    "label": label,
                    "bounding_box": bbox,
                    "text": text,
                    "paddle_index": block_index,
                    "block_id": block.get("block_id"),
                    "block_order": block.get("block_order"),
                }
            )
    return regions_by_page


def _unwrap_paddle_page_payload(value: Any) -> Mapping[str, Any] | Any:
    if isinstance(value, Mapping):
        res = value.get("res")
        if isinstance(res, Mapping):
            return res
        output = value.get("output")
        if isinstance(output, Mapping):
            return output
    return value


def _paddle_block_text(block: Any) -> str:
    if not isinstance(block, Mapping):
        return ""
    raw_text = (
        block.get("block_content")
        or block.get("pred_html")
        or block.get("html")
        or block.get("text")
        or block.get("rec_text")
    )
    if not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()
    if not text:
        return ""
    if _looks_like_html_table(text):
        return ""
    if "<" in text and ">" in text:
        text = _html_to_plain_text(text)
    return _normalize_ocr_output_text(text).strip()


def _looks_like_html_table(value: str) -> bool:
    return bool(re.search(r"(?is)<\s*table\b", value))


def _html_to_plain_text(value: str) -> str:
    text = value
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*tr\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*t[dh]\s*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _result_page_sizes(result: Any) -> dict[int, tuple[float | None, float | None]]:
    sizes: dict[int, tuple[float | None, float | None]] = {}
    parsed_pdf = getattr(result, "parsed_pdf", None)
    pages = getattr(parsed_pdf, "pages", []) if parsed_pdf is not None else []
    for page in pages or []:
        page_index = _to_int(getattr(page, "page_index", None))
        if page_index is None:
            continue
        image_size = getattr(page, "image_size", None)
        width = _to_float(getattr(image_size, "width", None)) if image_size is not None else None
        height = _to_float(getattr(image_size, "height", None)) if image_size is not None else None
        max_x, max_y = _page_coordinate_extents(page)
        if width and height and (max_x > width * 1.05 or max_y > height * 1.05):
            width = None
            height = None
        sizes[page_index] = (width, height)
    return sizes


def _page_coordinate_extents(page: Any) -> tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for block in getattr(page, "text_blocks", []) or []:
        box = getattr(block, "bounding_box", None)
        if box is None:
            continue
        max_x = max(max_x, float(getattr(box, "x", 0)) + float(getattr(box, "w", 0)))
        max_y = max(max_y, float(getattr(box, "y", 0)) + float(getattr(box, "h", 0)))
    for table in getattr(page, "table_blocks", []) or []:
        table_box = getattr(table, "bounding_box", None)
        if table_box is not None:
            max_x = max(max_x, float(getattr(table_box, "x", 0)) + float(getattr(table_box, "w", 0)))
            max_y = max(max_y, float(getattr(table_box, "y", 0)) + float(getattr(table_box, "h", 0)))
        for cell in getattr(table, "cells", []) or []:
            box = getattr(cell, "bounding_box", None)
            if box is None:
                continue
            max_x = max(max_x, float(getattr(box, "x", 0)) + float(getattr(box, "w", 0)))
            max_y = max(max_y, float(getattr(box, "y", 0)) + float(getattr(box, "h", 0)))
    return max_x, max_y


def _paddle_page_scale(
    payload: Mapping[str, Any],
    target_size: tuple[float | None, float | None] | None,
) -> tuple[float, float]:
    source_width = _to_float(payload.get("width") or payload.get("source_width"))
    source_height = _to_float(payload.get("height") or payload.get("source_height"))
    target_width, target_height = target_size or (None, None)
    if source_width and source_height and target_width and target_height:
        return float(target_width) / float(source_width), float(target_height) / float(source_height)
    return 1.0, 1.0


def _coerce_paddle_bbox(value: Any, *, scale_x: float, scale_y: float) -> dict[str, int] | None:
    if isinstance(value, Mapping):
        if {"x", "y", "w", "h"}.issubset(value):
            x = _to_float(value.get("x"))
            y = _to_float(value.get("y"))
            w = _to_float(value.get("w"))
            h = _to_float(value.get("h"))
            if x is None or y is None or w is None or h is None:
                return None
            return {
                "x": int(round(x * scale_x)),
                "y": int(round(y * scale_y)),
                "w": max(0, int(round(w * scale_x))),
                "h": max(0, int(round(h * scale_y))),
            }
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    numbers = [_to_float(item) for item in list(value)[:4]]
    if any(item is None for item in numbers):
        return None
    x0, y0, x1, y1 = [float(item) for item in numbers if item is not None]
    if x1 < x0 or y1 < y0:
        xs = [x0, x1]
        ys = [y0, y1]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    return {
        "x": int(round(x0 * scale_x)),
        "y": int(round(y0 * scale_y)),
        "w": max(0, int(round((x1 - x0) * scale_x))),
        "h": max(0, int(round((y1 - y0) * scale_y))),
    }


def _looks_like_page_number_candidate(
    text: str,
    bbox: Mapping[str, Any],
    page_height: float | None,
) -> bool:
    if not re.fullmatch(r"\d{1,4}", text.strip()):
        return False
    if page_height is None or page_height <= 0:
        return False
    y = _bbox_dict_value(bbox, "y")
    h = _bbox_dict_value(bbox, "h")
    center_y = y + h / 2.0
    return center_y <= page_height * 0.12 or center_y >= page_height * 0.88


def _apply_ocr_text_postprocessing(result: Any) -> Any:
    if not hasattr(result, "model_dump"):
        return result
    data = result.model_dump(mode="python")
    rewritten = _rewrite_ocr_text_values(data)
    return result.__class__.model_validate(rewritten)


def _rewrite_ocr_text_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_ocr_text_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_ocr_text_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_ocr_text_values(item) for item in value)
    if isinstance(value, str):
        return _normalize_ocr_output_text(value)
    return value


def _sync_page_text_from_structured_content(result: Any) -> Any:
    if not hasattr(result, "model_dump"):
        return result
    data = result.model_dump(mode="python")
    pages = data.get("pages")
    if not isinstance(pages, list):
        return result
    changed = False
    for page in pages:
        if not isinstance(page, dict):
            continue
        text = _compose_page_text_from_page_dict(page)
        if text != page.get("text"):
            page["text"] = text
            changed = True
    parsed_pdf = data.get("parsed_pdf")
    parsed_pages = parsed_pdf.get("pages") if isinstance(parsed_pdf, dict) else None
    if isinstance(parsed_pages, list):
        pages_by_index = {
            _to_int(page.get("page_index")) or 0: page
            for page in pages
            if isinstance(page, dict)
        }
        for parsed_page in parsed_pages:
            if not isinstance(parsed_page, dict):
                continue
            page_index = _to_int(parsed_page.get("page_index")) or 0
            source_page = pages_by_index.get(page_index)
            if source_page is None:
                continue
            for key in ("text_blocks", "table_blocks"):
                if key in source_page and parsed_page.get(key) != source_page.get(key):
                    parsed_page[key] = source_page.get(key)
                    changed = True
    if not changed:
        return result
    return result.__class__.model_validate(data)


def _compose_page_text_from_page_dict(page: Mapping[str, Any]) -> str | None:
    items: list[tuple[float, float, str]] = []
    for block in page.get("text_blocks") or []:
        if not isinstance(block, Mapping):
            continue
        if not _should_render_text_block_dict(block):
            continue
        text = _canonical_ocr_text(block.get("text"))
        if not text:
            continue
        box = block.get("bounding_box")
        items.append((_bbox_dict_value(box, "y"), _bbox_dict_value(box, "x"), text))
    for table in page.get("table_blocks") or []:
        if not isinstance(table, Mapping):
            continue
        text = _table_block_to_plain_text(table)
        if not text:
            continue
        box = table.get("bounding_box") or _table_cells_union_bbox(table)
        items.append((_bbox_dict_value(box, "y"), _bbox_dict_value(box, "x"), text))
    return _compose_unique_text_items(items)


def _compose_unique_text_items(items: Sequence[tuple[float, float, str]]) -> str | None:
    lines: list[str] = []
    seen_texts: list[str] = []
    for _, _, text in _sort_text_items_by_reading_order(items):
        if _is_redundant_text(text, seen_texts):
            continue
        seen_texts.append(text)
        lines.append(text)
    return "\n".join(lines) or None


def _sort_text_items_by_reading_order(
    items: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    if not items:
        return []
    sorted_items = sorted(items, key=lambda item: (item[0], item[1], item[2]))
    line_tolerance = 12.0
    lines: list[list[tuple[float, float, str]]] = []
    line_anchors: list[float] = []
    for item in sorted_items:
        y = float(item[0])
        if not lines or abs(y - line_anchors[-1]) > line_tolerance:
            lines.append([item])
            line_anchors.append(y)
            continue
        lines[-1].append(item)
        line_anchors[-1] = sum(float(existing[0]) for existing in lines[-1]) / len(lines[-1])

    flattened: list[tuple[float, float, str]] = []
    for line in lines:
        flattened.extend(sorted(line, key=lambda item: (item[1], item[0], item[2])))
    return flattened


def _canonical_ocr_text(value: Any) -> str:
    return _normalize_ocr_output_text(str(value or "")).strip()


def _canonical_match_text(value: Any) -> str:
    return _normalize_match_text(_canonical_ocr_text(value))


def _is_redundant_text(text: str, existing_texts: Sequence[str]) -> bool:
    candidate = _canonical_match_text(text)
    if not candidate:
        return True
    existing_norms = [_canonical_match_text(item) for item in existing_texts if str(item or "").strip()]
    for existing in existing_norms:
        if not existing:
            continue
        if candidate == existing:
            return True
        if len(candidate) >= 12 and candidate in existing:
            return True
        if len(existing) >= 12 and existing in candidate:
            return True

    candidate_units = _text_similarity_units(text)
    if not candidate_units:
        return False
    existing_units: set[str] = set()
    for item in existing_texts:
        existing_units.update(_text_similarity_units(item))
    if not existing_units:
        return False
    coverage = len(candidate_units & existing_units) / max(1, len(candidate_units))
    return coverage >= 0.82


def _annotate_paddle_seal_blocks(
    text_blocks: Sequence[dict[str, Any]],
    *,
    seal_regions: Sequence[Mapping[str, Any]],
    paddle_candidates: Sequence[Mapping[str, Any]],
) -> bool:
    if not seal_regions:
        return False
    changed = False
    paddle_text_regions = [
        candidate
        for candidate in paddle_candidates
        if _canonical_ocr_text(candidate.get("text"))
        and isinstance(candidate.get("bounding_box"), Mapping)
    ]
    for block in text_blocks:
        block_type = str(block.get("block_type") or "").strip().lower()
        if block_type not in {"seal", "stamp"}:
            continue
        if not _canonical_ocr_text(block.get("text")):
            continue
        block_box = block.get("bounding_box")
        if not isinstance(block_box, Mapping):
            continue
        matched_region = _matching_seal_region(block_box, seal_regions)
        if matched_region is None:
            continue
        text_confirmed = _has_confirming_paddle_text(block, paddle_text_regions)
        meta = dict(block.get("meta") or {})
        updated_meta = dict(meta)
        updated_meta["paddle_layout_label"] = matched_region.get("label") or "seal"
        updated_meta["paddle_layout_source"] = "paddleocr_ppstructurev3"
        updated_meta["paddle_text_confirmed"] = text_confirmed
        updated_meta["paddle_layout_region"] = {
            "bounding_box": matched_region.get("bounding_box"),
            "paddle_index": matched_region.get("paddle_index"),
            "block_id": matched_region.get("block_id"),
            "block_order": matched_region.get("block_order"),
        }
        if updated_meta == meta:
            continue
        block["meta"] = updated_meta
        changed = True
    return changed


def _matching_seal_region(
    block_box: Mapping[str, Any],
    seal_regions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for region in seal_regions:
        region_box = region.get("bounding_box")
        if not isinstance(region_box, Mapping):
            continue
        if _bbox_overlap_score(block_box, region_box) >= 0.05:
            return region
        if _bbox_center_distance_score(block_box, region_box) <= 1.4:
            return region
    return None


def _has_confirming_paddle_text(
    block: Mapping[str, Any],
    paddle_text_regions: Sequence[Mapping[str, Any]],
) -> bool:
    block_text = _canonical_ocr_text(block.get("text"))
    block_box = block.get("bounding_box")
    if not block_text or not isinstance(block_box, Mapping):
        return False
    for candidate in paddle_text_regions:
        candidate_text = _canonical_ocr_text(candidate.get("text"))
        candidate_box = candidate.get("bounding_box")
        if not candidate_text or not isinstance(candidate_box, Mapping):
            continue
        if _bbox_overlap_score(block_box, candidate_box) < 0.05:
            continue
        if _are_texts_near_duplicates(block_text, candidate_text):
            return True
    return False


def _merge_paddle_candidate_by_layout(
    candidate: Mapping[str, Any],
    existing_blocks: Sequence[Mapping[str, Any]],
) -> str | None:
    candidate_text = _canonical_ocr_text(candidate.get("text"))
    candidate_box = candidate.get("bounding_box")
    if not candidate_text or not isinstance(candidate_box, Mapping):
        return None

    overlapping_texts: list[str] = []
    near_duplicate_blocks: list[Mapping[str, Any]] = []
    for block in existing_blocks:
        if not isinstance(block, Mapping):
            continue
        existing_text = _canonical_ocr_text(block.get("text"))
        existing_box = block.get("bounding_box")
        if not existing_text or not isinstance(existing_box, Mapping):
            continue
        if _bbox_overlap_score(candidate_box, existing_box) < 0.25:
            continue
        overlapping_texts.append(existing_text)
        if _are_texts_near_duplicates(candidate_text, existing_text):
            near_duplicate_blocks.append(block)

    if len(near_duplicate_blocks) == 1 and _should_replace_text_block_with_paddle(
        candidate,
        near_duplicate_blocks[0],
    ):
        _replace_text_block_with_paddle_candidate(near_duplicate_blocks[0], candidate)
        return "updated"

    if near_duplicate_blocks:
        return "skip"

    if not overlapping_texts:
        return None
    return "skip" if _are_texts_near_duplicates(candidate_text, "\n".join(overlapping_texts)) else None


def _should_replace_text_block_with_paddle(
    candidate: Mapping[str, Any],
    existing_block: Mapping[str, Any],
) -> bool:
    block_type = str(existing_block.get("block_type") or "").strip().lower()
    if block_type in {"table_cell", "page_number"}:
        return False
    candidate_text = _canonical_ocr_text(candidate.get("text"))
    existing_text = _canonical_ocr_text(existing_block.get("text"))
    if not candidate_text or not existing_text:
        return False
    if candidate_text == existing_text:
        return False
    candidate_norm = _canonical_match_text(candidate_text)
    existing_norm = _canonical_match_text(existing_text)
    if not candidate_norm or not existing_norm:
        return False
    length_ratio = len(candidate_norm) / max(1, len(existing_norm))
    if len(candidate_norm) < len(existing_norm) and candidate_norm in existing_norm:
        return False
    if _is_paddle_overall_ocr_candidate(candidate) and length_ratio < 0.9:
        return False
    if length_ratio < 0.65 or length_ratio > 1.8:
        return False
    candidate_confidence = _to_float(candidate.get("confidence"))
    existing_confidence = _to_float(existing_block.get("confidence"))
    if candidate_confidence is not None and existing_confidence is not None:
        if candidate_confidence + 0.05 < existing_confidence:
            return False
    if existing_confidence is None:
        if candidate_confidence is not None and candidate_confidence < 0.9:
            return False
        return len(candidate_norm) >= len(existing_norm) + 2
    if existing_confidence < 0.85:
        return True
    return len(candidate_norm) >= len(existing_norm) + 2


def _replace_text_block_with_paddle_candidate(
    existing_block: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    if not isinstance(existing_block, dict):
        return
    original_text = str(existing_block.get("text") or "")
    existing_block["text"] = str(candidate.get("text") or "").strip()
    candidate_box = candidate.get("bounding_box")
    if isinstance(candidate_box, Mapping):
        existing_block["bounding_box"] = dict(candidate_box)
    candidate_confidence = _to_float(candidate.get("confidence"))
    if candidate_confidence is not None:
        existing_block["confidence"] = candidate_confidence
    meta = dict(existing_block.get("meta") or {})
    meta["text_fusion_source"] = "paddle_overlapping_text"
    meta["mineru_original_text"] = original_text
    meta["paddle_replacement_meta"] = dict(candidate.get("meta") or {})
    existing_block["meta"] = meta


def _is_paddle_overall_ocr_candidate(candidate: Mapping[str, Any]) -> bool:
    meta = candidate.get("meta")
    return isinstance(meta, Mapping) and meta.get("paddle_artifact_source") == "overall_ocr_res"


def _are_texts_near_duplicates(left: str, right: str) -> bool:
    left_norm = _canonical_match_text(left)
    right_norm = _canonical_match_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if len(left_norm) >= 12 and left_norm in right_norm:
        return True
    if len(right_norm) >= 12 and right_norm in left_norm:
        return True
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.62


def _bbox_overlap_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_x0 = _bbox_dict_value(left, "x")
    left_y0 = _bbox_dict_value(left, "y")
    left_x1 = left_x0 + _bbox_dict_value(left, "w")
    left_y1 = left_y0 + _bbox_dict_value(left, "h")
    right_x0 = _bbox_dict_value(right, "x")
    right_y0 = _bbox_dict_value(right, "y")
    right_x1 = right_x0 + _bbox_dict_value(right, "w")
    right_y1 = right_y0 + _bbox_dict_value(right, "h")
    width = max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0))
    height = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    if width <= 0.0 or height <= 0.0:
        return 0.0
    overlap = width * height
    left_area = max(1.0, (left_x1 - left_x0) * (left_y1 - left_y0))
    right_area = max(1.0, (right_x1 - right_x0) * (right_y1 - right_y0))
    return overlap / max(1.0, min(left_area, right_area))


def _bbox_center_distance_score(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_x = _bbox_dict_value(left, "x") + _bbox_dict_value(left, "w") / 2.0
    left_y = _bbox_dict_value(left, "y") + _bbox_dict_value(left, "h") / 2.0
    right_x = _bbox_dict_value(right, "x") + _bbox_dict_value(right, "w") / 2.0
    right_y = _bbox_dict_value(right, "y") + _bbox_dict_value(right, "h") / 2.0
    dx = abs(left_x - right_x)
    dy = abs(left_y - right_y)
    scale = max(
        1.0,
        _bbox_dict_value(left, "w"),
        _bbox_dict_value(left, "h"),
        _bbox_dict_value(right, "w"),
        _bbox_dict_value(right, "h"),
    )
    return (dx + dy) / scale


def _should_render_text_block_dict(block: Mapping[str, Any]) -> bool:
    meta = block.get("meta")
    if isinstance(meta, Mapping) and meta.get("render_text") is False:
        return False
    return True


def _should_render_text_block_obj(block: Any) -> bool:
    meta = getattr(block, "meta", None)
    if isinstance(meta, Mapping) and meta.get("render_text") is False:
        return False
    return True


def _text_similarity_units(value: Any) -> set[str]:
    text = _canonical_ocr_text(value)
    if not text:
        return set()
    parts = re.split(r"[\n\r\t,;:!?，。；：！？、（）()【】\[\]<>《》]+", text)
    units: set[str] = set()
    for part in parts:
        normalized = _normalize_match_text(part)
        if len(normalized) >= 4:
            units.add(normalized)
    whole = _normalize_match_text(text)
    if 4 <= len(whole) <= 40:
        units.add(whole)
    return units


def _table_block_to_plain_text(table: Any) -> str:
    cells = _as_table_cell_mappings(table)
    if not cells:
        return ""
    rows: dict[int, list[Mapping[str, Any]]] = {}
    fallback_row = 0
    for cell in cells:
        text = _canonical_ocr_text(cell.get("text"))
        if not text:
            continue
        row_index = _to_int(cell.get("row_index"))
        if row_index is None:
            row_index = fallback_row
            fallback_row += 1
        rows.setdefault(row_index, []).append(cell)

    rendered_rows: list[str] = []
    seen_rows: list[str] = []
    for row_index in sorted(rows):
        row_cells = sorted(
            rows[row_index],
            key=lambda cell: (
                _to_int(cell.get("col_index")) if _to_int(cell.get("col_index")) is not None else 10_000,
                _bbox_dict_value(cell.get("bounding_box"), "x"),
            ),
        )
        parts: list[str] = []
        seen_parts: list[str] = []
        for cell in row_cells:
            text = _canonical_ocr_text(cell.get("text"))
            if not text or _is_redundant_text(text, seen_parts):
                continue
            seen_parts.append(text)
            parts.append(text)
        row_text = " ".join(parts).strip()
        if row_text and not _is_redundant_text(row_text, seen_rows):
            seen_rows.append(row_text)
            rendered_rows.append(row_text)
    return "\n".join(rendered_rows)


def _as_table_cell_mappings(table: Any) -> list[Mapping[str, Any]]:
    raw_cells = table.get("cells") if isinstance(table, Mapping) else getattr(table, "cells", None)
    cells: list[Mapping[str, Any]] = []
    for cell in raw_cells or []:
        if isinstance(cell, Mapping):
            cells.append(cell)
        elif hasattr(cell, "model_dump"):
            cells.append(cell.model_dump(mode="python"))
    return cells


def _table_cells_union_bbox(table: Any) -> dict[str, float] | None:
    boxes = [
        cell.get("bounding_box")
        for cell in _as_table_cell_mappings(table)
        if isinstance(cell.get("bounding_box"), Mapping)
    ]
    if not boxes:
        return None
    x0 = min(float(box.get("x", 0)) for box in boxes)
    y0 = min(float(box.get("y", 0)) for box in boxes)
    x1 = max(float(box.get("x", 0)) + float(box.get("w", 0)) for box in boxes)
    y1 = max(float(box.get("y", 0)) + float(box.get("h", 0)) for box in boxes)
    return {"x": x0, "y": y0, "w": max(0.0, x1 - x0), "h": max(0.0, y1 - y0)}


def _normalize_ocr_output_text(value: str) -> str:
    text = value
    text = text.replace("（盖率）", "（盖章）")
    text = text.replace("(盖率)", "(盖章)")
    text = text.replace("签字盖率", "签字盖章")
    text = re.sub(r"(签字[（(])盖率([）)])", r"\1盖章\2", text)
    text = re.sub(
        r"([\u4e00-\u9fffA-Za-z]{1,20})[\[〔](\d{4})[\]〕](\s*\d+\s*号)",
        r"\1〔\2〕\3",
        text,
    )
    text = re.sub(
        r"([\u4e00-\u9fffA-Za-z]{1,20}(?:监办|府办|办发|府发|函|字|文))[（(](\d{4})[）)](\s*\d+\s*号)",
        r"\1〔\2〕\3",
        text,
    )
    return text


def _rewrite_markdown_artifacts_from_parsed(
    result: Any,
    *,
    artifact_dir: Path,
    output_stem: str,
) -> Any:
    markdown = _render_markdown_from_parsed_result(result)
    if not markdown.strip():
        return result

    markdown_paths = _markdown_artifact_paths(result, artifact_dir=artifact_dir)
    added_artifact = False
    if not markdown_paths:
        markdown_paths = [artifact_dir / f"{output_stem}.md"]
        added_artifact = True
    for path in markdown_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    if not added_artifact:
        return result
    artifacts = list(getattr(result, "artifacts", []) or [])
    artifacts.append(
        ArtifactRef(
            kind="markdown",
            uri=str(markdown_paths[0]),
            meta={"content_type": "text/markdown", "source": "merged_parsed_result"},
        )
    )
    return result.model_copy(update={"artifacts": artifacts})


def _rewrite_readable_artifacts_with_text_postprocessing(*, artifact_dir: Path) -> None:
    for path in sorted([*artifact_dir.glob("*.md"), *artifact_dir.glob("*.json")]):
        if not _should_postprocess_readable_artifact(path):
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rewritten = _normalize_ocr_output_text(original)
        if rewritten != original:
            path.write_text(rewritten, encoding="utf-8")


def _should_postprocess_readable_artifact(path: Path) -> bool:
    name = path.name
    if name == "paddle_table_structure.json":
        return False
    if name.endswith(".field_coordinates.json") or name.endswith(".error.json"):
        return False
    return path.suffix.lower() in {".md", ".json"}


def _markdown_artifact_paths(result: Any, *, artifact_dir: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for artifact in getattr(result, "artifacts", []) or []:
        if getattr(artifact, "kind", None) != "markdown":
            continue
        uri = getattr(artifact, "uri", None)
        if not uri:
            continue
        path = Path(str(uri))
        if path not in seen:
            paths.append(path)
            seen.add(path)
    for path in sorted(artifact_dir.glob("*.md")):
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _render_markdown_from_parsed_result(result: Any) -> str:
    if hasattr(result, "model_dump"):
        data = result.model_dump(mode="python")
        pages = data.get("pages") if isinstance(data, dict) else []
        if isinstance(pages, list):
            chunks: list[str] = []
            for page in pages:
                if not isinstance(page, Mapping):
                    continue
                page_index = _to_int(page.get("page_index")) or 0
                if len(pages) > 1:
                    chunks.append(f"# Page {page_index + 1}")
                page_text = _compose_page_text_from_page_dict(page)
                if page_text:
                    chunks.append(page_text)
            return "\n\n".join(chunks).strip() + "\n"

    pages = getattr(result, "pages", []) or []
    chunks: list[str] = []
    for page in pages:
        page_index = _to_int(getattr(page, "page_index", None)) or 0
        if len(pages) > 1:
            chunks.append(f"# Page {page_index + 1}")
        blocks = _sort_text_blocks_for_render(getattr(page, "text_blocks", []) or [])
        seen: set[str] = set()
        for block in blocks:
            if not _should_render_text_block_obj(block):
                continue
            text = str(getattr(block, "text", "") or "").strip()
            if not text:
                continue
            if "<" in text and ">" in text:
                text = _html_to_plain_text(text)
            text = _normalize_ocr_output_text(text).strip()
            normalized = _normalize_match_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            chunks.append(text)
    return "\n\n".join(chunks).strip() + "\n"


def _sort_text_blocks_for_render(blocks: Sequence[Any]) -> list[Any]:
    return sorted(
        blocks,
        key=lambda block: (
            _bbox_attr(getattr(block, "bounding_box", None), "y"),
            _bbox_attr(getattr(block, "bounding_box", None), "x"),
            str(getattr(block, "text", "")),
        ),
    )


def _sort_text_block_dicts(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        blocks,
        key=lambda block: (
            _bbox_dict_value(block.get("bounding_box"), "y"),
            _bbox_dict_value(block.get("bounding_box"), "x"),
            str(block.get("text") or ""),
        ),
    )


def _compose_page_text_from_block_dicts(blocks: Sequence[Mapping[str, Any]]) -> str | None:
    lines: list[str] = []
    seen_texts: list[str] = []
    for block in _sort_text_block_dicts([dict(item) for item in blocks]):
        if not _should_render_text_block_dict(block):
            continue
        text = _canonical_ocr_text(block.get("text"))
        if not text:
            continue
        if _is_redundant_text(text, seen_texts):
            continue
        seen_texts.append(text)
        lines.append(text)
    return "\n".join(lines) or None


def _bbox_attr(box: Any, key: str) -> float:
    value = getattr(box, key, None)
    parsed = _to_float(value)
    return parsed if parsed is not None else 0.0


def _bbox_dict_value(box: Any, key: str) -> float:
    if not isinstance(box, Mapping):
        return 0.0
    parsed = _to_float(box.get(key))
    return parsed if parsed is not None else 0.0


def _as_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_field_coordinate_artifacts(
    result: Any,
    *,
    keywords: Sequence[str],
    source_pdf_path: Path,
    annotation_pdf_path: Path,
    output_dir: Path,
    output_stem: str,
    table_engine: str,
) -> dict[str, Any]:
    resolved_keywords = _normalize_field_keywords(keywords)
    matches = _extract_field_coordinate_matches(result, resolved_keywords)
    coordinates_path = output_dir / f"{output_stem}.field_coordinates.json"
    annotated_pdf_path = output_dir / f"{output_stem}.field_annotations.pdf"
    _write_field_annotation_pdf(
        annotation_pdf_path,
        annotated_pdf_path,
        matches=matches,
        parsed_pdf=result.parsed_pdf,
    )
    payload = {
        "source_pdf": str(source_pdf_path),
        "annotation_pdf_source": str(annotation_pdf_path),
        "annotated_pdf": str(annotated_pdf_path),
        "table_engine": table_engine,
        "coord_space": str(getattr(result, "coord_space", "mineru_layout")),
        "field_keywords": list(resolved_keywords),
        "match_count": len(matches),
        "matches": matches,
    }
    coordinates_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "coordinates_path": str(coordinates_path),
        "annotation_pdf_path": str(annotated_pdf_path),
        "match_count": len(matches),
        "artifacts": [
            ArtifactRef(
                kind="field_coordinates_json",
                uri=str(coordinates_path),
                meta={
                    "field_keywords": list(resolved_keywords),
                    "match_count": len(matches),
                    "content_type": "application/json",
                },
            ),
            ArtifactRef(
                kind="field_annotation_pdf",
                uri=str(annotated_pdf_path),
                meta={
                    "field_keywords": list(resolved_keywords),
                    "match_count": len(matches),
                    "content_type": "application/pdf",
                },
            ),
        ],
    }


def _extract_field_coordinate_matches(result: Any, keywords: Sequence[str]) -> list[dict[str, Any]]:
    normalized_keywords = [(keyword, _normalize_match_text(keyword)) for keyword in keywords]
    if not normalized_keywords:
        return []

    matches: list[dict[str, Any]] = []
    parsed_pdf = getattr(result, "parsed_pdf", None)
    pages = getattr(parsed_pdf, "pages", []) if parsed_pdf is not None else []
    for page in pages:
        page_index = int(getattr(page, "page_index", 0))
        for block in getattr(page, "text_blocks", []) or []:
            text = str(getattr(block, "text", "") or "")
            normalized_text = _normalize_match_text(text)
            if not normalized_text:
                continue
            bounding_box = getattr(block, "bounding_box", None)
            if bounding_box is None:
                continue
            for keyword, normalized_keyword in normalized_keywords:
                if normalized_keyword and normalized_keyword in normalized_text:
                    matches.append(
                        _build_text_block_field_match(
                            keyword=keyword,
                            text=text,
                            page=page,
                            block=block,
                            bounding_box=bounding_box,
                        )
                    )
        for table in getattr(page, "table_blocks", []) or []:
            for cell in getattr(table, "cells", []) or []:
                text = str(getattr(cell, "text", "") or "")
                normalized_text = _normalize_match_text(text)
                if not normalized_text:
                    continue
                bounding_box = getattr(cell, "bounding_box", None)
                if bounding_box is None:
                    continue
                for keyword, normalized_keyword in normalized_keywords:
                    if normalized_keyword and normalized_keyword in normalized_text:
                        matches.append(
                            _build_field_match(
                                keyword=keyword,
                                text=text,
                                page=page,
                                table=table,
                                cell=cell,
                                bounding_box=bounding_box,
                            )
                        )

    matches = _dedupe_field_matches(matches)
    matches.sort(
        key=lambda item: (
            int(item["page_index"]),
            float(item["bounding_box"]["y"]),
            float(item["bounding_box"]["x"]),
            str(item["keyword"]),
        )
    )
    return matches


def _build_field_match(
    *,
    keyword: str,
    text: str,
    page: Any,
    table: Any,
    cell: Any,
    bounding_box: BoundingBox,
) -> dict[str, Any]:
    box = _bbox_to_dict(bounding_box)
    return {
        "keyword": keyword,
        "text": text,
        "page_index": int(getattr(page, "page_index", 0)),
        "page_number": int(getattr(page, "page_index", 0)) + 1,
        "source": "table_cell",
        "table_id": getattr(table, "table_id", None),
        "cell_id": getattr(cell, "cell_id", None),
        "row_index": getattr(cell, "row_index", None),
        "col_index": getattr(cell, "col_index", None),
        "row_span": getattr(cell, "row_span", None),
        "col_span": getattr(cell, "col_span", None),
        "confidence": getattr(cell, "confidence", None),
        "coord_space": getattr(table, "coord_space", "mineru_layout"),
        "bounding_box": box,
        "quad_points": _bbox_quad_points(box),
        "pdf_bounding_box": None,
        "pdf_quad_points": None,
        "meta": {
            "table_provider": getattr(table, "provider", None),
            "cell_meta": getattr(cell, "meta", {}),
        },
    }


def _build_text_block_field_match(
    *,
    keyword: str,
    text: str,
    page: Any,
    block: Any,
    bounding_box: BoundingBox,
) -> dict[str, Any]:
    box = _bbox_to_dict(bounding_box)
    meta = dict(getattr(block, "meta", {}) or {})
    return {
        "keyword": keyword,
        "text": text,
        "page_index": int(getattr(page, "page_index", 0)),
        "page_number": int(getattr(page, "page_index", 0)) + 1,
        "source": "text_block",
        "table_id": None,
        "cell_id": None,
        "row_index": None,
        "col_index": None,
        "row_span": None,
        "col_span": None,
        "confidence": getattr(block, "confidence", None),
        "coord_space": meta.get("coord_space") or getattr(page, "coord_space", "mineru_layout"),
        "bounding_box": box,
        "quad_points": _bbox_quad_points(box),
        "pdf_bounding_box": None,
        "pdf_quad_points": None,
        "meta": {
            "block_type": getattr(block, "block_type", None),
            "block_meta": meta,
        },
    }


def _dedupe_field_matches(matches: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for match in matches:
        box = match.get("bounding_box") if isinstance(match.get("bounding_box"), dict) else {}
        key = (
            match.get("keyword"),
            match.get("page_index"),
            round(float(box.get("x", 0)), 1),
            round(float(box.get("y", 0)), 1),
            round(float(box.get("w", 0)), 1),
            round(float(box.get("h", 0)), 1),
            _normalize_match_text(str(match.get("text") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        exact_deduped.append(match)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for match in exact_deduped:
        text_key = _normalize_match_text(str(match.get("text") or ""))
        if not text_key:
            deduped_key = (id(match),)
        else:
            deduped_key = (
                match.get("keyword"),
                match.get("page_index"),
                text_key,
            )
        groups.setdefault(deduped_key, []).append(match)

    deduped: list[dict[str, Any]] = []
    for group in groups.values():
        if len(group) <= 1:
            deduped.extend(group)
            continue
        source_classes = {_field_match_source_class(match) for match in group}
        if len(source_classes) <= 1:
            deduped.extend(group)
            continue
        best_rank = min(_field_match_source_rank(match) for match in group)
        deduped.extend(match for match in group if _field_match_source_rank(match) == best_rank)
    return deduped


def _field_match_source_class(match: Mapping[str, Any]) -> str:
    source = str(match.get("source") or "")
    meta = match.get("meta") if isinstance(match.get("meta"), Mapping) else {}
    block_meta = meta.get("block_meta") if isinstance(meta.get("block_meta"), Mapping) else {}
    cell_meta = meta.get("cell_meta") if isinstance(meta.get("cell_meta"), Mapping) else {}
    table_provider = str(meta.get("table_provider") or "")
    raw_values = [
        source,
        table_provider,
        str(block_meta.get("source") or ""),
        str(block_meta.get("paddle_artifact_source") or ""),
        str(cell_meta.get("source") or ""),
        str(cell_meta.get("merge_source") or ""),
    ]
    raw = " ".join(raw_values).lower()
    if source == "table_cell":
        return "table_cell"
    if "paddle" in raw or "ppstructure" in raw:
        return "paddle_text"
    if raw.strip():
        return "mineru_text"
    return "unknown"


def _field_match_source_rank(match: Mapping[str, Any]) -> int:
    source_class = _field_match_source_class(match)
    if source_class == "table_cell":
        return 0
    if source_class == "paddle_text":
        return 1
    if source_class == "mineru_text":
        return 2
    return 3


def _write_field_annotation_pdf(
    input_pdf_path: Path,
    output_pdf_path: Path,
    *,
    matches: list[dict[str, Any]],
    parsed_pdf: Any,
) -> None:
    try:
        import fitz  # PyMuPDF
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyMuPDF is required to write field annotation PDFs") from exc

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(input_pdf_path)) as doc:
        pages_by_index = {
            int(getattr(page, "page_index", 0)): page
            for page in (getattr(parsed_pdf, "pages", []) or [])
        }
        for match in matches:
            page_index = int(match["page_index"])
            if page_index < 0 or page_index >= len(doc):
                continue
            pdf_page = doc[page_index]
            rect = _match_pdf_rect(match, pages_by_index.get(page_index), pdf_page.rect)
            if rect is None:
                continue
            match["pdf_bounding_box"] = _pdf_rect_to_dict(rect)
            match["pdf_quad_points"] = _bbox_quad_points(match["pdf_bounding_box"])
            expanded = fitz.Rect(rect.x0 - 1.5, rect.y0 - 1.5, rect.x1 + 1.5, rect.y1 + 1.5)
            expanded &= pdf_page.rect
            pdf_page.draw_rect(
                expanded,
                color=(1, 0, 0),
                fill=(1, 1, 0),
                fill_opacity=0.18,
                width=1.4,
                overlay=True,
            )
        doc.save(str(output_pdf_path), garbage=4, deflate=True)


def _match_pdf_rect(match: dict[str, Any], page: Any, page_rect: Any) -> Any:
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyMuPDF is required to convert field coordinates to PDF points") from exc

    box = match.get("bounding_box")
    if not isinstance(box, dict):
        return None
    source_width, source_height = _page_coordinate_size(page)
    if not source_width or not source_height:
        source_width = float(page_rect.width)
        source_height = float(page_rect.height)
    if source_width <= 0 or source_height <= 0:
        return None
    scale_x = float(page_rect.width) / source_width
    scale_y = float(page_rect.height) / source_height
    x0 = float(box["x"]) * scale_x
    y0 = float(box["y"]) * scale_y
    x1 = float(box["x"] + box["w"]) * scale_x
    y1 = float(box["y"] + box["h"]) * scale_y
    rect = fitz.Rect(x0, y0, x1, y1)
    return rect & page_rect


def _page_coordinate_size(page: Any) -> tuple[float | None, float | None]:
    image_size = getattr(page, "image_size", None)
    if image_size is not None:
        width = getattr(image_size, "width", None)
        height = getattr(image_size, "height", None)
        if width and height:
            return float(width), float(height)
    return None, None


def _bbox_to_dict(box: BoundingBox) -> dict[str, float]:
    return {
        "x": float(box.x),
        "y": float(box.y),
        "w": float(box.w),
        "h": float(box.h),
    }


def _pdf_rect_to_dict(rect: Any) -> dict[str, float]:
    return {
        "x": round(float(rect.x0), 3),
        "y": round(float(rect.y0), 3),
        "w": round(float(rect.width), 3),
        "h": round(float(rect.height), 3),
        "x0": round(float(rect.x0), 3),
        "y0": round(float(rect.y0), 3),
        "x1": round(float(rect.x1), 3),
        "y1": round(float(rect.y1), 3),
    }


def _bbox_quad_points(box: dict[str, float]) -> list[dict[str, float]]:
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])
    return [
        {"x": round(x, 3), "y": round(y, 3)},
        {"x": round(x + w, 3), "y": round(y, 3)},
        {"x": round(x + w, 3), "y": round(y + h, 3)},
        {"x": round(x, 3), "y": round(y + h, 3)},
    ]


def _normalize_field_keywords(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    raw_items: list[str]
    if isinstance(value, str):
        raw_items = [value]
    else:
        raw_items = [str(item) for item in value]
    keywords: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        for part in re.split(r"[,，;；\n\r\t]+", item):
            keyword = part.strip()
            if not keyword:
                continue
            key = _normalize_match_text(keyword)
            if key and key not in seen:
                keywords.append(keyword)
                seen.add(key)
    return keywords


def _normalize_match_text(value: str) -> str:
    normalized = _normalize_ocr_output_text(value)
    return re.sub(
        r"[\s,.;:!?，。；：！？、（）()\[\]【】<>《》“”\"'‘’\-—_＿/\\]+",
        "",
        normalized,
    ).casefold()


def _load_field_keywords_json(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return _normalize_field_keywords(value)
    return _normalize_field_keywords(payload)


def _read_field_coordinate_summary(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _flatten_mineru_output_dir(result: Any, *, artifact_dir: Path, mineru_pdf_path: Path) -> Any:
    """Move MinerU's extra <doc_stem>/<parse_method> layers into this operator's artifact dir."""

    mineru_output_dir = artifact_dir / _safe_stem(mineru_pdf_path.name)
    if not mineru_output_dir.is_dir():
        return result
    old_root = mineru_output_dir.resolve()
    new_root = artifact_dir.resolve()
    if old_root == new_root:
        return result

    relocations = _mineru_output_relocations(old_root, new_root)
    parse_method_dir = mineru_output_dir / "auto"
    if parse_method_dir.is_dir():
        for child in list(parse_method_dir.iterdir()):
            destination = artifact_dir / child.name
            _remove_path_if_exists(destination)
            shutil.move(str(child), str(destination))
        try:
            parse_method_dir.rmdir()
        except OSError:
            pass

    for child in list(mineru_output_dir.iterdir()):
        destination = artifact_dir / child.name
        _remove_path_if_exists(destination)
        shutil.move(str(child), str(destination))
    try:
        mineru_output_dir.rmdir()
    except OSError:
        pass
    return _rewrite_model_path_prefixes(result, relocations)


def _mineru_output_relocations(old_root: Path, new_root: Path) -> list[tuple[Path, Path]]:
    relocations: list[tuple[Path, Path]] = []
    auto_dir = old_root / "auto"
    if auto_dir.is_dir():
        relocations.append((auto_dir, new_root))
    relocations.append((old_root, new_root))
    return relocations


def _rewrite_model_path_prefixes(model: Any, relocations: list[tuple[Path, Path]]) -> Any:
    if not relocations or not hasattr(model, "model_dump"):
        return model
    data = model.model_dump(mode="python")
    rewritten = _rewrite_path_prefixes(data, relocations)
    return model.__class__.model_validate(rewritten)


def _rewrite_path_prefixes(value: Any, relocations: list[tuple[Path, Path]]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_path_prefixes(item, relocations) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_path_prefixes(item, relocations) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_path_prefixes(item, relocations) for item in value)
    if isinstance(value, str):
        return _rewrite_path_string(value, relocations)
    return value


def _rewrite_path_string(value: str, relocations: list[tuple[Path, Path]]) -> str:
    for old_root, new_root in relocations:
        old_value = str(old_root)
        new_value = str(new_root)
        if value == old_value:
            return new_value
        if value.startswith(old_value + os.sep):
            return new_value + value[len(old_value):]
        if os.sep != "/" and value.startswith(old_value.replace(os.sep, "/") + "/"):
            old_posix = old_value.replace(os.sep, "/")
            new_posix = new_value.replace(os.sep, "/")
            return new_posix + value[len(old_posix):]
    return value


def _render_pdf_page_screenshots(
    pdf_path: Path,
    *,
    output_dir: Path,
    page_count: int,
    dpi: int,
) -> ArtifactRef:
    if page_count <= 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "page_manifest.jsonl"
        manifest_path.write_text("", encoding="utf-8")
        return ArtifactRef(
            kind="page_screenshots_manifest",
            uri=str(manifest_path),
            meta={"page_count": 0, "dpi": dpi},
        )
    if dpi <= 0:
        raise ValueError("page_screenshot_dpi must be greater than 0")
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm is required for page screenshot export")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "page_manifest.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    for page_number in range(1, page_count + 1):
        prefix = output_dir / f"page_{page_number:04d}"
        png_path = output_dir / f"page_{page_number:04d}.png"
        _unlink_if_exists(png_path)
        command = [
            pdftoppm,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "pdftoppm failed while exporting page screenshots: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        if not png_path.exists():
            raise RuntimeError(f"pdftoppm did not create expected screenshot: {png_path}")
        width, height = _read_image_size(png_path)
        manifest_rows.append(
            {
                "page_index": page_number - 1,
                "page_number": page_number,
                "image_path": str(png_path),
                "source_pdf": str(pdf_path),
                "dpi": dpi,
                "width": width,
                "height": height,
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return ArtifactRef(
        kind="page_screenshots_manifest",
        uri=str(manifest_path),
        meta={"page_count": page_count, "dpi": dpi, "output_dir": str(output_dir)},
    )


def _read_image_size(image_path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None, None
    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def _write_batch_csv(path: Path, items: list[PdfFileExtractResult]) -> None:
    fields = [
        "pdf_path",
        "success",
        "skipped",
        "json_path",
        "error_report",
        "error_type",
        "error_code",
        "error",
        "elapsed_seconds",
        "processing_seconds",
        "queue_wait_seconds",
        "wall_elapsed_seconds",
        "page_count",
        "text_block_count",
        "table_block_count",
        "table_cell_count",
        "table_engine",
        "paddle_table_mode",
        "input_type",
        "relative_input_path",
        "output_relative_dir",
        "converted_pdf_path",
        "page_screenshots_manifest",
        "field_keywords",
        "field_match_count",
        "field_coordinates_path",
        "field_annotation_pdf_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = item.model_dump(mode="json")
            writer.writerow({field: row.get(field) for field in fields})


def _find_artifact_uri(artifacts: list[Any], kind: str) -> str | None:
    for artifact in artifacts:
        if getattr(artifact, "kind", None) == kind:
            return getattr(artifact, "uri", None)
    return None


def _artifact_uri(artifacts: list[Any], kind: str) -> str | None:
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("kind") == kind:
            uri = artifact.get("uri")
            return str(uri) if uri is not None else None
    return None


def _result_layout_provider(result: Any) -> str | None:
    pages = getattr(result, "pages", []) or []
    if not pages:
        return None
    first_page = pages[0]
    page_meta = getattr(first_page, "page_meta", None)
    if isinstance(page_meta, Mapping):
        provider = page_meta.get("layout_provider")
        return str(provider) if provider is not None else None
    return None


def _resolve_api_url(api_url: str | None) -> str:
    if api_url is not None and api_url.strip():
        return api_url.strip()
    env_api_url = os.environ.get("MINERU_API_URL")
    if env_api_url is not None and env_api_url.strip():
        return env_api_url.strip()
    return DEFAULT_MINERU_API_URL


def _safe_stem(value: str) -> str:
    stem = Path(value).stem or value
    safe = INVALID_PATH_CHARS_RE.sub("_", stem).strip(" ._")
    return safe or "document"


def _bool_cli_value(value: bool) -> str:
    return "true" if value else "false"


def _validate_positive_int_or_none(value: int | None, *, option_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{option_name} must be a positive integer or None")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{option_name} must be a positive integer or None")
    return parsed


def _int_or_zero(value: int | None) -> int:
    return int(value) if value is not None else 0


def _zero_to_none(value: int) -> int | None:
    return int(value) if int(value) > 0 else None


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _remove_path_if_exists(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _daft_available() -> bool:
    try:
        import daft  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _import_daft() -> Any:
    try:
        import daft
    except ModuleNotFoundError as exc:
        raise RuntimeError("Daft is not installed in the current environment") from exc
    return daft


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _remove_stale_lock(lock_path: Path) -> None:
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return
    pid = 0
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    if pid and _process_is_alive(pid):
        return
    if time.time() - stat.st_mtime < 60:
        return
    _unlink_if_exists(lock_path)


@contextmanager
def _filesystem_concurrency_slot(
    *,
    lock_dir: Path | None,
    limit: int,
    timeout_seconds: float,
) -> Iterator[None]:
    if lock_dir is None:
        yield
        return
    lock_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    slot_path: Path | None = None
    while slot_path is None:
        for index in range(limit):
            candidate = lock_dir / f"slot-{index}.lock"
            try:
                fd = os.open(str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                _remove_stale_lock(candidate)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump({"pid": os.getpid(), "created_at": time.time(), "slot": index}, file)
            slot_path = candidate
            break
        if slot_path is not None:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for a MinerU concurrency slot in {lock_dir}")
        time.sleep(0.25)
    try:
        yield
    finally:
        _unlink_if_exists(slot_path)
