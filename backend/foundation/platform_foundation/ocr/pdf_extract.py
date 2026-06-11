from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import asyncio
import csv
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
                converted_pdf_path = artifact_dir / f"{output_stem}.converted.pdf"
                _convert_image_to_pdf(resolved_input, converted_pdf_path)
                pdf_for_mineru = converted_pdf_path

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
    return re.sub(r"\s+", "", value).casefold()


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
