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
import time
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

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
    pages_per_second: float = Field(default=0.0, ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    skipped_count: int = Field(default=0, ge=0)
    batch_report_path: str
    batch_report_csv_path: str
    failed_files: str
    failed_files_jsonl: str
    total_elapsed_seconds: float = Field(ge=0)
    items: list[PdfFileExtractResult] = Field(default_factory=list)


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
        queue_wait_seconds: float = 0.0,
        extra_args: Sequence[str] | None = None,
        mineru_options: dict[str, Any] | None = None,
    ) -> PdfFileExtractResult:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_pdf = Path(pdf_path).expanduser().resolve()
        resolved_source_name = source_file_name or resolved_pdf.name
        output_stem = _safe_stem(resolved_source_name or resolved_pdf.name)
        artifact_dir = output_root / output_stem
        json_path = output_root / f"{output_stem}.json"
        error_path = output_root / f"{output_stem}.error.json"
        artifact_dir.mkdir(parents=True, exist_ok=True)

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

        started = time.perf_counter()
        try:
            self._validate_pdf_path(resolved_pdf)
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
                resolved_pdf,
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
            parsed = parsed.model_copy(update={"source_file_name": resolved_source_name})
            json_path.write_text(dump_pure_mineru_json(parsed, indent=2), encoding="utf-8")
            processing_seconds = time.perf_counter() - started
            return PdfFileExtractResult(
                pdf_path=str(resolved_pdf),
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
            )
        except Exception as exc:
            processing_seconds = time.perf_counter() - started
            result = PdfFileExtractResult(
                pdf_path=str(resolved_pdf),
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
    def _validate_pdf_path(pdf_path: Path) -> None:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file():
            raise ValueError(f"PDF path is not a file: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF path must end with .pdf: {pdf_path}")


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
        pdfs = _list_pdfs(input_root, recursive=recursive)
        if limit is not None and limit > 0:
            pdfs = pdfs[:limit]
        if not pdfs:
            raise ValueError(f"No PDF files found in {input_root}")

        started = time.perf_counter()
        resolved_table_engine = _resolve_table_engine(
            use_paddle_tables=use_paddle_tables,
            table_engine=table_engine,
        )
        selected_engine = _resolve_batch_engine(engine=engine, concurrency=concurrency)
        tasks = [
            _BatchTask(
                pdf_path=pdf,
                source_file_name=pdf.name,
                output_dir=output_root,
                table_engine=resolved_table_engine,
                paddle_table_mode=paddle_table_mode,
                paddle_device=paddle_device,
                resume=resume,
                overwrite=overwrite,
            )
            for pdf in pdfs
        ]

        if selected_engine == "sequential":
            items = [self._run_task(task, None, 1, lock_acquire_timeout_seconds) for task in tasks]
        elif selected_engine == "thread":
            items = self._run_threaded(tasks, concurrency, lock_acquire_timeout_seconds)
        else:
            items = self._run_daft(tasks, concurrency, lock_acquire_timeout_seconds)

        return self._write_report(
            input_root=input_root,
            output_root=output_root,
            engine=selected_engine,
            concurrency=concurrency,
            table_engine=resolved_table_engine,
            paddle_table_mode=paddle_table_mode,
            paddle_device=paddle_device,
            started_at=started,
            items=items,
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
            return self.file_operator.extract_file(
                task.pdf_path,
                output_dir=task.output_dir,
                table_engine=task.table_engine,
                paddle_table_mode=task.paddle_table_mode,
                paddle_device=task.paddle_device,
                source_file_name=task.source_file_name,
                overwrite=True,
                queue_wait_seconds=queue_wait_seconds,
            )

    def _run_threaded(
        self,
        tasks: list["_BatchTask"],
        concurrency: int,
        lock_acquire_timeout_seconds: float,
    ) -> list[PdfFileExtractResult]:
        lock_dir = tasks[0].output_dir / ".mineru_concurrency_locks"
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
        concurrency: int,
        lock_acquire_timeout_seconds: float,
    ) -> list[PdfFileExtractResult]:
        daft = _import_daft()
        extract_udf = _build_daft_udf(daft, concurrency)
        lock_dir = tasks[0].output_dir / ".mineru_concurrency_locks"
        df = daft.from_pydict(
            {
                "pdf_path": [str(task.pdf_path) for task in tasks],
                "source_file_name": [task.source_file_name for task in tasks],
                "output_dir": [str(task.output_dir) for task in tasks],
                "table_engine": [task.table_engine for task in tasks],
                "paddle_table_mode": [task.paddle_table_mode for task in tasks],
                "paddle_device": [task.paddle_device or "" for task in tasks],
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
                daft.col("output_dir"),
                daft.col("table_engine"),
                daft.col("paddle_table_mode"),
                daft.col("paddle_device"),
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
        pages_per_second = round(page_count / total_elapsed_seconds, 3) if total_elapsed_seconds > 0 else 0.0
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
            pages_per_second=pages_per_second,
            success_count=sum(1 for item in items if item.success),
            failure_count=len(failed_items),
            skipped_count=sum(1 for item in items if item.skipped),
            batch_report_path=str(batch_report_path),
            batch_report_csv_path=str(batch_report_csv_path),
            failed_files=str(failed_files_path),
            failed_files_jsonl=str(failed_files_jsonl_path),
            total_elapsed_seconds=total_elapsed_seconds,
            items=items,
        )
        batch_report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report


class _BatchTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pdf_path: Path
    source_file_name: str
    output_dir: Path
    table_engine: str
    paddle_table_mode: str
    paddle_device: str | None
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
    output_dir: str,
    table_engine: str,
    paddle_table_mode: str,
    paddle_device: str,
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
        output_dir=Path(output_dir),
        table_engine=table_engine,
        paddle_table_mode=paddle_table_mode,
        paddle_device=paddle_device or None,
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
        output_dir: str,
        table_engine: str,
        paddle_table_mode: str,
        paddle_device: str,
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
            output_dir,
            table_engine,
            paddle_table_mode,
            paddle_device,
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
) -> PdfFileExtractResult | None:
    output_stem = _safe_stem(source_file_name or pdf_path.name)
    json_path = output_dir / f"{output_stem}.json"
    artifact_dir = output_dir / output_stem
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
    )


def _list_pdfs(input_root: Path, *, recursive: bool) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() != ".pdf":
            raise ValueError(f"Input file must be a PDF: {input_root}")
        return [input_root]
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"Input path is not a directory: {input_root}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    seen: set[str] = set()
    pdfs: list[Path] = []
    for path in sorted(input_root.glob(pattern)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        pdfs.append(resolved)
    return pdfs


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


def _resolve_api_url(api_url: str | None) -> str:
    if api_url is not None and api_url.strip():
        return api_url.strip()
    env_api_url = os.environ.get("MINERU_API_URL")
    if env_api_url is not None and env_api_url.strip():
        return env_api_url.strip()
    return DEFAULT_MINERU_API_URL


def _safe_stem(value: str) -> str:
    stem = Path(value).stem or value
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
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
