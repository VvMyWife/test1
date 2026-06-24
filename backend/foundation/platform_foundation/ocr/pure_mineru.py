from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import ArtifactRef, DocumentItem, ParsedPdf, TableBlock, TextBlock
from ..operators import LayoutExtractMinerUOperator, LayoutExtractMinerUPaddleTableOperator
from .mineru_layout import MinerULayoutOperateResult, operate

DEFAULT_MINERU_API_URL = "http://127.0.0.1:18000"
DEFAULT_TIMEOUT_SECONDS = 1800.0


class MinerUPdfPage(BaseModel):
    """Business-field-free page result for direct MinerU consumers."""

    model_config = ConfigDict(extra="forbid")

    page_index: int = Field(ge=0)
    text: str | None = None
    text_blocks: list[TextBlock] = Field(default_factory=list)
    table_blocks: list[TableBlock] = Field(default_factory=list)
    page_meta: dict[str, Any] = Field(default_factory=dict)
    layout_ref: ArtifactRef | None = None


class MinerUPdfResult(BaseModel):
    """Pure MinerU JSON result: one PDF in, one structured JSON out."""

    model_config = ConfigDict(extra="forbid")

    source_pdf: str
    source_file_name: str
    page_count: int = Field(ge=0)
    coord_space: str
    layout_ref: ArtifactRef | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    parsed_pdf: ParsedPdf
    pages: list[MinerUPdfPage] = Field(default_factory=list)


def extract_pdf(
    pdf_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    api_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    parse_method: str = "auto",
    backend: str = "pipeline",
    lang: str = "ch",
    extra_args: Sequence[str] | None = None,
    table_engine: str = "ocr",
    paddle_table_mode: str = "auto",
    paddle_device: str | None = None,
    mineru_options: Mapping[str, Any] | None = None,
    operator_factory: Callable[[], LayoutExtractMinerUOperator] = LayoutExtractMinerUPaddleTableOperator,
) -> MinerUPdfResult:
    """Extract one local PDF and return a business-field-free MinerU result."""

    resolved_pdf = Path(pdf_path).expanduser().resolve()
    options = _build_mineru_options(
        output_dir=output_dir,
        api_url=api_url,
        timeout_seconds=timeout_seconds,
        parse_method=parse_method,
        backend=backend,
        lang=lang,
        extra_args=extra_args,
        table_engine=table_engine,
        paddle_table_mode=paddle_table_mode,
        paddle_device=paddle_device,
        mineru_options=mineru_options,
    )
    document = _build_internal_document(resolved_pdf, mineru_options=options)
    result = operate(document, mineru_options=options, operator_factory=operator_factory)
    return to_pure_mineru_result(result, source_pdf=resolved_pdf)


def to_pure_mineru_result(
    result: MinerULayoutOperateResult,
    *,
    source_pdf: str | Path,
) -> MinerUPdfResult:
    resolved_pdf = Path(source_pdf).expanduser().resolve()
    return MinerUPdfResult(
        source_pdf=str(resolved_pdf),
        source_file_name=resolved_pdf.name,
        page_count=result.page_count,
        coord_space=result.coord_space,
        layout_ref=result.layout_ref,
        artifacts=result.artifacts,
        parsed_pdf=result.parsed_pdf.model_copy(update={"pdf_path": str(resolved_pdf)}),
        pages=[
            MinerUPdfPage(
                page_index=page.page_index,
                text=page.text,
                text_blocks=page.text_blocks,
                table_blocks=page.table_blocks,
                page_meta=dict(page.page_meta),
                layout_ref=page.layout_ref,
            )
            for page in result.pages
        ],
    )


def dump_pure_mineru_json(result: MinerUPdfResult, *, indent: int | None = 2) -> str:
    """Serialize the external JSON shape without duplicating page payloads."""

    return result.model_dump_json(indent=indent, exclude={"parsed_pdf"})


def _build_mineru_options(
    *,
    output_dir: str | Path | None,
    api_url: str | None,
    timeout_seconds: float,
    parse_method: str,
    backend: str,
    lang: str,
    extra_args: Sequence[str] | None,
    table_engine: str,
    paddle_table_mode: str,
    paddle_device: str | None,
    mineru_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    options: dict[str, Any] = dict(mineru_options or {})
    if output_dir is not None:
        options["output_dir"] = str(Path(output_dir).expanduser().resolve())
    options.setdefault("parse_method", parse_method)
    options.setdefault("backend", backend)
    options.setdefault("lang", lang)
    options.setdefault("timeout_seconds", timeout_seconds)
    options.setdefault("api_url", _resolve_api_url(api_url))
    resolved_table_engine = str(options.get("table_engine") or table_engine).strip().lower()
    if resolved_table_engine not in {"ocr", "paddle"}:
        raise ValueError("table_engine must be 'ocr' or 'paddle'")
    options["table_engine"] = resolved_table_engine
    if extra_args is None:
        options.setdefault("extra_args", ["--formula", "false", "--table", "true"])
    else:
        options.setdefault("extra_args", list(extra_args))
    if resolved_table_engine == "paddle":
        options.setdefault("enable_table_cell_refine", True)
        options.setdefault("enable_paddle_table_refine", True)
        options.setdefault("table_cell_refine_fail_open", False)
        options.setdefault("emit_table_cells_as_text_blocks", False)
        options.setdefault("paddle_table_mode", paddle_table_mode)
        if paddle_device:
            options.setdefault("paddle_device", paddle_device)
    else:
        options.setdefault("enable_table_cell_refine", False)
        options.setdefault("enable_paddle_table_refine", False)
    return options


def _resolve_api_url(api_url: str | None) -> str:
    if api_url is not None and api_url.strip():
        return api_url.strip()
    env_api_url = os.environ.get("MINERU_API_URL")
    if env_api_url is not None and env_api_url.strip():
        return env_api_url.strip()
    return DEFAULT_MINERU_API_URL


def _build_internal_document(
    pdf_path: Path,
    *,
    mineru_options: Mapping[str, Any],
) -> DocumentItem:
    doc_id = _safe_doc_id(pdf_path)
    return DocumentItem(
        archive_id="mineru",
        archive_owner_user_id="mineru",
        triggered_by_user_id="mineru",
        doc_id=doc_id,
        file_uri=str(pdf_path),
        mime_type="application/pdf",
        meta={"source": "pure_mineru", "mineru_options": dict(mineru_options)},
    )


def _safe_doc_id(pdf_path: Path) -> str:
    stem = pdf_path.stem.strip() or pdf_path.name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe or "document"
