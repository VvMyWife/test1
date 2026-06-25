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
DEFAULT_HYBRID_EFFORT = "medium"
SUPPORTED_TABLE_ENGINES = {"ocr", "ocr_pipeline", "ocr_vl", "ocr_hybrid", "paddle"}


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


class MinerUTableEngineResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_table_engine: str
    canonical_table_engine: str
    mineru_backend: str
    extra_args_suffix: list[str] = Field(default_factory=list)


def extract_pdf(
    input_path: str | Path,
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
    """Extract one local file (PDF, PNG, JPG, BMP, TIFF) and return a business-field-free MinerU result."""

    resolved_input = Path(input_path).expanduser().resolve()
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
    document = _build_internal_document(resolved_input, mineru_options=options)
    result = operate(document, mineru_options=options, operator_factory=operator_factory)
    return to_pure_mineru_result(result, source_pdf=resolved_input)


def to_pure_mineru_result(
    result: MinerULayoutOperateResult,
    *,
    source_pdf: str | Path,
) -> MinerUPdfResult:
    resolved_source = Path(source_pdf).expanduser().resolve()
    return MinerUPdfResult(
        source_pdf=str(resolved_source),
        source_file_name=resolved_source.name,
        page_count=result.page_count,
        coord_space=result.coord_space,
        layout_ref=result.layout_ref,
        artifacts=result.artifacts,
        parsed_pdf=result.parsed_pdf.model_copy(update={"pdf_path": str(resolved_source)}),
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
    resolution = resolve_mineru_table_engine(
        str(options.get("table_engine") or table_engine or "ocr")
    )
    options.setdefault("parse_method", parse_method)
    options.setdefault("backend", resolution.mineru_backend or backend)
    options.setdefault("lang", lang)
    options.setdefault("timeout_seconds", timeout_seconds)
    options.setdefault("api_url", _resolve_api_url(api_url))
    options["table_engine"] = resolution.canonical_table_engine
    if extra_args is None:
        extra_args_value = ["--formula", "false", "--table", "true"]
    else:
        extra_args_value = list(extra_args)
    if resolution.extra_args_suffix:
        existing_args = {str(arg).strip().lower() for arg in extra_args_value}
        for index, arg in enumerate(resolution.extra_args_suffix):
            if index % 2 == 0 and str(arg).strip().lower() in existing_args:
                continue
            extra_args_value.append(arg)
    options.setdefault("extra_args", extra_args_value)
    if resolution.canonical_table_engine == "paddle":
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


def resolve_mineru_table_engine(table_engine: str | None) -> MinerUTableEngineResolution:
    requested = str(table_engine or "ocr").strip().lower()
    alias_map = {
        "ocr": ("ocr", "pipeline", []),
        "ocr_pipeline": ("ocr", "pipeline", []),
        "ocr_vl": ("ocr", "vlm-engine", []),
        "ocr_hybrid": ("ocr", "hybrid-engine", ["--effort", DEFAULT_HYBRID_EFFORT]),
        "paddle": ("paddle", "pipeline", []),
    }
    if requested not in alias_map:
        supported = "', '".join(sorted(SUPPORTED_TABLE_ENGINES))
        raise ValueError(f"table_engine must be one of '{supported}'")
    canonical, backend, extra_args_suffix = alias_map[requested]
    return MinerUTableEngineResolution(
        requested_table_engine=requested,
        canonical_table_engine=canonical,
        mineru_backend=backend,
        extra_args_suffix=list(extra_args_suffix),
    )


def _resolve_api_url(api_url: str | None) -> str:
    if api_url is not None and api_url.strip():
        return api_url.strip()
    env_api_url = os.environ.get("MINERU_API_URL")
    if env_api_url is not None and env_api_url.strip():
        return env_api_url.strip()
    return DEFAULT_MINERU_API_URL


def _build_internal_document(
    input_path: Path,
    *,
    mineru_options: Mapping[str, Any],
) -> DocumentItem:
    doc_id = _safe_doc_id(input_path)
    return DocumentItem(
        archive_id="mineru",
        archive_owner_user_id="mineru",
        triggered_by_user_id="mineru",
        doc_id=doc_id,
        file_uri=str(input_path),
        mime_type=_resolve_mime_type(input_path),
        meta={"source": "pure_mineru", "mineru_options": dict(mineru_options)},
    )


def _resolve_mime_type(file_path: Path) -> str:
    """Map file suffix to standard MIME type for MinerU processing."""
    suffix = file_path.suffix.lower()
    _MIME_MAP: dict[str, str] = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    return _MIME_MAP.get(suffix, "application/octet-stream")


def _safe_doc_id(file_path: Path) -> str:
    stem = file_path.stem.strip() or file_path.name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe or "document"
