from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import (
    ArtifactRef,
    DocumentItem,
    ImageSize,
    PageItem,
    ParsedPage,
    ParsedPdf,
)
from ..operators import LayoutExtractMinerUOperator, OperatorContext


class MinerULayoutOperateResult(BaseModel):
    """Foundation OCR result independent from FastAPI or deployment concerns."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    run_id: str
    doc_id: str
    page_count: int = Field(ge=0)
    coord_space: str
    layout_ref: ArtifactRef | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    parsed_pdf: ParsedPdf
    pages: list[PageItem] = Field(default_factory=list)


def operate(
    document: DocumentItem | Mapping[str, Any],
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    config_version: str | None = None,
    tags: Mapping[str, str] | None = None,
    mineru_options: Mapping[str, Any] | None = None,
    operator_factory: Callable[[], LayoutExtractMinerUOperator] = LayoutExtractMinerUOperator,
) -> MinerULayoutOperateResult:
    """Extract OCR/layout data for one PDF document.

    This is the reusable foundation entrypoint. It can be called by Daft UDFs,
    platform services, tests, or scripts without importing FastAPI.
    """

    resolved_trace_id = trace_id or str(uuid4())
    resolved_run_id = run_id or str(uuid4())
    resolved_document = _build_document(document, mineru_options=mineru_options)
    ctx = OperatorContext(
        trace_id=resolved_trace_id,
        run_id=resolved_run_id,
        config_version=config_version,
        tags=dict(tags or {}),
    )

    raw_pages = list(
        operator_factory().process(
            ctx,
            iter([resolved_document.model_dump(mode="python")]),
            path="item",
        )
    )
    pages = [PageItem.model_validate(page) for page in raw_pages]
    first_page = pages[0] if pages else None
    coord_space = (
        str(first_page.page_meta.get("coord_space", "mineru_layout"))
        if first_page is not None
        else "mineru_layout"
    )
    layout_ref = first_page.layout_ref if first_page is not None else None
    artifacts = _extract_artifacts(first_page)
    parsed_pdf = ParsedPdf(
        pdf_path=resolved_document.file_uri,
        total_pages=len(pages),
        pages=[
            ParsedPage(
                page_index=page.page_index,
                text_blocks=page.text_blocks,
                table_blocks=page.table_blocks,
                image_size=_extract_image_size(page),
            )
            for page in pages
        ],
    )

    return MinerULayoutOperateResult(
        trace_id=resolved_trace_id,
        run_id=resolved_run_id,
        doc_id=resolved_document.doc_id,
        page_count=len(pages),
        coord_space=coord_space,
        layout_ref=layout_ref,
        artifacts=artifacts,
        parsed_pdf=parsed_pdf,
        pages=pages,
    )


def _build_document(
    document: DocumentItem | Mapping[str, Any],
    *,
    mineru_options: Mapping[str, Any] | None,
) -> DocumentItem:
    resolved_document = (
        document if isinstance(document, DocumentItem) else DocumentItem.model_validate(document)
    )
    if not mineru_options:
        return resolved_document

    document_meta = dict(resolved_document.meta)
    document_meta["mineru_options"] = dict(mineru_options)
    return resolved_document.model_copy(update={"meta": document_meta})


def _extract_artifacts(first_page: PageItem | None) -> list[ArtifactRef]:
    if first_page is None:
        return []
    raw_artifacts = first_page.page_meta.get("mineru_artifacts")
    if not isinstance(raw_artifacts, list):
        return [first_page.layout_ref] if first_page.layout_ref is not None else []

    artifacts: list[ArtifactRef] = []
    for raw_artifact in raw_artifacts:
        try:
            artifacts.append(ArtifactRef.model_validate(raw_artifact))
        except Exception:
            continue
    if not artifacts and first_page.layout_ref is not None:
        artifacts.append(first_page.layout_ref)
    return artifacts


def _extract_image_size(page: PageItem) -> ImageSize | None:
    raw_image_size = page.page_meta.get("image_size")
    if isinstance(raw_image_size, dict):
        try:
            return ImageSize.model_validate(raw_image_size)
        except Exception:
            return None

    width = page.page_meta.get("width")
    height = page.page_meta.get("height")
    if width is None or height is None:
        return None
    try:
        return ImageSize(width=int(round(float(width))), height=int(round(float(height))))
    except (TypeError, ValueError):
        return None
