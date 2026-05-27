from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse
from urllib.request import url2pathname

from ..contracts import (
    ArtifactRef,
    BoundingBox,
    CoordSpace,
    ImageSize,
    ParsedPage,
    ParsedPdf,
    TableBlock,
    TextBlock,
)
from .paddle_table import (
    PaddleTableApiClient,
    PaddleTableStructureError,
    PaddleTableStructureResult,
    PaddleTableStructureService,
)

JsonDict = dict[str, Any]
CommandRunner = Callable[[list[str], float | None], None]


class MinerUServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "MINERU_SERVICE_ERROR",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = dict(details or {})


@dataclass(frozen=True)
class MinerUPageResult:
    page_index: int
    text: str | None = None
    text_blocks: tuple[TextBlock, ...] = ()
    table_blocks: tuple[TableBlock, ...] = ()
    image_size: ImageSize | None = None
    page_meta: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class MinerUDocumentParseResult:
    pages: tuple[MinerUPageResult, ...]
    middle_json_ref: ArtifactRef
    page_count: int
    coord_space: CoordSpace = CoordSpace.MINERU_LAYOUT
    parsed_pdf: ParsedPdf | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    meta: JsonDict = field(default_factory=dict)


@runtime_checkable
class MinerUDocumentService(Protocol):
    def parse_document(
        self,
        *,
        file_uri: str,
        mime_type: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MinerUDocumentParseResult: ...


def _run_subprocess(command: list[str], timeout_seconds: float | None) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:  # pragma: no cover
        raise MinerUServiceError(
            f"MinerU command not found: {command[0]}",
            code="MINERU_COMMAND_NOT_FOUND",
            retryable=False,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MinerUServiceError(
            "MinerU command timed out",
            code="MINERU_TIMEOUT",
            retryable=True,
            details={"timeout_seconds": timeout_seconds},
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise MinerUServiceError(
            "MinerU command failed",
            code="MINERU_COMMAND_FAILED",
            retryable=False,
            details={
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        ) from exc


@dataclass(frozen=True)
class InlineMinerUDocumentService:
    result: MinerUDocumentParseResult

    def parse_document(
        self,
        *,
        file_uri: str,
        mime_type: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MinerUDocumentParseResult:
        return self.result


@dataclass(frozen=True)
class MinerUCliConfig:
    command: tuple[str, ...] = ("mineru",)
    output_root: str | None = None
    parse_method: str | None = None
    backend: str | None = None
    lang: str | None = None
    api_url: str | None = None
    timeout_seconds: float | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MinerUCliDocumentService:
    config: MinerUCliConfig = MinerUCliConfig()
    command_runner: CommandRunner = _run_subprocess

    def parse_document(
        self,
        *,
        file_uri: str,
        mime_type: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MinerUDocumentParseResult:
        resolved_file = _coerce_local_file_path(file_uri)
        if resolved_file.suffix.lower() != ".pdf":
            raise MinerUServiceError(
                "MinerU CLI currently expects a local PDF path",
                code="MINERU_UNSUPPORTED_INPUT",
                retryable=False,
                details={"file_uri": file_uri, "mime_type": mime_type},
            )

        opts = dict(options or {})
        output_dir = _prepare_output_dir(self.config.output_root, opts.get("output_dir"))
        command = self._build_command(resolved_file, output_dir, opts)
        self.command_runner(command, _coerce_float(opts.get("timeout_seconds"), self.config.timeout_seconds))

        middle_json_path = _locate_single_artifact(output_dir, "*_middle.json")
        content_list_path = _locate_optional_single_artifact(output_dir, "*_content_list.json")
        middle_payload = json.loads(middle_json_path.read_text(encoding="utf-8"))
        text_blocks_by_page = None
        table_candidates_by_page = None
        if content_list_path is not None:
            content_list_payload = json.loads(content_list_path.read_text(encoding="utf-8"))
            text_blocks_by_page = parse_mineru_content_list_json(content_list_payload)
            table_candidates_by_page = parse_mineru_table_candidates_json(
                content_list_payload,
                base_dir=content_list_path.parent,
            )
        parsed = parse_mineru_middle_json(
            middle_payload,
            middle_json_uri=str(middle_json_path),
            parser_version=str(opts.get("parser_version") or "mineru-cli"),
            text_blocks_by_page=text_blocks_by_page,
            table_candidates_by_page=table_candidates_by_page,
        )
        parsed = _maybe_refine_table_cells_with_paddle(
            input_path=resolved_file,
            output_dir=output_dir,
            parsed=parsed,
            options=opts,
        )

        meta = dict(parsed.meta)
        meta["output_dir"] = str(output_dir)
        artifacts = _collect_mineru_artifacts(
            output_dir=output_dir,
            middle_json_ref=parsed.middle_json_ref,
        )
        artifacts = _merge_artifacts(artifacts, parsed.artifacts)
        return MinerUDocumentParseResult(
            pages=parsed.pages,
            middle_json_ref=parsed.middle_json_ref,
            page_count=parsed.page_count,
            coord_space=parsed.coord_space,
            parsed_pdf=parsed.parsed_pdf,
            artifacts=artifacts,
            meta=meta,
        )

    def _build_command(self, input_path: Path, output_dir: Path, options: Mapping[str, Any]) -> list[str]:
        command = [*self.config.command, "-p", str(input_path), "-o", str(output_dir)]

        parse_method = _coerce_optional_str(options.get("parse_method"), self.config.parse_method)
        backend = _coerce_optional_str(options.get("backend"), self.config.backend)
        lang = _coerce_optional_str(options.get("lang"), self.config.lang)
        api_url = _coerce_optional_str(options.get("api_url"), self.config.api_url)

        if parse_method:
            command.extend(["--method", parse_method])
        if backend:
            command.extend(["--backend", backend])
        if lang:
            command.extend(["--lang", lang])
        if api_url:
            command.extend(["--api-url", api_url])

        extra_args = options.get("extra_args")
        if isinstance(extra_args, (list, tuple)):
            command.extend(str(arg) for arg in extra_args)
        command.extend(self.config.extra_args)
        return command


def parse_mineru_middle_json(
    middle_json: Mapping[str, Any],
    *,
    middle_json_uri: str,
    parser_version: str = "mineru",
    text_blocks_by_page: Mapping[int, tuple[TextBlock, ...]] | None = None,
    table_candidates_by_page: Mapping[int, tuple[JsonDict, ...]] | None = None,
) -> MinerUDocumentParseResult:
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list):
        raise MinerUServiceError(
            "MinerU middle.json must contain a pdf_info list",
            code="MINERU_INVALID_MIDDLE_JSON",
            retryable=False,
        )

    pages: list[MinerUPageResult] = []
    for fallback_index, page_payload in enumerate(pdf_info):
        if not isinstance(page_payload, Mapping):
            raise MinerUServiceError(
                "Each MinerU page payload must be a mapping",
                code="MINERU_INVALID_MIDDLE_JSON",
                retryable=False,
                details={"page_index": fallback_index},
            )

        page_index = _coerce_page_index(page_payload.get("page_idx"), fallback_index)
        text_blocks = tuple(text_blocks_by_page.get(page_index, ())) if text_blocks_by_page else ()
        if not text_blocks:
            text_blocks = _extract_text_blocks(page_payload)
        text = _extract_page_text(page_payload, text_blocks=text_blocks)
        width, height = _extract_page_size(page_payload)
        image_size = _build_image_size(width, height)
        page_meta: JsonDict = {
            "layout_provider": "mineru",
            "page_text_source": "para_blocks" if page_payload.get("para_blocks") else "preproc_blocks",
        }
        if width is not None:
            page_meta["width"] = width
        if height is not None:
            page_meta["height"] = height
        table_candidates = (
            tuple(table_candidates_by_page.get(page_index, ())) if table_candidates_by_page else ()
        )
        if not table_candidates:
            table_candidates = _extract_table_candidates(page_payload)
        if table_candidates:
            page_meta["mineru_has_table"] = True
            page_meta["mineru_table_count"] = len(table_candidates)
            page_meta["mineru_table_candidates"] = list(table_candidates)
        table_blocks = _extract_mineru_table_blocks(
            page_payload,
            page_index=page_index,
            table_candidates=table_candidates,
        )
        if table_blocks:
            page_meta["mineru_table_output"] = {
                "provider": "mineru",
                "table_count": len(table_blocks),
                "html_count": sum(1 for table in table_blocks if table.html),
            }

        pages.append(
            MinerUPageResult(
                page_index=page_index,
                text=text,
                text_blocks=text_blocks,
                table_blocks=table_blocks,
                image_size=image_size,
                page_meta=page_meta,
            )
        )

    pages.sort(key=lambda item: item.page_index)
    artifact_meta = {
        "content_type": "application/json",
        "page_count": len(pages),
        "parser_version": parser_version,
        "coord_space": CoordSpace.MINERU_LAYOUT.value,
    }
    parsed_pdf = ParsedPdf(
        pdf_path=None,
        total_pages=len(pages),
        pages=[
            ParsedPage(
                page_index=page.page_index,
                text_blocks=list(page.text_blocks),
                table_blocks=list(page.table_blocks),
                image_size=page.image_size,
            )
            for page in pages
        ],
    )
    return MinerUDocumentParseResult(
        pages=tuple(pages),
        middle_json_ref=ArtifactRef(kind="middle_json", uri=middle_json_uri, meta=artifact_meta),
        page_count=len(pages),
        coord_space=CoordSpace.MINERU_LAYOUT,
        parsed_pdf=parsed_pdf,
        artifacts=(ArtifactRef(kind="middle_json", uri=middle_json_uri, meta=artifact_meta),),
        meta={"layout_provider": "mineru", "parser_version": parser_version},
    )


def parse_mineru_content_list_json(content_list: Any) -> dict[int, tuple[TextBlock, ...]]:
    """Parse MinerU ``*_content_list.json`` into Daft-friendly text blocks.

    The official MinerU output stores block-level text as a flat list with
    ``page_idx`` and ``bbox=[x0,y0,x1,y1]``. This is the closest grain to Daft's
    ``ocr_page`` example, where each page contains ``text_blocks`` with a
    ``bounding_box``.
    """

    if not isinstance(content_list, list):
        raise MinerUServiceError(
            "MinerU content_list.json must contain a list",
            code="MINERU_INVALID_CONTENT_LIST",
            retryable=False,
        )

    grouped: dict[int, list[TextBlock]] = {}
    for fallback_index, item in enumerate(content_list):
        if not isinstance(item, Mapping):
            continue

        text = _coerce_optional_str(item.get("text"))
        bbox = _coerce_bbox(item.get("bbox"))
        if text is None or bbox is None:
            continue

        page_index = _coerce_page_index(item.get("page_idx"), fallback_index)
        meta: JsonDict = {}
        for key in ("text_level", "img_path", "table_caption", "table_footnote"):
            if key in item:
                meta[key] = item[key]

        grouped.setdefault(page_index, []).append(
            TextBlock(
                text=text,
                bounding_box=bbox,
                block_type=_coerce_optional_str(item.get("type")),
                confidence=_coerce_confidence(item.get("score")),
                meta=meta,
            )
        )

    return {page_index: tuple(blocks) for page_index, blocks in grouped.items()}


def parse_mineru_table_candidates_json(
    content_list: Any,
    *,
    base_dir: Path,
) -> dict[int, tuple[JsonDict, ...]]:
    if not isinstance(content_list, list):
        raise MinerUServiceError(
            "MinerU content_list.json must contain a list",
            code="MINERU_INVALID_CONTENT_LIST",
            retryable=False,
        )

    grouped: dict[int, list[JsonDict]] = {}
    for fallback_index, item in enumerate(content_list):
        if not isinstance(item, Mapping) or item.get("type") != "table":
            continue
        bbox = _coerce_bbox(item.get("bbox"))
        if bbox is None:
            continue
        page_index = _coerce_page_index(item.get("page_idx"), fallback_index)
        image_uri = _resolve_relative_artifact(base_dir, item.get("img_path"))
        captions = _as_str_list(item.get("table_caption"))
        footnotes = _as_str_list(item.get("table_footnote"))
        grouped.setdefault(page_index, []).append(
            {
                "source": "mineru_content_list",
                "page_index": page_index,
                "bbox": bbox.model_dump(mode="python"),
                "image_uri": image_uri,
                "caption": "\n".join(captions) if captions else None,
                "footnote": "\n".join(footnotes) if footnotes else None,
            }
        )
    return {page_index: tuple(candidates) for page_index, candidates in grouped.items()}


def _extract_mineru_table_blocks(
    page_payload: Mapping[str, Any],
    *,
    page_index: int,
    table_candidates: tuple[JsonDict, ...],
) -> tuple[TableBlock, ...]:
    raw_table_blocks = _extract_raw_table_blocks(page_payload)
    tables: list[TableBlock] = []
    used_candidate_indexes: set[int] = set()

    for table_index, raw_table in enumerate(raw_table_blocks):
        bbox = _coerce_table_bbox(raw_table.get("bbox")) or _coerce_bbox(_bbox_from_children(raw_table))
        if bbox is None:
            continue
        html = _extract_first_html(raw_table)
        matched_candidate = _match_table_candidate(
            bbox,
            table_candidates=table_candidates,
            used_indexes=used_candidate_indexes,
        )
        meta: JsonDict = {
            "source": "mineru_middle_json_table",
            "raw_source": _coerce_optional_str(raw_table.get("type"), "table"),
            "raw_table": dict(raw_table),
        }
        if matched_candidate is not None:
            meta["candidate"] = dict(matched_candidate)
        tables.append(
            TableBlock(
                table_id=f"p{page_index}-mineru-t{table_index}",
                page_index=page_index,
                provider="mineru",
                bounding_box=bbox,
                coord_space=CoordSpace.MINERU_LAYOUT.value,
                html=html,
                cells=[],
                confidence=_coerce_confidence(raw_table.get("score")),
                meta=meta,
            )
        )

    if raw_table_blocks:
        return tuple(tables)

    for candidate_index, candidate in enumerate(table_candidates):
        if candidate_index in used_candidate_indexes:
            continue
        bbox = _coerce_table_bbox(candidate.get("bbox"))
        if bbox is None:
            continue
        table_index = len(tables)
        tables.append(
            TableBlock(
                table_id=f"p{page_index}-mineru-t{table_index}",
                page_index=page_index,
                provider="mineru",
                bounding_box=bbox,
                coord_space=CoordSpace.MINERU_LAYOUT.value,
                cells=[],
                meta={
                    "source": "mineru_content_list_table",
                    "raw_source": candidate.get("source"),
                    "candidate": dict(candidate),
                },
            )
        )

    return tuple(tables)


def _extract_raw_table_blocks(page_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blocks = page_payload.get("para_blocks")
    if not isinstance(blocks, list):
        blocks = page_payload.get("preproc_blocks")
    if not isinstance(blocks, list):
        return []
    return [block for block in blocks if isinstance(block, Mapping) and block.get("type") == "table"]


def _match_table_candidate(
    bbox: BoundingBox,
    *,
    table_candidates: tuple[JsonDict, ...],
    used_indexes: set[int],
) -> Mapping[str, Any] | None:
    for index, candidate in enumerate(table_candidates):
        if index in used_indexes:
            continue
        candidate_bbox = _coerce_table_bbox(candidate.get("bbox"))
        if candidate_bbox is None:
            continue
        if _bbox_iou(bbox, candidate_bbox) >= 0.70:
            used_indexes.add(index)
            return candidate
    return None


def _maybe_refine_table_cells_with_paddle(
    *,
    input_path: Path,
    output_dir: Path,
    parsed: MinerUDocumentParseResult,
    options: Mapping[str, Any],
) -> MinerUDocumentParseResult:
    if not (
        _coerce_bool(options.get("enable_table_cell_refine"))
        or _coerce_bool(options.get("enable_paddle_table_refine"))
    ):
        return parsed

    table_candidates_by_page = _extract_table_candidates_by_page(parsed)
    if (
        _coerce_bool(options.get("table_cell_refine_when_tables_present"), True)
        and not _coerce_bool(options.get("table_cell_refine_force"))
        and not table_candidates_by_page
    ):
        return _attach_table_refine_skipped(parsed, reason="mineru_table_not_detected")

    provider = _coerce_optional_str(options.get("table_cell_refine_provider"), "paddleocr")
    if provider not in {"paddleocr", "paddleocr_ppstructurev3"}:
        raise MinerUServiceError(
            f"Unsupported table cell refinement provider: {provider}",
            code="TABLE_CELL_REFINEMENT_PROVIDER_UNSUPPORTED",
            retryable=False,
            details={"provider": provider},
        )

    try:
        table_service = _build_paddle_table_service(options)
        table_result = table_service.extract_tables(
            file_uri=str(input_path),
            output_dir=output_dir,
            target_page_sizes={page.page_index: page.image_size for page in parsed.pages},
            options=options,
            table_candidates_by_page=table_candidates_by_page,
            page_text_blocks_by_page={page.page_index: page.text_blocks for page in parsed.pages},
        )
    except PaddleTableStructureError as exc:
        if _coerce_bool(options.get("table_cell_refine_fail_open"), True):
            return _attach_table_refine_warning(parsed, exc)
        raise MinerUServiceError(
            "PaddleOCR table cell refinement failed",
            code="PADDLE_TABLE_REFINEMENT_FAILED",
            retryable=False,
            details={"error": str(exc)},
        ) from exc

    return _merge_table_result(
        parsed,
        table_result,
        emit_text_blocks=_coerce_bool(options.get("emit_table_cells_as_text_blocks"), True),
        replace_existing_table_blocks=_coerce_bool(
            options.get("replace_ocr_table_blocks_with_paddle"),
            True,
        ),
    )


def _build_paddle_table_service(options: Mapping[str, Any]) -> PaddleTableStructureService | PaddleTableApiClient:
    api_url = _coerce_optional_str(options.get("paddle_table_api_url"))
    if api_url is None:
        api_url = _coerce_optional_str(os.environ.get("PADDLE_TABLE_API_URL"))
    if api_url:
        timeout_seconds = _coerce_float(options.get("paddle_table_api_timeout_seconds"))
        if timeout_seconds is None:
            timeout_seconds = _coerce_float(options.get("timeout_seconds")) or 1800.0
        return PaddleTableApiClient(api_url=api_url, timeout_seconds=timeout_seconds)
    return PaddleTableStructureService()


def _extract_table_candidates_by_page(parsed: MinerUDocumentParseResult) -> dict[int, tuple[JsonDict, ...]]:
    grouped: dict[int, tuple[JsonDict, ...]] = {}
    for page in parsed.pages:
        raw_candidates = page.page_meta.get("mineru_table_candidates")
        if isinstance(raw_candidates, list):
            candidates = tuple(item for item in raw_candidates if isinstance(item, Mapping))
            if candidates:
                grouped[page.page_index] = candidates
    return grouped


def _merge_table_result(
    parsed: MinerUDocumentParseResult,
    table_result: PaddleTableStructureResult,
    *,
    emit_text_blocks: bool,
    replace_existing_table_blocks: bool,
) -> MinerUDocumentParseResult:
    pages: list[MinerUPageResult] = []
    for page in parsed.pages:
        table_blocks = tuple(table_result.tables_by_page.get(page.page_index, ()))
        if not table_blocks:
            pages.append(page)
            continue

        text_blocks = (
            _drop_ocr_table_fallback_text_blocks(page.text_blocks)
            if replace_existing_table_blocks
            else list(page.text_blocks)
        )
        if replace_existing_table_blocks:
            text_blocks = _drop_text_blocks_inside_table_blocks(text_blocks, table_blocks)
        if emit_text_blocks:
            text_blocks.extend(_table_cells_to_text_blocks(table_blocks))
        existing_table_blocks = () if replace_existing_table_blocks else page.table_blocks
        page_text = _text_from_text_blocks(text_blocks)

        page_meta = dict(page.page_meta)
        table_cell_refine_meta = {
            "provider": table_result.meta.get("provider", "paddleocr_ppstructurev3"),
            "table_count": len(table_blocks),
            "cell_count": sum(len(table.cells) for table in table_blocks),
            "artifact": (
                table_result.artifact_ref.model_dump(mode="python")
                if table_result.artifact_ref is not None
                else None
            ),
        }
        for key in ("mode", "transport", "api_url"):
            if table_result.meta.get(key) is not None:
                table_cell_refine_meta[key] = table_result.meta[key]
        page_meta["table_cell_refine"] = table_cell_refine_meta
        pages.append(
            MinerUPageResult(
                page_index=page.page_index,
                text=page_text,
                text_blocks=tuple(text_blocks),
                table_blocks=tuple([*existing_table_blocks, *table_blocks]),
                image_size=page.image_size,
                page_meta=page_meta,
            )
        )

    parsed_pdf = ParsedPdf(
        pdf_path=parsed.parsed_pdf.pdf_path if parsed.parsed_pdf is not None else None,
        total_pages=len(pages),
        pages=[
            ParsedPage(
                page_index=page.page_index,
                text_blocks=list(page.text_blocks),
                table_blocks=list(page.table_blocks),
                image_size=page.image_size,
            )
            for page in pages
        ],
    )
    artifacts = list(parsed.artifacts)
    if table_result.artifact_ref is not None:
        artifacts.append(table_result.artifact_ref)
    meta = dict(parsed.meta)
    meta["table_cell_refine"] = dict(table_result.meta)
    return MinerUDocumentParseResult(
        pages=tuple(pages),
        middle_json_ref=parsed.middle_json_ref,
        page_count=parsed.page_count,
        coord_space=parsed.coord_space,
        parsed_pdf=parsed_pdf,
        artifacts=_merge_artifacts(tuple(artifacts)),
        meta=meta,
    )


def _table_cells_to_text_blocks(table_blocks: tuple[TableBlock, ...]) -> list[TextBlock]:
    text_blocks: list[TextBlock] = []
    for table in table_blocks:
        for cell in table.cells:
            if cell.bounding_box is None or not cell.text.strip():
                continue
            text_blocks.append(
                TextBlock(
                    text=cell.text,
                    bounding_box=cell.bounding_box,
                    block_type="table_cell",
                    confidence=cell.confidence,
                    meta={
                        "source": "paddleocr_table",
                        "table_id": table.table_id,
                        "cell_id": cell.cell_id,
                        "row_index": cell.row_index,
                        "col_index": cell.col_index,
                        "coord_space": table.coord_space,
                    },
                )
            )
    return text_blocks


def _drop_ocr_table_fallback_text_blocks(text_blocks: tuple[TextBlock, ...]) -> list[TextBlock]:
    return [
        block
        for block in text_blocks
        if block.meta.get("source") != "mineru_ocr_table_fallback"
    ]


def _drop_text_blocks_inside_table_blocks(
    text_blocks: list[TextBlock],
    table_blocks: tuple[TableBlock, ...],
) -> list[TextBlock]:
    table_boxes = [table.bounding_box for table in table_blocks if table.bounding_box is not None]
    if not table_boxes:
        return text_blocks
    return [
        block
        for block in text_blocks
        if block.bounding_box is None
        or block.block_type == "table_cell"
        or not any(_bbox_center_inside(block.bounding_box, table_box) for table_box in table_boxes)
    ]


def _text_from_text_blocks(text_blocks: list[TextBlock]) -> str | None:
    text = "\n".join(block.text for block in text_blocks if block.text.strip())
    return text or None


def _attach_table_refine_warning(
    parsed: MinerUDocumentParseResult,
    exc: Exception,
) -> MinerUDocumentParseResult:
    page_warning = {
        "enabled": True,
        "success": False,
        "fail_open": True,
        "error": str(exc),
    }
    pages = tuple(
        MinerUPageResult(
            page_index=page.page_index,
            text=page.text,
            text_blocks=page.text_blocks,
            table_blocks=page.table_blocks,
            image_size=page.image_size,
            page_meta={**page.page_meta, "table_cell_refine": page_warning},
        )
        for page in parsed.pages
    )
    parsed_pdf = ParsedPdf(
        pdf_path=parsed.parsed_pdf.pdf_path if parsed.parsed_pdf is not None else None,
        total_pages=len(pages),
        pages=[
            ParsedPage(
                page_index=page.page_index,
                text_blocks=list(page.text_blocks),
                table_blocks=list(page.table_blocks),
                image_size=page.image_size,
            )
            for page in pages
        ],
    )
    meta = dict(parsed.meta)
    meta["table_cell_refine"] = page_warning
    return MinerUDocumentParseResult(
        pages=pages,
        middle_json_ref=parsed.middle_json_ref,
        page_count=parsed.page_count,
        coord_space=parsed.coord_space,
        parsed_pdf=parsed_pdf,
        artifacts=parsed.artifacts,
        meta=meta,
    )


def _attach_table_refine_skipped(
    parsed: MinerUDocumentParseResult,
    *,
    reason: str,
) -> MinerUDocumentParseResult:
    meta = dict(parsed.meta)
    meta["table_cell_refine"] = {
        "enabled": True,
        "success": True,
        "skipped": True,
        "reason": reason,
    }
    return MinerUDocumentParseResult(
        pages=parsed.pages,
        middle_json_ref=parsed.middle_json_ref,
        page_count=parsed.page_count,
        coord_space=parsed.coord_space,
        parsed_pdf=parsed.parsed_pdf,
        artifacts=parsed.artifacts,
        meta=meta,
    )


def _extract_first_html(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        html = _coerce_optional_str(payload.get("html"))
        if html is not None:
            return html
        for value in payload.values():
            found = _extract_first_html(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_first_html(item)
            if found is not None:
                return found
    return None


def _coerce_table_bbox(value: Any) -> BoundingBox | None:
    if isinstance(value, BoundingBox):
        return value
    if isinstance(value, Mapping):
        if {"x", "y", "w", "h"}.issubset(value):
            x = _coerce_float(value.get("x"))
            y = _coerce_float(value.get("y"))
            w = _coerce_float(value.get("w"))
            h = _coerce_float(value.get("h"))
            if x is None or y is None or w is None or h is None:
                return None
            return BoundingBox(
                x=int(round(x)),
                y=int(round(y)),
                w=max(0, int(round(w))),
                h=max(0, int(round(h))),
            )
        return _coerce_bbox(list(value.values()))
    return _coerce_bbox(value)


def _bbox_center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx = inner.x + inner.w / 2.0
    cy = inner.y + inner.h / 2.0
    return outer.x <= cx <= outer.x + outer.w and outer.y <= cy <= outer.y + outer.h


def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.w, right.x + right.w)
    y1 = min(left.y + left.h, right.y + right.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = left.w * left.h + right.w * right.h - intersection
    return float(intersection / union) if union > 0 else 0.0


def _coerce_non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _coerce_local_file_path(file_uri: str) -> Path:
    if _looks_like_windows_absolute_path(file_uri):
        path = Path(file_uri).expanduser().resolve()
        if not path.exists():
            raise MinerUServiceError(
                f"Input PDF does not exist: {file_uri}",
                code="MINERU_INPUT_NOT_FOUND",
                retryable=False,
            )
        return path

    parsed = urlparse(file_uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            raw_path = url2pathname(parsed.path)
            if parsed.netloc:
                raw_path = f"//{parsed.netloc}{raw_path}"
        else:
            raw_path = file_uri
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise MinerUServiceError(
                f"Input PDF does not exist: {file_uri}",
                code="MINERU_INPUT_NOT_FOUND",
                retryable=False,
            )
        return path

    raise MinerUServiceError(
        "MinerU CLI only supports local file paths or file:// URIs",
        code="MINERU_UNSUPPORTED_INPUT",
        retryable=False,
        details={"file_uri": file_uri},
    )


def _looks_like_windows_absolute_path(file_uri: str) -> bool:
    return len(file_uri) >= 3 and file_uri[1] == ":" and file_uri[2] in ("\\", "/")


def _prepare_output_dir(base_output_root: str | None, explicit_output_dir: Any) -> Path:
    if explicit_output_dir is not None and str(explicit_output_dir).strip():
        output_dir = Path(str(explicit_output_dir)).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    if base_output_root:
        base_dir = Path(base_output_root).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="mineru-", dir=base_dir))

    return Path(tempfile.mkdtemp(prefix="mineru-"))


def _locate_single_artifact(output_dir: Path, pattern: str) -> Path:
    matches = sorted(output_dir.rglob(pattern))
    if not matches:
        raise MinerUServiceError(
            f"MinerU output missing expected artifact: {pattern}",
            code="MINERU_OUTPUT_NOT_FOUND",
            retryable=False,
            details={"output_dir": str(output_dir)},
        )
    if len(matches) > 1:
        raise MinerUServiceError(
            f"MinerU output produced multiple artifacts for pattern: {pattern}",
            code="MINERU_AMBIGUOUS_OUTPUT",
            retryable=False,
            details={"output_dir": str(output_dir), "matches": [str(path) for path in matches]},
        )
    return matches[0]


def _coerce_optional_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _resolve_relative_artifact(base_dir: Path, value: Any) -> str | None:
    text = _coerce_optional_str(value)
    if text is None:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_confidence(value: Any) -> float | None:
    score = _coerce_float(value)
    if score is None or score < 0.0 or score > 1.0:
        return None
    return score


def _coerce_page_index(value: Any, fallback_index: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback_index


def _extract_page_size(page_payload: Mapping[str, Any]) -> tuple[float | None, float | None]:
    page_size = page_payload.get("page_size")
    if isinstance(page_size, (list, tuple)) and len(page_size) >= 2:
        try:
            return float(page_size[0]), float(page_size[1])
        except (TypeError, ValueError):
            return None, None
    return None, None


def _extract_page_text(
    page_payload: Mapping[str, Any],
    *,
    text_blocks: tuple[TextBlock, ...] = (),
) -> str | None:
    if text_blocks:
        text = "\n".join(block.text for block in text_blocks if block.text.strip())
        return text or None

    blocks = page_payload.get("para_blocks")
    if not isinstance(blocks, list):
        blocks = page_payload.get("preproc_blocks")
    if not isinstance(blocks, list):
        return None

    fragments = [fragment for fragment in _iter_text_fragments(blocks) if fragment]
    if not fragments:
        return None
    return "\n".join(fragments)


def _extract_text_blocks(page_payload: Mapping[str, Any]) -> tuple[TextBlock, ...]:
    blocks = page_payload.get("para_blocks")
    if not isinstance(blocks, list):
        blocks = page_payload.get("preproc_blocks")
    if not isinstance(blocks, list):
        return ()

    text_blocks: list[TextBlock] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        text = "\n".join(fragment for fragment in _iter_text_fragments(block) if fragment).strip()
        bbox = _coerce_bbox(block.get("bbox")) or _coerce_bbox(_bbox_from_children(block))
        if not text or bbox is None:
            continue
        meta: JsonDict = {"source": "middle_json"}
        if "level" in block:
            meta["text_level"] = block["level"]
        text_blocks.append(
            TextBlock(
                text=text,
                bounding_box=bbox,
                block_type=_coerce_optional_str(block.get("type")),
                confidence=_coerce_confidence(block.get("score")),
                meta=meta,
            )
        )
    return tuple(text_blocks)


def _extract_table_candidates(page_payload: Mapping[str, Any]) -> tuple[JsonDict, ...]:
    blocks = page_payload.get("para_blocks")
    if not isinstance(blocks, list):
        blocks = page_payload.get("preproc_blocks")
    if not isinstance(blocks, list):
        return ()

    candidates: list[JsonDict] = []
    page_index = _coerce_page_index(page_payload.get("page_idx"), 0)
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping) or block.get("type") != "table":
            continue
        bbox = _coerce_bbox(block.get("bbox")) or _coerce_bbox(_bbox_from_children(block))
        if bbox is None:
            continue
        candidates.append(
            {
                "source": "mineru_middle_json",
                "page_index": page_index,
                "table_index": index,
                "bbox": bbox.model_dump(mode="python"),
                "image_uri": None,
                "caption": "\n".join(_iter_text_fragments(block.get("table_caption"))),
                "footnote": "\n".join(_iter_text_fragments(block.get("table_footnote"))),
            }
        )
    return tuple(candidates)


def _iter_text_fragments(payload: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            fragments.extend(_iter_text_fragments(item))
        return fragments

    if isinstance(payload, Mapping):
        for key in ("content", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    fragments.append(normalized)
                break

        for key in (
            "blocks",
            "lines",
            "spans",
            "image_caption",
            "image_footnote",
            "table_caption",
            "table_footnote",
        ):
            child = payload.get(key)
            if child is not None:
                fragments.extend(_iter_text_fragments(child))
    return fragments


def _build_image_size(width: float | None, height: float | None) -> ImageSize | None:
    if width is None or height is None:
        return None
    return ImageSize(width=int(round(width)), height=int(round(height)))


def _coerce_bbox(value: Any) -> BoundingBox | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0 = float(value[0])
        y0 = float(value[1])
        x1 = float(value[2])
        y1 = float(value[3])
    except (TypeError, ValueError):
        return None
    return BoundingBox(
        x=int(round(x0)),
        y=int(round(y0)),
        w=max(0, int(round(x1 - x0))),
        h=max(0, int(round(y1 - y0))),
    )


def _bbox_from_children(payload: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    bboxes: list[tuple[float, float, float, float]] = []
    for child in _iter_bboxes(payload):
        if isinstance(child, (list, tuple)) and len(child) >= 4:
            try:
                bboxes.append((float(child[0]), float(child[1]), float(child[2]), float(child[3])))
            except (TypeError, ValueError):
                continue
    if not bboxes:
        return None
    return (
        min(item[0] for item in bboxes),
        min(item[1] for item in bboxes),
        max(item[2] for item in bboxes),
        max(item[3] for item in bboxes),
    )


def _iter_bboxes(payload: Any) -> list[Any]:
    bboxes: list[Any] = []
    if isinstance(payload, list):
        for item in payload:
            bboxes.extend(_iter_bboxes(item))
        return bboxes
    if isinstance(payload, Mapping):
        bbox = payload.get("bbox")
        if bbox is not None:
            bboxes.append(bbox)
        for key in ("blocks", "lines", "spans"):
            child = payload.get(key)
            if child is not None:
                bboxes.extend(_iter_bboxes(child))
    return bboxes


def _locate_optional_single_artifact(output_dir: Path, pattern: str) -> Path | None:
    matches = sorted(output_dir.rglob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise MinerUServiceError(
            f"MinerU output produced multiple artifacts for pattern: {pattern}",
            code="MINERU_AMBIGUOUS_OUTPUT",
            retryable=False,
            details={"output_dir": str(output_dir), "matches": [str(path) for path in matches]},
        )
    return matches[0]


def _collect_mineru_artifacts(*, output_dir: Path, middle_json_ref: ArtifactRef) -> tuple[ArtifactRef, ...]:
    artifacts: list[ArtifactRef] = [middle_json_ref]
    artifact_specs = (
        ("*_content_list.json", "content_list_json", "application/json"),
        ("*_content_list_v2.json", "content_list_v2_json", "application/json"),
        ("*_model.json", "model_json", "application/json"),
        ("*.md", "markdown", "text/markdown"),
        ("*_layout.pdf", "layout_pdf", "application/pdf"),
        ("*_span.pdf", "span_pdf", "application/pdf"),
        ("*_origin.pdf", "origin_pdf", "application/pdf"),
    )
    for pattern, kind, content_type in artifact_specs:
        path = _locate_optional_single_artifact(output_dir, pattern)
        if path is None or str(path) == middle_json_ref.uri:
            continue
        artifacts.append(
            ArtifactRef(
                kind=kind,
                uri=str(path),
                meta={"content_type": content_type},
            )
        )
    return tuple(artifacts)


def _merge_artifacts(*artifact_groups: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
    merged: list[ArtifactRef] = []
    seen: set[tuple[str, str | None]] = set()
    for artifacts in artifact_groups:
        for artifact in artifacts:
            key = (artifact.kind, artifact.uri)
            if key in seen:
                continue
            seen.add(key)
            merged.append(artifact)
    return tuple(merged)
