from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from typing import Any

from ..contracts import ArtifactRef, BoundingBox, CoordSpace, ImageSize, ParsedPage, ParsedPdf, TableBlock, TextBlock
from .mineru import MinerUDocumentParseResult, MinerUPageResult
from .paddle_table import (
    LOGGER as PADDLE_TABLE_LOGGER,
    PaddleTableStructureError,
    PaddleTableStructureService,
    _cached_paddle_object,
    _coerce_bool,
    _coerce_mapping,
    _coerce_optional_str,
    _extract_result_json,
    _jsonable,
    _prepare_paddle_runtime,
    _resolve_paddle_device,
    _unwrap_result_payload,
)

JsonDict = dict[str, Any]
_PADDLE_DOCUMENT_CACHE_LOCK = threading.Lock()
_PADDLE_DOCUMENT_CACHE: dict[str, Any] = {}
_LAST_PADDLE_DOCUMENT_RUNTIME: JsonDict = {}
LOGGER = logging.getLogger(__name__)


class PaddleDocumentVLError(PaddleTableStructureError):
    def __init__(self, message: str, *, code: str = "PADDLE_VL_REQUIRED_FAILED", detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.detail = dict(detail or {})


@dataclass(frozen=True)
class PaddleDocumentService:
    def parse_document(
        self,
        *,
        file_uri: str,
        mime_type: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MinerUDocumentParseResult:
        del mime_type
        input_path = Path(file_uri).expanduser().resolve()
        opts = dict(options or {})
        output_dir = Path(
            _coerce_optional_str(opts.get("output_dir")) or tempfile.mkdtemp(prefix="paddle_doc_")
        ).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        page_image_paths = _materialize_page_images(input_path, output_dir=output_dir / "paddle_pages")
        layout_pages = _predict_pages_with_layout(input_path, opts)
        ocr_pages = _predict_pages_with_ocr(input_path, opts)
        vl_pages = _predict_pages_with_vl(input_path, opts)

        layout_artifact = _write_artifact_json(
            output_dir=output_dir,
            filename="paddle_layout.json",
            payload={"provider": "paddle_ppstructurev3", "pages": layout_pages},
            kind="paddle_layout_json",
        )
        ocr_artifact = _write_artifact_json(
            output_dir=output_dir,
            filename="paddle_ocr.json",
            payload={"provider": "paddle_ocr_v5", "pages": ocr_pages},
            kind="paddle_ocr_json",
        )
        vl_artifact = _write_artifact_json(
            output_dir=output_dir,
            filename="paddle_vl.json",
            payload={
                "provider": "paddle_vl",
                "vl_version": _resolve_paddle_vl_version(opts),
                "pages": vl_pages,
            },
            kind="paddle_vl_json",
        )

        page_results = _build_pages_from_paddle_outputs(
            layout_pages=layout_pages,
            ocr_pages=ocr_pages,
            vl_pages=vl_pages,
            page_image_paths=page_image_paths,
        )

        table_candidates_by_page = {
            page.page_index: tuple(
                {
                    "page_index": table.page_index,
                    "bbox": table.bounding_box.model_dump(mode="python") if table.bounding_box is not None else None,
                    "image_uri": _crop_table_region(
                        page_image_paths[page.page_index],
                        table.bounding_box,
                        output_dir=output_dir / "table_crops" / f"page_{page.page_index + 1:04d}",
                        crop_index=index,
                    ),
                    "caption": None,
                    "footnote": None,
                }
                for index, table in enumerate(page.table_blocks)
                if table.bounding_box is not None and page.page_index in page_image_paths
            )
            for page in page_results
        }
        text_blocks_by_page = {
            page.page_index: tuple(page.text_blocks)
            for page in page_results
        }

        table_service = PaddleTableStructureService()
        table_result = table_service.extract_tables(
            file_uri=str(input_path),
            output_dir=output_dir,
            target_page_sizes={page.page_index: page.image_size for page in page_results},
            options=opts,
            table_candidates_by_page=table_candidates_by_page,
            page_text_blocks_by_page=text_blocks_by_page,
        )

        merged_pages = _merge_table_result_into_pages(page_results, table_result.tables_by_page)
        parsed_pdf = ParsedPdf(
            pdf_path=str(input_path),
            total_pages=len(merged_pages),
            pages=[
                ParsedPage(
                    page_index=page.page_index,
                    text_blocks=list(page.text_blocks),
                    table_blocks=list(page.table_blocks),
                    image_size=page.image_size,
                )
                for page in merged_pages
            ],
        )
        artifacts = [layout_artifact, ocr_artifact, vl_artifact]
        if table_result.artifact_ref is not None:
            artifacts.append(table_result.artifact_ref)

        return MinerUDocumentParseResult(
            pages=tuple(merged_pages),
            middle_json_ref=layout_artifact,
            page_count=len(merged_pages),
            coord_space=CoordSpace.IMAGE_PIXELS,
            parsed_pdf=parsed_pdf,
            artifacts=tuple(artifacts),
            meta={
                "layout_provider": "paddle",
                "ocr_provider": "paddle_ocr_v5",
                "table_provider": table_result.meta.get("provider", "paddleocr_table_structure"),
                "vl_provider": "paddle_vl",
                "paddle_vl_version": _resolve_paddle_vl_version(opts),
            },
        )


def warmup_paddle_document_models(
    *,
    options: Mapping[str, Any] | None = None,
    modes: Sequence[str] = ("layout", "ocr", "table", "vl"),
) -> JsonDict:
    opts = dict(options or {})
    warmed: list[str] = []
    for mode in modes:
        normalized = mode.strip().lower()
        if normalized in {"layout", "ppstructurev3"}:
            _build_layout_pipeline(opts)
            warmed.append("layout")
        elif normalized in {"ocr", "ocr_v5"}:
            _build_ocr_pipeline(opts)
            warmed.append("ocr")
        elif normalized in {"table", "table_structure"}:
            _build_table_pipeline(opts)
            warmed.append("table")
        elif normalized in {"vl", "vl1.6", "vl1.5"}:
            _build_vl_pipeline(opts)
            warmed.append("vl")
        else:
            raise PaddleTableStructureError(f"Unsupported Paddle document warmup mode: {mode}")
    return {"warmed": warmed, "cache": paddle_document_cache_info()}


def paddle_document_cache_info() -> JsonDict:
    with _PADDLE_DOCUMENT_CACHE_LOCK:
        return {
            "cache_size": len(_PADDLE_DOCUMENT_CACHE),
            "cache_keys": sorted(_PADDLE_DOCUMENT_CACHE.keys()),
            "runtime": dict(_LAST_PADDLE_DOCUMENT_RUNTIME),
        }


def _predict_pages_with_layout(input_path: Path, options: Mapping[str, Any]) -> list[JsonDict]:
    pipeline = _build_layout_pipeline(options)
    predict_kwargs = dict(_coerce_mapping(options.get("paddle_layout_predict_kwargs")) or {})
    return [
        _jsonable(_extract_result_json(result))
        for result in pipeline.predict(input=str(input_path), **predict_kwargs)
    ]


def _predict_pages_with_ocr(input_path: Path, options: Mapping[str, Any]) -> list[JsonDict]:
    pipeline = _build_ocr_pipeline(options)
    predict_kwargs = dict(_coerce_mapping(options.get("paddle_ocr_predict_kwargs")) or {})
    return [
        _jsonable(_extract_result_json(result))
        for result in pipeline.predict(input=str(input_path), **predict_kwargs)
    ]


def _predict_pages_with_vl(input_path: Path, options: Mapping[str, Any]) -> list[JsonDict]:
    if not _coerce_bool(options.get("enable_paddle_vl"), True):
        return []
    predict_kwargs = dict(_coerce_mapping(options.get("paddle_vl_predict_kwargs")) or {})
    try:
        pipeline = _build_vl_pipeline(options)
    except Exception as exc:
        raise _build_vl_required_error(
            input_path=input_path,
            options=options,
            reason="build_failed",
            exc=exc,
            predict_kwargs=predict_kwargs,
        ) from exc

    try:
        pages = [
            _jsonable(_extract_result_json(result))
            for result in pipeline.predict(input=str(input_path), **predict_kwargs)
        ]
    except Exception as exc:
        raise _build_vl_required_error(
            input_path=input_path,
            options=options,
            reason="predict_failed",
            exc=exc,
            predict_kwargs=predict_kwargs,
        ) from exc

    if not pages:
        raise _build_vl_required_error(
            input_path=input_path,
            options=options,
            reason="empty_pages",
            exc=None,
            predict_kwargs=predict_kwargs,
        )
    return pages


def _build_vl_required_error(
    *,
    input_path: Path,
    options: Mapping[str, Any],
    reason: str,
    exc: Exception | None,
    predict_kwargs: Mapping[str, Any],
) -> PaddleDocumentVLError:
    vl_version = _resolve_paddle_vl_version(options)
    pipeline_version = _resolve_paddle_vl_pipeline_version(options)
    resolved_device = _resolve_paddle_device(options)
    runtime = paddle_document_cache_info().get("runtime") or {}
    detail: JsonDict = {
        "stage": "vl",
        "reason": reason,
        "input_path": str(input_path),
        "paddle_vl_version": vl_version,
        "pipeline_version": pipeline_version,
        "resolved_device": resolved_device,
        "predict_kwargs": _jsonable(dict(predict_kwargs)),
        "init_kwargs": _jsonable(_coerce_mapping(options.get("paddle_vl_init_kwargs")) or {}),
        "runtime": _jsonable(runtime),
        "enable_paddle_vl": True,
    }
    if exc is not None:
        detail["exception_type"] = type(exc).__name__
        detail["exception_message"] = str(exc)
    message = (
        f"Paddle VL required but failed for {input_path} "
        f"(reason={reason}, paddle_vl_version={vl_version}, pipeline_version={pipeline_version}, "
        f"device={resolved_device or 'cpu-or-none'}, runtime={json.dumps(detail['runtime'], ensure_ascii=False, sort_keys=True)})"
    )
    LOGGER.error(message)
    return PaddleDocumentVLError(message, detail=detail)


def _build_layout_pipeline(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import PPStructureV3
    except Exception as exc:  # pragma: no cover
        raise PaddleTableStructureError("PaddleOCR PPStructureV3 is not installed") from exc

    init_kwargs: JsonDict = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "use_table_recognition": True,
        "use_formula_recognition": False,
        "use_seal_recognition": False,
        "use_chart_recognition": False,
    }
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_layout_init_kwargs")) or {})
    _record_runtime("layout", init_kwargs)
    return _cached_document_object(
        cache_prefix="paddle_layout",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_document_disable_pipeline_cache")),
        factory=lambda kwargs: PPStructureV3(**kwargs),
        fallback_keys={"use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation", "device"},
    )


def _build_ocr_pipeline(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover
        raise PaddleTableStructureError("PaddleOCR OCR pipeline is not installed") from exc

    init_kwargs: JsonDict = {}
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_ocr_init_kwargs")) or {})
    _record_runtime("ocr", init_kwargs)
    return _cached_document_object(
        cache_prefix="paddle_ocr",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_document_disable_pipeline_cache")),
        factory=lambda kwargs: PaddleOCR(**kwargs),
        fallback_keys={"device"},
    )


def _build_table_pipeline(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import TableStructureRecognition
    except Exception as exc:  # pragma: no cover
        raise PaddleTableStructureError("PaddleOCR TableStructureRecognition is not installed") from exc

    init_kwargs: JsonDict = {"model_name": "SLANet_plus"}
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_table_structure_init_kwargs")) or {})
    _record_runtime("table", init_kwargs)
    return _cached_document_object(
        cache_prefix="paddle_doc_table",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_document_disable_pipeline_cache")),
        factory=lambda kwargs: TableStructureRecognition(**kwargs),
    )


def _build_vl_pipeline(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import PaddleOCRVL
    except Exception as exc:  # pragma: no cover
        raise PaddleTableStructureError("PaddleOCRVL is not installed") from exc

    init_kwargs: JsonDict = {}
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.setdefault("pipeline_version", _resolve_paddle_vl_pipeline_version(options))
    init_kwargs.update(_coerce_mapping(options.get("paddle_vl_init_kwargs")) or {})
    _record_runtime("vl", init_kwargs)
    return _cached_document_object(
        cache_prefix="paddle_vl",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_document_disable_pipeline_cache")),
        factory=lambda kwargs: PaddleOCRVL(**kwargs),
        fallback_keys={"device", "pipeline_version"},
    )


def _cached_document_object(
    *,
    cache_prefix: str,
    init_kwargs: Mapping[str, Any],
    disable_cache: bool,
    factory: Any,
    fallback_keys: set[str] | None = None,
) -> Any:
    cache_key = f"{cache_prefix}:{json.dumps(dict(init_kwargs), sort_keys=True, default=str)}"
    if disable_cache:
        return _cached_paddle_object(
            cache_prefix=cache_prefix,
            init_kwargs=init_kwargs,
            disable_cache=True,
            factory=factory,
            fallback_keys=fallback_keys,
        )
    with _PADDLE_DOCUMENT_CACHE_LOCK:
        cached = _PADDLE_DOCUMENT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        cached = _cached_paddle_object(
            cache_prefix=cache_prefix,
            init_kwargs=init_kwargs,
            disable_cache=True,
            factory=factory,
            fallback_keys=fallback_keys,
        )
        _PADDLE_DOCUMENT_CACHE[cache_key] = cached
        return cached


def _build_pages_from_paddle_outputs(
    *,
    layout_pages: Sequence[Mapping[str, Any]],
    ocr_pages: Sequence[Mapping[str, Any]],
    vl_pages: Sequence[Mapping[str, Any]],
    page_image_paths: Mapping[int, Path],
) -> list[MinerUPageResult]:
    ocr_lines_by_page = _extract_ocr_lines_by_page(ocr_pages, page_image_paths=page_image_paths)
    vl_blocks_by_page = _extract_vl_blocks_by_page(vl_pages, page_image_paths=page_image_paths)

    pages: list[MinerUPageResult] = []
    for fallback_page_index, raw_layout_page in enumerate(layout_pages):
        page_payload = _unwrap_result_payload(raw_layout_page)
        page_index = _coerce_page_index(page_payload.get("page_index"), fallback_page_index)
        image_path = page_image_paths.get(page_index)
        image_size = _image_size_for_path(image_path)
        source_width = _coerce_number(page_payload.get("width"))
        source_height = _coerce_number(page_payload.get("height"))
        scale_x, scale_y = _page_scale(
            source_width=source_width,
            source_height=source_height,
            image_size=image_size,
        )

        consumed_ocr_indexes: set[int] = set()
        text_blocks: list[TextBlock] = []
        layout_table_blocks: list[TableBlock] = []
        for block_index, raw_block in enumerate(_as_mapping_list(page_payload.get("parsing_res_list"))):
            label = str(raw_block.get("block_label") or "text").strip().lower()
            bbox = _coerce_bbox_to_model(
                raw_block.get("block_bbox") or raw_block.get("bbox"),
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if bbox is None:
                continue
            if label == "table":
                layout_table_blocks.append(
                    TableBlock(
                        table_id=f"p{page_index}-layout-t{len(layout_table_blocks)}",
                        page_index=page_index,
                        provider="paddle_layout",
                        bounding_box=bbox,
                        coord_space=CoordSpace.IMAGE_PIXELS.value,
                        meta={
                            "source": "paddle_ppstructurev3",
                            "block_label": label,
                            "block_order": raw_block.get("block_order"),
                            "block_id": raw_block.get("block_id"),
                            "paddle_index": block_index,
                        },
                    )
                )
                continue

            block_text, matched_indexes = _block_text_from_ocr_lines(
                bbox,
                ocr_lines_by_page.get(page_index, []),
                consumed_indexes=consumed_ocr_indexes,
                fallback_text=_extract_layout_block_text(raw_block),
            )
            consumed_ocr_indexes.update(matched_indexes)
            if not block_text:
                continue
            text_blocks.append(
                TextBlock(
                    text=block_text,
                    bounding_box=bbox,
                    block_type=label,
                    confidence=_coerce_number(raw_block.get("score") or raw_block.get("confidence")),
                    meta={
                        "source": "paddle_ppstructurev3",
                        "block_label": label,
                        "block_order": raw_block.get("block_order"),
                        "block_id": raw_block.get("block_id"),
                        "paddle_index": block_index,
                        "coord_space": CoordSpace.IMAGE_PIXELS.value,
                    },
                )
            )

        for line_index, line in enumerate(ocr_lines_by_page.get(page_index, [])):
            if line_index in consumed_ocr_indexes:
                continue
            text_blocks.append(line)

        text_blocks = _dedupe_text_blocks(text_blocks)
        text_blocks = _merge_vl_blocks(text_blocks, vl_blocks_by_page.get(page_index, []))

        page_text = _compose_page_text(text_blocks, tuple(layout_table_blocks))
        pages.append(
            MinerUPageResult(
                page_index=page_index,
                text=page_text,
                text_blocks=tuple(sorted(text_blocks, key=lambda block: (block.bounding_box.y, block.bounding_box.x))),
                table_blocks=tuple(layout_table_blocks),
                image_size=image_size,
                page_meta={
                    "layout_provider": "paddle",
                    "coord_space": CoordSpace.IMAGE_PIXELS.value,
                    "width": image_size.width if image_size is not None else None,
                    "height": image_size.height if image_size is not None else None,
                    "paddle_page_source_width": source_width,
                    "paddle_page_source_height": source_height,
                },
            )
        )
    return pages


def _extract_ocr_lines_by_page(
    raw_pages: Sequence[Mapping[str, Any]],
    *,
    page_image_paths: Mapping[int, Path],
) -> dict[int, list[TextBlock]]:
    grouped: dict[int, list[TextBlock]] = {}
    for fallback_page_index, raw_page in enumerate(raw_pages):
        page_payload = _unwrap_result_payload(raw_page)
        page_index = _coerce_page_index(page_payload.get("page_index"), fallback_page_index)
        image_size = _image_size_for_path(page_image_paths.get(page_index))
        scale_x, scale_y = _page_scale(
            source_width=_coerce_number(page_payload.get("width")),
            source_height=_coerce_number(page_payload.get("height")),
            image_size=image_size,
        )
        overall = page_payload.get("overall_ocr_res") if isinstance(page_payload, Mapping) else None
        if not isinstance(overall, Mapping):
            overall = page_payload
        texts = overall.get("rec_texts")
        boxes = overall.get("rec_boxes")
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            continue
        if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
            continue
        scores = overall.get("rec_scores")
        grouped.setdefault(page_index, [])
        for index, raw_text in enumerate(texts):
            text = _normalize_text(raw_text)
            if not text:
                continue
            bbox = _coerce_bbox_to_model(boxes[index] if index < len(boxes) else None, scale_x=scale_x, scale_y=scale_y)
            if bbox is None:
                continue
            confidence = None
            if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)) and index < len(scores):
                confidence = _coerce_number(scores[index])
            grouped[page_index].append(
                TextBlock(
                    text=text,
                    bounding_box=bbox,
                    block_type="paddle_ocr_text",
                    confidence=confidence,
                    meta={
                        "source": "paddle_ocr_v5",
                        "paddle_index": index,
                        "coord_space": CoordSpace.IMAGE_PIXELS.value,
                    },
                )
            )
    return grouped


def _extract_vl_blocks_by_page(
    raw_pages: Sequence[Mapping[str, Any]],
    *,
    page_image_paths: Mapping[int, Path],
) -> dict[int, list[TextBlock]]:
    grouped: dict[int, list[TextBlock]] = {}
    for fallback_page_index, raw_page in enumerate(raw_pages):
        page_payload = _unwrap_result_payload(raw_page)
        page_index = _coerce_page_index(page_payload.get("page_index"), fallback_page_index)
        image_size = _image_size_for_path(page_image_paths.get(page_index))
        scale_x, scale_y = _page_scale(
            source_width=_coerce_number(page_payload.get("width")),
            source_height=_coerce_number(page_payload.get("height")),
            image_size=image_size,
        )
        grouped.setdefault(page_index, [])
        for block_index, block in enumerate(_as_mapping_list(page_payload.get("parsing_res_list"))):
            label = str(block.get("block_label") or "").strip().lower()
            if label not in {"seal", "stamp", "formula"}:
                continue
            bbox = _coerce_bbox_to_model(
                block.get("block_bbox") or block.get("bbox"),
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if bbox is None:
                continue
            text = _extract_layout_block_text(block)
            grouped[page_index].append(
                TextBlock(
                    text=text,
                    bounding_box=bbox,
                    block_type=label,
                    confidence=_coerce_number(block.get("score") or block.get("confidence")),
                    meta={
                        "source": "paddle_vl",
                        "block_label": label,
                        "paddle_index": block_index,
                        "coord_space": CoordSpace.IMAGE_PIXELS.value,
                    },
                )
            )
    return grouped


def _merge_table_result_into_pages(
    pages: Sequence[MinerUPageResult],
    tables_by_page: Mapping[int, Sequence[TableBlock]],
) -> list[MinerUPageResult]:
    merged_pages: list[MinerUPageResult] = []
    for page in pages:
        paddle_tables = tuple(tables_by_page.get(page.page_index, ()))
        if not paddle_tables:
            merged_pages.append(page)
            continue
        text_blocks = [
            block
            for block in page.text_blocks
            if not any(_bbox_overlap(block.bounding_box, table.bounding_box) >= 0.15 for table in paddle_tables if table.bounding_box is not None)
        ]
        page_text = _compose_page_text(text_blocks, paddle_tables)
        page_meta = dict(page.page_meta)
        page_meta["table_cell_refine"] = {
            "provider": "paddleocr_table_structure",
            "table_count": len(paddle_tables),
            "cell_count": sum(len(table.cells) for table in paddle_tables),
        }
        merged_pages.append(
            MinerUPageResult(
                page_index=page.page_index,
                text=page_text,
                text_blocks=tuple(text_blocks),
                table_blocks=tuple(paddle_tables),
                image_size=page.image_size,
                page_meta=page_meta,
            )
        )
    return merged_pages


def _compose_page_text(text_blocks: Sequence[TextBlock], table_blocks: Sequence[TableBlock]) -> str | None:
    items: list[tuple[float, float, str]] = []
    seen: set[str] = set()
    for block in sorted(text_blocks, key=lambda item: (item.bounding_box.y, item.bounding_box.x)):
        text = _normalize_text(block.text)
        if not text:
            continue
        normalized = _normalize_key(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append((float(block.bounding_box.y), float(block.bounding_box.x), text))
    for table in table_blocks:
        table_text = _table_to_text(table)
        if not table_text:
            continue
        normalized = _normalize_key(table_text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        box = table.bounding_box or BoundingBox(x=0, y=0, w=0, h=0)
        items.append((float(box.y), float(box.x), table_text))
    if not items:
        return None
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return "\n".join(text for _, _, text in items)


def _table_to_text(table: TableBlock) -> str | None:
    if table.cells:
        rows: dict[int, list[tuple[int, str]]] = {}
        for cell in table.cells:
            text = _normalize_text(cell.text)
            if not text:
                continue
            row = cell.row_index if cell.row_index is not None else 0
            col = cell.col_index if cell.col_index is not None else 0
            rows.setdefault(row, []).append((col, text))
        if rows:
            return "\n".join(" | ".join(text for _, text in sorted(items)) for _, items in sorted(rows.items()))
    html = _coerce_optional_str(table.html)
    return _normalize_text(html)


def _merge_vl_blocks(text_blocks: list[TextBlock], vl_blocks: Sequence[TextBlock]) -> list[TextBlock]:
    merged = list(text_blocks)
    for vl_block in vl_blocks:
        updated = False
        for index, block in enumerate(merged):
            if _bbox_overlap(block.bounding_box, vl_block.bounding_box) < 0.12:
                continue
            meta = dict(block.meta)
            meta["vl_block_type"] = vl_block.block_type
            meta["vl_source"] = "paddle_vl"
            merged[index] = block.model_copy(update={"meta": meta})
            updated = True
            break
        if not updated:
            merged.append(vl_block)
    return merged


def _dedupe_text_blocks(blocks: Sequence[TextBlock]) -> list[TextBlock]:
    deduped: list[TextBlock] = []
    for block in sorted(blocks, key=lambda item: (item.bounding_box.y, item.bounding_box.x, item.text)):
        normalized = _normalize_key(block.text)
        if not normalized:
            continue
        if any(
            normalized == _normalize_key(existing.text)
            and _bbox_overlap(existing.bounding_box, block.bounding_box) >= 0.5
            for existing in deduped
        ):
            continue
        deduped.append(block)
    return deduped


def _block_text_from_ocr_lines(
    bbox: BoundingBox,
    lines: Sequence[TextBlock],
    *,
    consumed_indexes: set[int],
    fallback_text: str | None,
) -> tuple[str | None, set[int]]:
    matched_indexes: set[int] = set()
    matched_lines: list[TextBlock] = []
    for index, line in enumerate(lines):
        if index in consumed_indexes:
            continue
        if _bbox_center_inside(line.bounding_box, bbox):
            matched_indexes.add(index)
            matched_lines.append(line)
    if matched_lines:
        matched_lines = sorted(matched_lines, key=lambda item: (item.bounding_box.y, item.bounding_box.x))
        text = "\n".join(_normalize_text(item.text) for item in matched_lines if _normalize_text(item.text))
        if text:
            return text, matched_indexes
    return (_normalize_text(fallback_text) or None), matched_indexes


def _extract_layout_block_text(block: Mapping[str, Any]) -> str:
    raw_text = (
        block.get("block_content")
        or block.get("rec_text")
        or block.get("text")
        or block.get("html")
    )
    if not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()
    if not text:
        return ""
    if "<" in text and ">" in text:
        text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_text(text)


def _materialize_page_images(input_path: Path, *, output_dir: Path) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return {0: input_path}
    pdftoppm = shutil_which("pdftoppm")
    if pdftoppm is None:
        raise PaddleTableStructureError("pdftoppm is required for full Paddle PDF rendering")
    prefix = output_dir / "page"
    completed = subprocess.run(
        [pdftoppm, "-png", "-r", "144", str(input_path), str(prefix)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PaddleTableStructureError(
            f"pdftoppm failed while rendering PDF pages: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    page_paths: dict[int, Path] = {}
    for index, path in enumerate(sorted(output_dir.glob("page-*.png"))):
        page_paths[index] = path
    if not page_paths:
        raise PaddleTableStructureError(f"pdftoppm produced no page images for {input_path}")
    return page_paths


def _crop_table_region(
    page_image_path: Path,
    bbox: BoundingBox | None,
    *,
    output_dir: Path,
    crop_index: int,
) -> str | None:
    if bbox is None:
        return None
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PaddleTableStructureError("Pillow is required for Paddle table crops") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"table_{crop_index:04d}.png"
    with Image.open(page_image_path) as image:
        left = max(0, int(bbox.x))
        top = max(0, int(bbox.y))
        right = min(image.width, int(bbox.x + bbox.w))
        bottom = min(image.height, int(bbox.y + bbox.h))
        if right <= left or bottom <= top:
            return None
        image.crop((left, top, right, bottom)).save(crop_path)
    return str(crop_path)


def _write_artifact_json(
    *,
    output_dir: Path,
    filename: str,
    payload: Mapping[str, Any],
    kind: str,
) -> ArtifactRef:
    path = output_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ArtifactRef(kind=kind, uri=str(path), meta={"content_type": "application/json"})


def _record_runtime(component: str, init_kwargs: Mapping[str, Any]) -> None:
    payload = {
        "component": component,
        "init_kwargs": _jsonable(dict(init_kwargs)),
        "python_executable": os.sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with _PADDLE_DOCUMENT_CACHE_LOCK:
        _LAST_PADDLE_DOCUMENT_RUNTIME[component] = payload
    LOGGER.info("Prepared Paddle document component=%s init_kwargs=%s", component, dict(init_kwargs))
    PADDLE_TABLE_LOGGER.info("Prepared Paddle document component=%s init_kwargs=%s", component, dict(init_kwargs))


def _resolve_paddle_vl_version(options: Mapping[str, Any]) -> str:
    explicit = _coerce_optional_str(options.get("paddle_vl_version"))
    if explicit in {"1.5", "1.6"}:
        return explicit
    env_value = _coerce_optional_str(os.environ.get("PADDLE_VL_VERSION"))
    if env_value in {"1.5", "1.6"}:
        return env_value
    return "1.6"


def _resolve_paddle_vl_pipeline_version(options: Mapping[str, Any]) -> str:
    version = _resolve_paddle_vl_version(options)
    if version == "1.5":
        return "v1.5"
    return "v1"


def _page_scale(
    *,
    source_width: float | None,
    source_height: float | None,
    image_size: ImageSize | None,
) -> tuple[float, float]:
    if image_size is None or not source_width or not source_height:
        return 1.0, 1.0
    return float(image_size.width) / float(source_width), float(image_size.height) / float(source_height)


def _coerce_bbox_to_model(value: Any, *, scale_x: float, scale_y: float) -> BoundingBox | None:
    if isinstance(value, Mapping):
        if {"x", "y", "w", "h"}.issubset(value):
            x = _coerce_number(value.get("x"))
            y = _coerce_number(value.get("y"))
            w = _coerce_number(value.get("w"))
            h = _coerce_number(value.get("h"))
            if x is None or y is None or w is None or h is None:
                return None
            return BoundingBox(
                x=int(round(x * scale_x)),
                y=int(round(y * scale_y)),
                w=max(0, int(round(w * scale_x))),
                h=max(0, int(round(h * scale_y))),
            )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        nums = [_coerce_number(item) for item in value[:4]]
        if any(item is None for item in nums):
            return None
        x0, y0, x1, y1 = [float(item) for item in nums if item is not None]
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BoundingBox(
            x=int(round(x0 * scale_x)),
            y=int(round(y0 * scale_y)),
            w=max(0, int(round((x1 - x0) * scale_x))),
            h=max(0, int(round((y1 - y0) * scale_y))),
        )
    return None


def _coerce_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_page_index(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[\W_]+", "", _normalize_text(value)).casefold()


def _image_size_for_path(path: Path | None) -> ImageSize | None:
    if path is None:
        return None
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return None
    with Image.open(path) as image:
        return ImageSize(width=int(image.width), height=int(image.height))


def _bbox_center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx = inner.x + inner.w / 2.0
    cy = inner.y + inner.h / 2.0
    return outer.x <= cx <= outer.x + outer.w and outer.y <= cy <= outer.y + outer.h


def _bbox_overlap(left: BoundingBox, right: BoundingBox | None) -> float:
    if right is None:
        return 0.0
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.w, right.x + right.w)
    y1 = min(left.y + left.h, right.y + right.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = float((x1 - x0) * (y1 - y0))
    left_area = float(max(1, left.w * left.h))
    right_area = float(max(1, right.w * right.h))
    return intersection / min(left_area, right_area)


def _as_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def shutil_which(command: str) -> str | None:
    from shutil import which

    return which(command)
