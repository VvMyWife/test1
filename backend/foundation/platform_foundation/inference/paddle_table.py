from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
import base64
import json
import os
from pathlib import Path
import re
from statistics import mean, median
import threading
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from ..contracts import ArtifactRef, BoundingBox, ImageSize, TableBlock, TableCell, TextBlock

JsonDict = dict[str, Any]
_PIPELINE_CACHE: dict[str, Any] = {}
_PIPELINE_CACHE_LOCK = threading.Lock()


class PaddleTableStructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaddleTableStructureResult:
    tables_by_page: Mapping[int, tuple[TableBlock, ...]]
    artifact_ref: ArtifactRef | None = None
    meta: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PaddleTableStructureService:
    """Optional PaddleOCR table-cell refinement.

    The preferred batch path uses MinerU table crops plus PaddleOCR's table
    structure module. It keeps the Paddle model alive in-process and avoids
    running a full PP-StructureV3 layout pass over every page.
    """

    def extract_tables(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any] | None = None,
        table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
        page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]] | None = None,
    ) -> PaddleTableStructureResult:
        opts = dict(options or {})
        candidates = dict(table_candidates_by_page or {})
        mode = _coerce_optional_str(opts.get("paddle_table_mode"))
        if mode is None:
            mode = "table_structure" if candidates else "ppstructurev3"

        if mode in {"table_structure", "table_structure_recognition", "mineru_crops"}:
            return self._extract_tables_from_mineru_crops(
                output_dir=output_dir,
                options=opts,
                table_candidates_by_page=candidates,
                page_text_blocks_by_page=dict(page_text_blocks_by_page or {}),
            )

        return self._extract_tables_with_ppstructurev3(
            file_uri=file_uri,
            output_dir=output_dir,
            target_page_sizes=target_page_sizes,
            options=opts,
        )

    def _extract_tables_with_ppstructurev3(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any],
    ) -> PaddleTableStructureResult:
        predict_kwargs = dict(_coerce_mapping(options.get("paddle_table_predict_kwargs")) or {})

        raw_pages: list[JsonDict] = []
        try:
            pipeline = _build_ppstructure_v3(options)
            result_iter = pipeline.predict(input=file_uri, **predict_kwargs)
            for result in result_iter:
                raw_pages.append(_jsonable(_extract_result_json(result)))
        except Exception as exc:  # pragma: no cover - depends on Paddle runtime
            raise PaddleTableStructureError(f"PaddleOCR table recognition failed: {exc}") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "paddle_table_structure.json"
        artifact_payload = {
            "provider": "paddleocr_ppstructurev3",
            "mode": "ppstructurev3",
            "source_file": file_uri,
            "pages": raw_pages,
        }
        artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        artifact_ref = ArtifactRef(
            kind="paddle_table_json",
            uri=str(artifact_path),
            meta={
                "content_type": "application/json",
                "provider": "paddleocr_ppstructurev3",
                "mode": "ppstructurev3",
                "page_result_count": len(raw_pages),
            },
        )
        return parse_paddle_structure_tables(
            raw_pages,
            target_page_sizes=target_page_sizes,
            artifact_ref=artifact_ref,
        )

    def _extract_tables_from_mineru_crops(
        self,
        *,
        output_dir: Path,
        options: Mapping[str, Any],
        table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
        page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
    ) -> PaddleTableStructureResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_results: list[JsonDict] = []
        tables_by_page: dict[int, list[TableBlock]] = {}
        table_count = 0
        cell_count = 0

        candidates = [
            (page_index, candidate)
            for page_index, page_candidates in table_candidates_by_page.items()
            for candidate in page_candidates
            if _coerce_optional_str(candidate.get("image_uri"))
        ]
        if not candidates:
            return _write_table_artifact(
                output_dir=output_dir,
                provider="paddleocr_table_structure",
                mode="mineru_crops",
                raw_results=[],
                tables_by_page={},
                table_count=0,
                cell_count=0,
            )

        try:
            model = _build_table_structure_model(options)
            for table_index, (page_index, candidate) in enumerate(candidates):
                image_path = _coerce_optional_str(candidate.get("image_uri"))
                if image_path is None:
                    continue
                result_iter = model.predict(input=image_path, **dict(_coerce_mapping(options.get("paddle_table_predict_kwargs")) or {}))
                for result in result_iter:
                    raw_payload = _jsonable(_extract_result_json(result))
                    raw_results.append({"candidate": dict(candidate), "result": raw_payload})
                    table = _parse_table_structure_module_result(
                        raw_payload,
                        page_index=page_index,
                        table_index=table_index,
                        candidate=candidate,
                        page_text_blocks=page_text_blocks_by_page.get(page_index, ()),
                    )
                    if table is None:
                        continue
                    tables_by_page.setdefault(page_index, []).append(table)
                    table_count += 1
                    cell_count += len(table.cells)
        except Exception as exc:  # pragma: no cover - depends on Paddle runtime
            raise PaddleTableStructureError(f"PaddleOCR table structure recognition failed: {exc}") from exc

        return _write_table_artifact(
            output_dir=output_dir,
            provider="paddleocr_table_structure",
            mode="mineru_crops",
            raw_results=raw_results,
            tables_by_page=tables_by_page,
            table_count=table_count,
            cell_count=cell_count,
        )


@dataclass(frozen=True)
class PaddleTableApiClient:
    """HTTP client for a resident Paddle table service."""

    api_url: str
    timeout_seconds: float = 1800.0

    def extract_tables(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any] | None = None,
        table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
        page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]] | None = None,
    ) -> PaddleTableStructureResult:
        opts = dict(options or {})
        timeout_seconds = _coerce_float(opts.get("paddle_table_api_timeout_seconds"))
        if timeout_seconds is None:
            timeout_seconds = self.timeout_seconds
        send_file_bytes = _should_send_file_bytes(self.api_url, opts)
        payload = {
            "file_uri": file_uri,
            "file_name": _path_name_from_uri(file_uri),
            "output_dir": str(output_dir),
            "target_page_sizes": {
                str(page): size.model_dump(mode="json") if size is not None else None
                for page, size in target_page_sizes.items()
            },
            "options": _jsonable(opts),
            "table_candidates_by_page": {
                str(page): [
                    _jsonable_candidate_for_api(candidate, send_file_bytes=send_file_bytes)
                    for candidate in candidates
                ]
                for page, candidates in dict(table_candidates_by_page or {}).items()
            },
            "page_text_blocks_by_page": {
                str(page): [
                    block.model_dump(mode="json") if hasattr(block, "model_dump") else _jsonable(block)
                    for block in blocks
                ]
                for page, blocks in dict(page_text_blocks_by_page or {}).items()
            },
        }
        if send_file_bytes:
            payload["file_bytes_b64"] = _read_local_file_b64(file_uri)
        endpoint = f"{self.api_url.rstrip('/')}/api/v1/paddle/table-extract"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise PaddleTableStructureError(
                f"Paddle table API failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except Exception as exc:  # pragma: no cover - depends on resident API runtime
            raise PaddleTableStructureError(f"Paddle table API request failed: {exc}") from exc

        if not response_payload.get("success"):
            raise PaddleTableStructureError(
                str(response_payload.get("error") or "Paddle table API returned success=false")
            )
        response_data = response_payload["data"]
        result = paddle_table_result_from_payload(response_data)
        if send_file_bytes:
            result = _materialize_api_artifact(response_data, output_dir=output_dir, result=result)
        meta = dict(result.meta)
        meta["transport"] = "http"
        meta["api_url"] = self.api_url
        return PaddleTableStructureResult(
            tables_by_page=result.tables_by_page,
            artifact_ref=result.artifact_ref,
            meta=meta,
        )


def paddle_table_result_to_payload(result: PaddleTableStructureResult) -> JsonDict:
    artifact_payload: Any = None
    if result.artifact_ref is not None:
        artifact_uri = _coerce_optional_str(result.artifact_ref.uri)
        if artifact_uri:
            artifact_path = Path(artifact_uri)
            if artifact_path.is_file():
                try:
                    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                except Exception:
                    artifact_payload = None
    return {
        "tables_by_page": {
            str(page): [table.model_dump(mode="json") for table in tables]
            for page, tables in result.tables_by_page.items()
        },
        "artifact_ref": (
            result.artifact_ref.model_dump(mode="json") if result.artifact_ref is not None else None
        ),
        "artifact_payload": _jsonable(artifact_payload),
        "meta": _jsonable(result.meta),
    }


def paddle_table_result_from_payload(payload: Mapping[str, Any]) -> PaddleTableStructureResult:
    raw_tables = payload.get("tables_by_page") or {}
    tables_by_page: dict[int, tuple[TableBlock, ...]] = {}
    if isinstance(raw_tables, Mapping):
        for raw_page, raw_items in raw_tables.items():
            try:
                page_index = int(raw_page)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
                continue
            tables_by_page[page_index] = tuple(
                TableBlock.model_validate(item) for item in raw_items if isinstance(item, Mapping)
            )
    raw_artifact = payload.get("artifact_ref")
    artifact_ref = ArtifactRef.model_validate(raw_artifact) if isinstance(raw_artifact, Mapping) else None
    meta = dict(payload.get("meta") or {}) if isinstance(payload.get("meta"), Mapping) else {}
    return PaddleTableStructureResult(
        tables_by_page=tables_by_page,
        artifact_ref=artifact_ref,
        meta=meta,
    )


def _materialize_api_artifact(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
    result: PaddleTableStructureResult,
) -> PaddleTableStructureResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "paddle_table_structure.json"
    artifact_payload = payload.get("artifact_payload")
    if not isinstance(artifact_payload, Mapping):
        artifact_payload = {
            "provider": result.meta.get("provider"),
            "mode": result.meta.get("mode"),
            "table_count": result.meta.get("table_count"),
            "cell_count": result.meta.get("cell_count"),
            "tables_by_page": {
                str(page): [table.model_dump(mode="json") for table in tables]
                for page, tables in result.tables_by_page.items()
            },
        }
    artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_meta: JsonDict = {
        "content_type": "application/json",
        "provider": result.meta.get("provider"),
        "mode": result.meta.get("mode"),
    }
    if result.artifact_ref is not None:
        artifact_meta.update(result.artifact_ref.meta)
    artifact_ref = ArtifactRef(kind="paddle_table_json", uri=str(artifact_path), meta=artifact_meta)
    return PaddleTableStructureResult(
        tables_by_page=result.tables_by_page,
        artifact_ref=artifact_ref,
        meta=dict(result.meta),
    )


def warmup_paddle_table_models(
    *,
    options: Mapping[str, Any] | None = None,
    modes: Sequence[str] = ("table_structure", "ppstructurev3"),
) -> JsonDict:
    opts = dict(options or {})
    warmed: list[str] = []
    for mode in modes:
        normalized = mode.strip().lower()
        if normalized in {"table_structure", "table_structure_recognition", "mineru_crops"}:
            _build_table_structure_model(opts)
            warmed.append("table_structure")
        elif normalized in {"ppstructurev3", "pp_structure_v3"}:
            _build_ppstructure_v3(opts)
            warmed.append("ppstructurev3")
        else:
            raise PaddleTableStructureError(f"Unsupported warmup mode: {mode}")
    return {"warmed": warmed, "cache": paddle_table_cache_info()}


def paddle_table_cache_info() -> JsonDict:
    with _PIPELINE_CACHE_LOCK:
        return {
            "cache_size": len(_PIPELINE_CACHE),
            "cache_keys": sorted(_PIPELINE_CACHE.keys()),
        }


def parse_paddle_structure_tables(
    page_payloads: Sequence[Mapping[str, Any]],
    *,
    target_page_sizes: Mapping[int, ImageSize | None] | None = None,
    artifact_ref: ArtifactRef | None = None,
) -> PaddleTableStructureResult:
    target_sizes = dict(target_page_sizes or {})
    tables_by_page: dict[int, list[TableBlock]] = {}
    table_count = 0
    cell_count = 0

    for fallback_page_index, raw_payload in enumerate(page_payloads):
        payload = _unwrap_result_payload(raw_payload)
        page_index = _coerce_int(payload.get("page_index"), fallback_page_index)
        source_width = _coerce_float(payload.get("width"))
        source_height = _coerce_float(payload.get("height"))
        scale_x, scale_y, coord_space = _resolve_scale(
            source_width=source_width,
            source_height=source_height,
            target_size=target_sizes.get(page_index),
        )

        for table_index, raw_table in enumerate(_as_mappings(payload.get("table_res_list"))):
            table = _parse_table_block(
                raw_table,
                page_index=page_index,
                table_index=table_index,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=0.0,
                offset_y=0.0,
                coord_space=coord_space,
                source_width=source_width,
                source_height=source_height,
            )
            if table is None:
                continue
            tables_by_page.setdefault(page_index, []).append(table)
            table_count += 1
            cell_count += len(table.cells)

    return PaddleTableStructureResult(
        tables_by_page={page: tuple(tables) for page, tables in tables_by_page.items()},
        artifact_ref=artifact_ref,
        meta={
            "provider": "paddleocr_ppstructurev3",
            "mode": "ppstructurev3",
            "table_count": table_count,
            "cell_count": cell_count,
        },
    )


def _write_table_artifact(
    *,
    output_dir: Path,
    provider: str,
    mode: str,
    raw_results: Sequence[Mapping[str, Any]],
    tables_by_page: Mapping[int, Sequence[TableBlock]],
    table_count: int,
    cell_count: int,
) -> PaddleTableStructureResult:
    artifact_path = output_dir / "paddle_table_structure.json"
    artifact_payload = {
        "provider": provider,
        "mode": mode,
        "table_count": table_count,
        "cell_count": cell_count,
        "raw_results": list(raw_results),
    }
    artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_ref = ArtifactRef(
        kind="paddle_table_json",
        uri=str(artifact_path),
        meta={
            "content_type": "application/json",
            "provider": provider,
            "mode": mode,
            "table_count": table_count,
            "cell_count": cell_count,
        },
    )
    return PaddleTableStructureResult(
        tables_by_page={page: tuple(tables) for page, tables in tables_by_page.items()},
        artifact_ref=artifact_ref,
        meta={
            "provider": provider,
            "mode": mode,
            "table_count": table_count,
            "cell_count": cell_count,
        },
    )


def _build_table_structure_model(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import TableStructureRecognition
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PaddleTableStructureError("PaddleOCR TableStructureRecognition is not installed") from exc

    init_kwargs: JsonDict = {"model_name": "SLANet_plus"}
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_table_structure_init_kwargs")) or {})

    return _cached_paddle_object(
        cache_prefix="table_structure",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_table_disable_pipeline_cache")),
        factory=lambda kwargs: TableStructureRecognition(**kwargs),
    )


def _build_ppstructure_v3(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import PPStructureV3
    except Exception as exc:  # pragma: no cover - optional dependency
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
    init_kwargs.update(_coerce_mapping(options.get("paddle_table_init_kwargs")) or {})

    return _cached_paddle_object(
        cache_prefix="ppstructurev3",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_table_disable_pipeline_cache")),
        factory=lambda kwargs: PPStructureV3(**kwargs),
        fallback_keys={"use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation", "device"},
    )


def _cached_paddle_object(
    *,
    cache_prefix: str,
    init_kwargs: Mapping[str, Any],
    disable_cache: bool,
    factory: Any,
    fallback_keys: set[str] | None = None,
) -> Any:
    cache_key = f"{cache_prefix}:{json.dumps(dict(init_kwargs), sort_keys=True, default=str)}"
    if disable_cache:
        return _build_paddle_object(
            init_kwargs=init_kwargs,
            factory=factory,
            fallback_keys=fallback_keys,
        )

    with _PIPELINE_CACHE_LOCK:
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        model, resolved_cache_key = _build_paddle_object(
            init_kwargs=init_kwargs,
            factory=factory,
            fallback_keys=fallback_keys,
            cache_prefix=cache_prefix,
        )
        cached = _PIPELINE_CACHE.get(resolved_cache_key)
        if cached is not None:
            return cached
        cache_key = resolved_cache_key
        _PIPELINE_CACHE[cache_key] = model
    return model


def _build_paddle_object(
    *,
    init_kwargs: Mapping[str, Any],
    factory: Any,
    fallback_keys: set[str] | None = None,
    cache_prefix: str | None = None,
) -> Any:
    try:
        model = factory(dict(init_kwargs))
        if cache_prefix is None:
            return model
        return model, f"{cache_prefix}:{json.dumps(dict(init_kwargs), sort_keys=True, default=str)}"
    except TypeError:
        if not fallback_keys:
            raise
        fallback_kwargs = {key: value for key, value in init_kwargs.items() if key in fallback_keys}
        model = factory(fallback_kwargs)
        if cache_prefix is None:
            return model
        return model, f"{cache_prefix}:{json.dumps(fallback_kwargs, sort_keys=True, default=str)}"


def _prepare_paddle_runtime() -> None:
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _resolve_paddle_device(options: Mapping[str, Any]) -> str | None:
    configured = (
        _coerce_optional_str(options.get("paddle_device"))
        or _coerce_optional_str(os.environ.get("MINERU_PADDLE_DEVICE"))
    )
    if configured:
        return configured

    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        return None
    return None


def _parse_table_structure_module_result(
    raw_payload: Mapping[str, Any],
    *,
    page_index: int,
    table_index: int,
    candidate: Mapping[str, Any],
    page_text_blocks: Sequence[TextBlock],
) -> TableBlock | None:
    payload = _unwrap_result_payload(raw_payload)
    table_bbox = _coerce_bbox(candidate.get("bbox"), scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0)
    if table_bbox is None:
        return None

    image_width, image_height = _image_size(_coerce_optional_str(candidate.get("image_uri")))
    if image_width and image_height:
        scale_x = table_bbox.w / image_width
        scale_y = table_bbox.h / image_height
    else:
        scale_x = scale_y = 1.0

    raw_boxes = _as_list(payload.get("bbox")) or _as_list(payload.get("cell_box_list"))
    candidates: list[JsonDict] = []
    for cell_index, raw_box in enumerate(raw_boxes):
        bbox = _coerce_bbox(
            raw_box,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=float(table_bbox.x),
            offset_y=float(table_bbox.y),
        )
        if bbox is None:
            continue
        candidates.append(
            {
                "cell_id": f"p{page_index}-t{table_index}-c{cell_index}",
                "text": "",
                "bounding_box": bbox,
                "confidence": _coerce_confidence(payload.get("structure_score")),
                "meta": {
                    "source": "paddleocr_table_structure",
                    "paddle_index": cell_index,
                    "bbox_source": "table_structure.bbox",
                },
            }
        )

    html = _coerce_optional_str(payload.get("pred_html")) or _structure_to_html(payload.get("structure"))
    _assign_grid_indices(candidates)
    _merge_wrapped_cell_fragments_from_html(candidates, html)
    _fill_cell_text_from_page_blocks(candidates, page_text_blocks)
    cells = [TableCell(**candidate) for candidate in candidates]
    normalized_html = None if cells else html
    if not cells and not html:
        return None

    return TableBlock(
        table_id=f"p{page_index}-t{table_index}",
        page_index=page_index,
        provider="paddleocr_table_structure",
        bounding_box=table_bbox,
        coord_space="mineru_layout",
        html=normalized_html,
        cells=cells,
        confidence=_coerce_confidence(payload.get("structure_score")),
        meta={
            "source": "paddleocr_table_structure",
            "image_uri": candidate.get("image_uri"),
            "caption": candidate.get("caption"),
            "raw_cell_count": len(raw_boxes),
            "text_fill_source": "mineru_text_blocks",
        },
    )


def _parse_table_block(
    raw_table: Mapping[str, Any],
    *,
    page_index: int,
    table_index: int,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    coord_space: str,
    source_width: float | None,
    source_height: float | None,
) -> TableBlock | None:
    ocr_pred = _coerce_mapping(raw_table.get("table_ocr_pred")) or {}
    texts = [str(value).strip() for value in _as_list(ocr_pred.get("rec_texts"))]
    scores = [_coerce_confidence(value) for value in _as_list(ocr_pred.get("rec_scores"))]
    rec_boxes = _as_list(ocr_pred.get("rec_boxes"))
    cell_boxes = _as_list(raw_table.get("cell_box_list"))

    if cell_boxes and len(cell_boxes) == len(texts):
        raw_boxes = cell_boxes
        bbox_source = "cell_box_list"
    elif rec_boxes:
        raw_boxes = rec_boxes
        bbox_source = "table_ocr_pred.rec_boxes"
    else:
        raw_boxes = cell_boxes
        bbox_source = "cell_box_list"

    candidates: list[JsonDict] = []
    item_count = max(len(raw_boxes), len(texts))
    for cell_index in range(item_count):
        bbox = (
            _coerce_bbox(
                raw_boxes[cell_index],
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            if cell_index < len(raw_boxes)
            else None
        )
        text = texts[cell_index] if cell_index < len(texts) else ""
        if bbox is None and not text:
            continue
        candidates.append(
            {
                "cell_id": f"p{page_index}-t{table_index}-c{cell_index}",
                "text": text,
                "bounding_box": bbox,
                "confidence": scores[cell_index] if cell_index < len(scores) else None,
                "meta": {
                    "source": "paddleocr_table",
                    "paddle_index": cell_index,
                    "bbox_source": bbox_source,
                },
            }
        )

    html = _coerce_optional_str(raw_table.get("pred_html"))
    _assign_grid_indices(candidates)
    _merge_wrapped_cell_fragments_from_html(candidates, html)
    cells = [TableCell(**candidate) for candidate in candidates]
    table_bbox = _union_bounding_boxes([cell.bounding_box for cell in cells])
    normalized_html = None if cells else html
    if not cells and not html:
        return None

    return TableBlock(
        table_id=f"p{page_index}-t{table_index}",
        page_index=page_index,
        provider="paddleocr_ppstructurev3",
        bounding_box=table_bbox,
        coord_space=coord_space,
        html=normalized_html,
        cells=cells,
        meta={
            "source": "paddleocr_ppstructurev3",
            "source_width": source_width,
            "source_height": source_height,
            "raw_cell_count": len(cell_boxes),
            "raw_text_count": len(texts),
        },
    )


def _fill_cell_text_from_page_blocks(candidates: list[JsonDict], text_blocks: Sequence[TextBlock]) -> None:
    used_block_indexes: set[int] = set()
    sorted_blocks = sorted(
        [
            (index, block)
            for index, block in enumerate(text_blocks)
            if block.bounding_box is not None and block.block_type != "table_cell" and block.text.strip()
        ],
        key=lambda item: (item[1].bounding_box.y, item[1].bounding_box.x),
    )
    for candidate in candidates:
        cell_box = candidate.get("bounding_box")
        if cell_box is None:
            continue
        hits: list[TextBlock] = []
        for block_index, block in sorted_blocks:
            if block_index in used_block_indexes:
                continue
            if _bbox_center_inside(block.bounding_box, cell_box):
                hits.append(block)
                used_block_indexes.add(block_index)
        if not hits:
            continue
        candidate["text"] = "\n".join(block.text for block in hits)
        confidences = [block.confidence for block in hits if block.confidence is not None]
        if confidences:
            candidate["confidence"] = float(mean(confidences))
        meta = dict(candidate.get("meta") or {})
        meta["text_block_count"] = len(hits)
        candidate["meta"] = meta


def _bbox_center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    cx = inner.x + inner.w / 2.0
    cy = inner.y + inner.h / 2.0
    return outer.x <= cx <= outer.x + outer.w and outer.y <= cy <= outer.y + outer.h


def _assign_grid_indices(candidates: list[JsonDict]) -> None:
    boxes = [item.get("bounding_box") for item in candidates if item.get("bounding_box") is not None]
    if not boxes:
        return

    heights = [float(box.h) for box in boxes if box.h > 0]
    widths = [float(box.w) for box in boxes if box.w > 0]
    row_centers = _cluster_centers(
        [box.y + box.h / 2.0 for box in boxes],
        tolerance=max(8.0, (median(heights) if heights else 16.0) * 0.60),
    )
    col_centers = _cluster_centers(
        [box.x + box.w / 2.0 for box in boxes],
        tolerance=max(8.0, (median(widths) if widths else 16.0) * 0.60),
    )

    for item in candidates:
        box = item.get("bounding_box")
        if box is None:
            continue
        item["row_index"] = _nearest_index(row_centers, box.y + box.h / 2.0)
        item["col_index"] = _nearest_index(col_centers, box.x + box.w / 2.0)


def _merge_wrapped_cell_fragments_from_html(candidates: list[JsonDict], html: str | None) -> None:
    if not html or len(candidates) < 2:
        _normalize_candidate_texts(candidates)
        return

    _normalize_candidate_texts(candidates)
    existing_texts = {_normalize_table_cell_text(str(candidate.get("text") or "")) for candidate in candidates}
    targets = [
        text
        for text in _html_cell_texts(html)
        if len(text) >= 4 and text not in existing_texts
    ]
    if not targets:
        return

    used_indexes: set[int] = set()
    remove_indexes: set[int] = set()
    for target in targets:
        chain = _find_wrapped_fragment_chain(
            candidates,
            target=target,
            used_indexes=used_indexes,
        )
        if len(chain) < 2:
            continue
        _merge_candidate_chain(candidates, chain, target=target)
        used_indexes.update(chain)
        remove_indexes.update(chain[1:])

    if remove_indexes:
        candidates[:] = [
            candidate
            for index, candidate in enumerate(candidates)
            if index not in remove_indexes
        ]


def _normalize_candidate_texts(candidates: list[JsonDict]) -> None:
    for candidate in candidates:
        text = _normalize_table_cell_text(str(candidate.get("text") or ""))
        if text:
            candidate["text"] = text


def _find_wrapped_fragment_chain(
    candidates: list[JsonDict],
    *,
    target: str,
    used_indexes: set[int],
) -> list[int]:
    for index, candidate in enumerate(candidates):
        if index in used_indexes:
            continue
        text = _normalize_table_cell_text(str(candidate.get("text") or ""))
        if not text or text == target or not target.startswith(text):
            continue

        chain = [index]
        consumed = text
        current = candidate
        while consumed != target:
            next_index = _find_next_fragment_index(
                candidates,
                target_fragment=target[len(consumed):],
                current=current,
                used_indexes=used_indexes | set(chain),
            )
            if next_index is None:
                break
            next_text = _normalize_table_cell_text(str(candidates[next_index].get("text") or ""))
            consumed += next_text
            chain.append(next_index)
            current = candidates[next_index]

        if consumed == target:
            return chain
    return []


def _find_next_fragment_index(
    candidates: list[JsonDict],
    *,
    target_fragment: str,
    current: Mapping[str, Any],
    used_indexes: set[int],
) -> int | None:
    current_box = current.get("bounding_box")
    if not isinstance(current_box, BoundingBox):
        return None

    matches: list[tuple[float, float, int]] = []
    for index, candidate in enumerate(candidates):
        if index in used_indexes:
            continue
        text = _normalize_table_cell_text(str(candidate.get("text") or ""))
        if not text or not target_fragment.startswith(text):
            continue
        box = candidate.get("bounding_box")
        if not isinstance(box, BoundingBox):
            continue
        if _is_same_line_right_fragment(current_box, box) or _is_next_line_fragment(current_box, box):
            y_gap = max(0, box.y - (current_box.y + current_box.h))
            x_delta = abs((box.x + box.w / 2.0) - (current_box.x + current_box.w / 2.0))
            matches.append((float(y_gap), float(x_delta), index))
            continue

    if not matches:
        return None
    return min(matches)[2]


def _is_same_line_right_fragment(current_box: BoundingBox, next_box: BoundingBox) -> bool:
    vertical_overlap = _vertical_overlap_ratio(current_box, next_box)
    if vertical_overlap < 0.45:
        return False
    x_gap = next_box.x - (current_box.x + current_box.w)
    if x_gap < -max(6, min(current_box.w, next_box.w) * 0.5):
        return False
    return x_gap <= max(48, current_box.h * 2.5, next_box.h * 2.5)


def _is_next_line_fragment(current_box: BoundingBox, next_box: BoundingBox) -> bool:
    if next_box.y < current_box.y - max(4, current_box.h * 0.25):
        return False
    y_gap = max(0, next_box.y - (current_box.y + current_box.h))
    if y_gap > max(48, current_box.h * 2.5, next_box.h * 2.5):
        return False
    if _horizontal_overlap_ratio(current_box, next_box) >= 0.35:
        return True
    x_delta = abs((next_box.x + next_box.w / 2.0) - (current_box.x + current_box.w / 2.0))
    return x_delta <= max(48, current_box.w * 2.5, next_box.w * 2.5)


def _vertical_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    y0 = max(left.y, right.y)
    y1 = min(left.y + left.h, right.y + right.h)
    if y1 <= y0:
        return 0.0
    return float((y1 - y0) / max(1, min(left.h, right.h)))


def _merge_candidate_chain(candidates: list[JsonDict], chain: list[int], *, target: str) -> None:
    first = candidates[chain[0]]
    fragments = [candidates[index] for index in chain]
    boxes = [
        candidate.get("bounding_box")
        for candidate in fragments
        if isinstance(candidate.get("bounding_box"), BoundingBox)
    ]
    row_indexes = [
        int(candidate["row_index"])
        for candidate in fragments
        if candidate.get("row_index") is not None
    ]
    col_indexes = [
        int(candidate["col_index"])
        for candidate in fragments
        if candidate.get("col_index") is not None
    ]

    first["text"] = target
    first["bounding_box"] = _union_bounding_boxes(boxes)
    if row_indexes:
        first["row_index"] = min(row_indexes)
        first["row_span"] = 1
    if col_indexes:
        first["col_index"] = min(col_indexes)
        first["col_span"] = 1

    meta = dict(first.get("meta") or {})
    meta["merge_source"] = "paddle_pred_html"
    meta["merge_type"] = "wrapped_cell_text"
    first["meta"] = meta


def _html_cell_texts(html: str) -> list[str]:
    parser = _HtmlCellTextParser()
    try:
        parser.feed(html)
    except Exception:
        return []

    texts: list[str] = []
    seen: set[str] = set()
    for text in parser.cells:
        normalized = _normalize_table_cell_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        texts.append(normalized)
    return texts


class _HtmlCellTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_cell = False
        self._current: list[str] = []
        self.cells: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"td", "th"}:
            self._in_cell = True
            self._current = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._in_cell:
            text = " ".join(self._current).strip()
            if text:
                self.cells.append(text)
            self._in_cell = False
            self._current = []


def _normalize_table_cell_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)


def _horizontal_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    x0 = max(left.x, right.x)
    x1 = min(left.x + left.w, right.x + right.w)
    if x1 <= x0:
        return 0.0
    return float((x1 - x0) / max(1, min(left.w, right.w)))


def _cluster_centers(values: Sequence[float], *, tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - (sum(clusters[-1]) / len(clusters[-1]))) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _nearest_index(values: Sequence[float], target: float) -> int | None:
    if not values:
        return None
    return min(range(len(values)), key=lambda index: abs(values[index] - target))


def _resolve_scale(
    *,
    source_width: float | None,
    source_height: float | None,
    target_size: ImageSize | None,
) -> tuple[float, float, str]:
    if source_width and source_height and target_size is not None:
        return target_size.width / source_width, target_size.height / source_height, "mineru_layout"
    return 1.0, 1.0, "image_pixels"


def _coerce_bbox(
    raw_box: Any,
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> BoundingBox | None:
    if isinstance(raw_box, Mapping):
        if {"x", "y", "w", "h"}.issubset(raw_box):
            x = _coerce_float(raw_box.get("x"))
            y = _coerce_float(raw_box.get("y"))
            w = _coerce_float(raw_box.get("w"))
            h = _coerce_float(raw_box.get("h"))
            if x is None or y is None or w is None or h is None:
                return None
            return BoundingBox(
                x=int(round(offset_x + x * scale_x)),
                y=int(round(offset_y + y * scale_y)),
                w=max(0, int(round(w * scale_x))),
                h=max(0, int(round(h * scale_y))),
            )
        raw_box = list(raw_box.values())

    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)):
        return None

    if len(raw_box) >= 8 and len(raw_box) % 2 == 0 and all(_is_number(value) for value in raw_box):
        points = [(float(raw_box[i]), float(raw_box[i + 1])) for i in range(0, len(raw_box), 2)]
        x0 = min(point[0] for point in points)
        y0 = min(point[1] for point in points)
        x1 = max(point[0] for point in points)
        y1 = max(point[1] for point in points)
    elif len(raw_box) >= 4 and all(_is_number(value) for value in raw_box[:4]):
        x0, y0, x1, y1 = (float(value) for value in raw_box[:4])
    else:
        points = []
        for point in raw_box:
            if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
                px = _coerce_float(point[0])
                py = _coerce_float(point[1])
                if px is not None and py is not None:
                    points.append((px, py))
        if not points:
            return None
        x0 = min(point[0] for point in points)
        y0 = min(point[1] for point in points)
        x1 = max(point[0] for point in points)
        y1 = max(point[1] for point in points)

    scaled_x0 = offset_x + x0 * scale_x
    scaled_y0 = offset_y + y0 * scale_y
    scaled_x1 = offset_x + x1 * scale_x
    scaled_y1 = offset_y + y1 * scale_y
    return BoundingBox(
        x=int(round(min(scaled_x0, scaled_x1))),
        y=int(round(min(scaled_y0, scaled_y1))),
        w=max(0, int(round(abs(scaled_x1 - scaled_x0)))),
        h=max(0, int(round(abs(scaled_y1 - scaled_y0)))),
    )


def _union_bounding_boxes(boxes: Sequence[BoundingBox | None]) -> BoundingBox | None:
    real_boxes = [box for box in boxes if box is not None]
    if not real_boxes:
        return None
    x0 = min(box.x for box in real_boxes)
    y0 = min(box.y for box in real_boxes)
    x1 = max(box.x + box.w for box in real_boxes)
    y1 = max(box.y + box.h for box in real_boxes)
    return BoundingBox(x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0))


def _extract_result_json(result: Any) -> Any:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is not None:
        return payload
    if hasattr(result, "to_json"):
        return result.to_json()
    if isinstance(result, Mapping):
        return result
    raise PaddleTableStructureError("PaddleOCR returned a result without JSON payload")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _jsonable_candidate_for_api(candidate: Mapping[str, Any], *, send_file_bytes: bool) -> JsonDict:
    payload = _jsonable(candidate)
    if not isinstance(payload, dict) or not send_file_bytes:
        return payload if isinstance(payload, dict) else {}

    image_uri = _coerce_optional_str(payload.get("image_uri"))
    if image_uri is None:
        return payload
    image_path = _local_path_from_uri(image_uri)
    if image_path is None or not image_path.is_file():
        return payload
    payload["image_name"] = image_path.name
    payload["image_bytes_b64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return payload


def _should_send_file_bytes(api_url: str, options: Mapping[str, Any]) -> bool:
    configured = options.get("paddle_table_api_send_file_bytes")
    if configured is None:
        configured = os.environ.get("PADDLE_TABLE_API_SEND_FILE_BYTES", "auto")
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        host = (urllib.parse.urlparse(api_url).hostname or "").strip().lower()
        return host not in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}
    return _coerce_bool(configured, default=False)


def _read_local_file_b64(file_uri: str) -> str:
    path = _local_path_from_uri(file_uri)
    if path is None or not path.is_file():
        raise PaddleTableStructureError(f"Cannot send Paddle table API file bytes; local file does not exist: {file_uri}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _path_name_from_uri(file_uri: str) -> str | None:
    path = _local_path_from_uri(file_uri)
    if path is not None:
        return path.name
    parsed = urllib.parse.urlparse(file_uri)
    if parsed.path:
        return Path(urllib.request.url2pathname(parsed.path)).name
    return None


def _local_path_from_uri(file_uri: str) -> Path | None:
    parsed = urllib.parse.urlparse(file_uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    raw_path = urllib.request.url2pathname(parsed.path) if parsed.scheme == "file" else file_uri
    try:
        return Path(raw_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _unwrap_result_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("res"), Mapping):
        return payload["res"]
    return payload


def _as_mappings(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _as_list(value) if isinstance(item, Mapping)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return _as_list(value.tolist())
    return [value]


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_confidence(value: Any) -> float | None:
    score = _coerce_float(value)
    if score is None or score < 0.0 or score > 1.0:
        return None
    return score


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


def _structure_to_html(value: Any) -> str | None:
    fragments = [str(item) for item in _as_list(value)]
    html = "".join(fragments).strip()
    return html or None


def _image_size(image_uri: str | None) -> tuple[int | None, int | None]:
    if not image_uri:
        return None, None
    try:
        from PIL import Image

        with Image.open(image_uri) as image:
            return image.size
    except Exception:
        return None, None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
