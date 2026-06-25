from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from statistics import mean, median
import subprocess
import sys
import threading
from typing import Any
import urllib.error
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
    text_blocks_by_page: Mapping[int, tuple[TextBlock, ...]] = field(default_factory=dict)
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
            mode = "auto"
        mode = mode.strip().lower()

        if mode in {"auto", "paddle3_auto", "paddle_auto"}:
            return self._extract_with_paddle3_auto(
                file_uri=file_uri,
                output_dir=output_dir,
                target_page_sizes=target_page_sizes,
                options=opts,
                table_candidates_by_page=candidates,
                page_text_blocks_by_page=dict(page_text_blocks_by_page or {}),
            )

        if mode in {"table_structure", "table_structure_recognition", "mineru_crops"}:
            return self._extract_tables_from_mineru_crops(
                output_dir=output_dir,
                options=opts,
                table_candidates_by_page=candidates,
                page_text_blocks_by_page=dict(page_text_blocks_by_page or {}),
            )

        if mode in {"ppocrv5", "pp_ocrv5", "paddleocr", "ocr"}:
            return self._extract_text_with_ppocrv5(
                file_uri=_paddle_input_uri(file_uri, opts),
                output_dir=output_dir,
                target_page_sizes=target_page_sizes,
                options=opts,
                mode="ppocrv5",
            )

        if mode in {"paddleocr_vl", "paddleocrvl", "vl"}:
            return self._extract_with_paddleocr_vl(
                file_uri=_paddle_input_uri(file_uri, opts),
                output_dir=output_dir,
                target_page_sizes=target_page_sizes,
                options=opts,
                mode="paddleocr_vl",
            )

        return self._extract_tables_with_ppstructurev3(
            file_uri=_paddle_input_uri(file_uri, opts),
            output_dir=output_dir,
            target_page_sizes=target_page_sizes,
            options=opts,
        )

    def _extract_with_paddle3_auto(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any],
        table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
        page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
    ) -> PaddleTableStructureResult:
        paddle_file_uri = _paddle_input_uri(file_uri, options)
        if _has_seal_hints(
            options=options,
            table_candidates_by_page=table_candidates_by_page,
            page_text_blocks_by_page=page_text_blocks_by_page,
        ):
            return self._extract_text_and_seals_with_paddle(
                file_uri=paddle_file_uri,
                output_dir=output_dir,
                target_page_sizes=target_page_sizes,
                options=options,
                page_text_blocks_by_page=page_text_blocks_by_page,
            )

        if table_candidates_by_page:
            try:
                result = self._extract_tables_with_ppstructurev3(
                    file_uri=paddle_file_uri,
                    output_dir=output_dir,
                    target_page_sizes=target_page_sizes,
                    options=options,
                )
                if _should_fallback_ppstructurev3_result(
                    result,
                    table_candidates_by_page=table_candidates_by_page,
                    options=options,
                ):
                    fallback = self._extract_tables_from_mineru_crops(
                        output_dir=output_dir,
                        options=options,
                        table_candidates_by_page=table_candidates_by_page,
                        page_text_blocks_by_page=page_text_blocks_by_page,
                    )
                    meta = dict(fallback.meta)
                    meta["mode"] = "auto_ppstructurev3_quality_fallback_mineru_crops"
                    meta["auto_route"] = "table"
                    meta["fallback_from"] = "ppstructurev3"
                    meta["fallback_reason"] = "cell_coverage_below_mineru_html"
                    return PaddleTableStructureResult(
                        tables_by_page=fallback.tables_by_page,
                        text_blocks_by_page=fallback.text_blocks_by_page,
                        artifact_ref=fallback.artifact_ref,
                        meta=meta,
                    )
                meta = dict(result.meta)
                meta["mode"] = "auto_ppstructurev3"
                meta["auto_route"] = "table"
                return PaddleTableStructureResult(
                    tables_by_page=result.tables_by_page,
                    text_blocks_by_page=result.text_blocks_by_page,
                    artifact_ref=result.artifact_ref,
                    meta=meta,
                )
            except PaddleTableStructureError as exc:
                if not _coerce_bool(options.get("paddle_auto_fallback_to_mineru_crops"), True):
                    raise
                fallback = self._extract_tables_from_mineru_crops(
                    output_dir=output_dir,
                    options=options,
                    table_candidates_by_page=table_candidates_by_page,
                    page_text_blocks_by_page=page_text_blocks_by_page,
                )
                meta = dict(fallback.meta)
                meta["mode"] = "auto_ppstructurev3_fallback_mineru_crops"
                meta["auto_route"] = "table"
                meta["fallback_from"] = "ppstructurev3"
                meta["fallback_error"] = str(exc)
                return PaddleTableStructureResult(
                    tables_by_page=fallback.tables_by_page,
                    text_blocks_by_page=fallback.text_blocks_by_page,
                    artifact_ref=fallback.artifact_ref,
                    meta=meta,
                )

        return self._extract_text_with_ppocrv5(
            file_uri=paddle_file_uri,
            output_dir=output_dir,
            target_page_sizes=target_page_sizes,
            options=options,
            mode="ppocrv5",
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

        parsed = parse_paddle_structure_tables(
            raw_pages,
            target_page_sizes=target_page_sizes,
        )
        table_count = int(parsed.meta.get("table_count") or 0)
        cell_count = int(parsed.meta.get("cell_count") or 0)
        return _write_table_artifact(
            output_dir=output_dir,
            provider="paddleocr_ppstructurev3",
            mode="ppstructurev3",
            raw_results=raw_pages,
            tables_by_page=parsed.tables_by_page,
            text_blocks_by_page=parsed.text_blocks_by_page,
            table_count=table_count,
            cell_count=cell_count,
            source_file=file_uri,
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

    def _extract_text_with_ppocrv5(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any],
        mode: str,
    ) -> PaddleTableStructureResult:
        raw_pages, text_blocks_by_page, text_count = _predict_ppocrv5_text_blocks(
            file_uri=file_uri,
            target_page_sizes=target_page_sizes,
            options=options,
        )
        return _write_table_artifact(
            output_dir=output_dir,
            provider="paddleocr_ppocrv5",
            mode=mode,
            raw_results=raw_pages,
            tables_by_page={},
            text_blocks_by_page=text_blocks_by_page,
            table_count=0,
            cell_count=0,
            text_block_count=text_count,
            source_file=file_uri,
            replace_text_blocks=True,
        )

    def _extract_text_and_seals_with_paddle(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any],
        page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
    ) -> PaddleTableStructureResult:
        seal_blocks_by_page, seal_count, raw_seal_results = _predict_seal_crops_with_paddleocr_vl(
            output_dir=output_dir,
            options=options,
            page_text_blocks_by_page=page_text_blocks_by_page,
        )
        if seal_count <= 0:
            raise PaddleTableStructureError("PaddleOCR-VL seal crop recognition produced no text")
        raw_ocr_pages, text_blocks_by_page, ocr_text_count = _predict_layout_crop_text_blocks_with_ppocrv5(
            file_uri=file_uri,
            output_dir=output_dir,
            options=options,
            page_text_blocks_by_page=page_text_blocks_by_page,
        )
        if ocr_text_count <= 0:
            if not _coerce_bool(options.get("paddle_ocr_crop_fallback_to_full_page"), False):
                raise PaddleTableStructureError("PaddleOCR layout crop recognition produced no text")
            raw_ocr_pages, text_blocks_by_page, ocr_text_count = _predict_ppocrv5_text_blocks(
                file_uri=file_uri,
                target_page_sizes=target_page_sizes,
                options=options,
            )

        combined_text_blocks = _merge_text_blocks_by_page(text_blocks_by_page, seal_blocks_by_page)
        return _write_table_artifact(
            output_dir=output_dir,
            provider="paddleocr_ppocrv5_paddleocr_vl_seal",
            mode="seal_vl_crops_ppocrv5",
            raw_results=[
                {"provider": "paddleocr_ppocrv5", "pages": raw_ocr_pages},
                {"provider": "paddleocr_vl_seal_crops", "pages": raw_seal_results},
            ],
            tables_by_page={},
            text_blocks_by_page=combined_text_blocks,
            table_count=0,
            cell_count=0,
            text_block_count=ocr_text_count + seal_count,
            source_file=file_uri,
            replace_text_blocks=True,
        )

    def _extract_with_paddleocr_vl(
        self,
        *,
        file_uri: str,
        output_dir: Path,
        target_page_sizes: Mapping[int, ImageSize | None],
        options: Mapping[str, Any],
        mode: str,
    ) -> PaddleTableStructureResult:
        predict_kwargs = dict(_coerce_mapping(options.get("paddle_vl_predict_kwargs")) or {})
        predict_kwargs.setdefault("use_queues", False)
        predict_kwargs.setdefault(
            "max_new_tokens",
            _coerce_int(options.get("paddle_vl_max_new_tokens"), 2048),
        )
        raw_pages: list[JsonDict] = []
        try:
            model = _build_paddleocr_vl(options)
            result_iter = model.predict(input=file_uri, **predict_kwargs)
            for result in result_iter:
                raw_pages.append(_jsonable(_extract_result_json(result)))
        except Exception as exc:  # pragma: no cover - depends on Paddle runtime
            raise PaddleTableStructureError(
                f"PaddleOCR-VL recognition failed: {type(exc).__name__}: {exc!r}"
            ) from exc

        text_blocks_by_page, text_count = parse_paddle_ocr_text_blocks(
            raw_pages,
            target_page_sizes=target_page_sizes,
            provider="paddleocr_vl",
        )
        return _write_table_artifact(
            output_dir=output_dir,
            provider="paddleocr_vl",
            mode=mode,
            raw_results=raw_pages,
            tables_by_page={},
            text_blocks_by_page=text_blocks_by_page,
            table_count=0,
            cell_count=0,
            text_block_count=text_count,
            source_file=file_uri,
            replace_text_blocks=bool(text_blocks_by_page),
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
        payload = {
            "file_uri": file_uri,
            "output_dir": str(output_dir),
            "target_page_sizes": {
                str(page): size.model_dump(mode="json") if size is not None else None
                for page, size in target_page_sizes.items()
            },
            "options": _jsonable(opts),
            "table_candidates_by_page": {
                str(page): [_jsonable(candidate) for candidate in candidates]
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
        result = paddle_table_result_from_payload(response_payload["data"])
        meta = dict(result.meta)
        meta["transport"] = "http"
        meta["api_url"] = self.api_url
        return PaddleTableStructureResult(
            tables_by_page=result.tables_by_page,
            text_blocks_by_page=result.text_blocks_by_page,
            artifact_ref=result.artifact_ref,
            meta=meta,
        )


def paddle_table_result_to_payload(result: PaddleTableStructureResult) -> JsonDict:
    return {
        "tables_by_page": {
            str(page): [table.model_dump(mode="json") for table in tables]
            for page, tables in result.tables_by_page.items()
        },
        "text_blocks_by_page": {
            str(page): [block.model_dump(mode="json") for block in blocks]
            for page, blocks in result.text_blocks_by_page.items()
        },
        "artifact_ref": (
            result.artifact_ref.model_dump(mode="json") if result.artifact_ref is not None else None
        ),
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
    raw_text_blocks = payload.get("text_blocks_by_page") or {}
    text_blocks_by_page: dict[int, tuple[TextBlock, ...]] = {}
    if isinstance(raw_text_blocks, Mapping):
        for raw_page, raw_items in raw_text_blocks.items():
            try:
                page_index = int(raw_page)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
                continue
            text_blocks_by_page[page_index] = tuple(
                TextBlock.model_validate(item) for item in raw_items if isinstance(item, Mapping)
            )
    raw_artifact = payload.get("artifact_ref")
    artifact_ref = ArtifactRef.model_validate(raw_artifact) if isinstance(raw_artifact, Mapping) else None
    meta = dict(payload.get("meta") or {}) if isinstance(payload.get("meta"), Mapping) else {}
    return PaddleTableStructureResult(
        tables_by_page=tables_by_page,
        text_blocks_by_page=text_blocks_by_page,
        artifact_ref=artifact_ref,
        meta=meta,
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
    text_blocks_by_page: dict[int, list[TextBlock]] = {}
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
        page_blocks = _parse_ocr_payload_text_blocks(
            payload,
            page_index=page_index,
            scale_x=scale_x,
            scale_y=scale_y,
            coord_space=coord_space,
            provider="paddleocr_ppstructurev3",
        )
        if page_blocks:
            text_blocks_by_page.setdefault(page_index, []).extend(page_blocks)

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
                page_text_blocks=page_blocks,
            )
            if table is None:
                continue
            tables_by_page.setdefault(page_index, []).append(table)
            table_count += 1
            cell_count += len(table.cells)

    return PaddleTableStructureResult(
        tables_by_page={page: tuple(tables) for page, tables in tables_by_page.items()},
        text_blocks_by_page={page: tuple(blocks) for page, blocks in text_blocks_by_page.items()},
        artifact_ref=artifact_ref,
        meta={
            "provider": "paddleocr_ppstructurev3",
            "mode": "ppstructurev3",
            "table_count": table_count,
            "cell_count": cell_count,
            "text_block_count": sum(len(blocks) for blocks in text_blocks_by_page.values()),
        },
    )


def parse_paddle_ocr_text_blocks(
    page_payloads: Sequence[Mapping[str, Any]],
    *,
    target_page_sizes: Mapping[int, ImageSize | None] | None = None,
    provider: str,
) -> tuple[dict[int, tuple[TextBlock, ...]], int]:
    target_sizes = dict(target_page_sizes or {})
    text_blocks_by_page: dict[int, list[TextBlock]] = {}
    text_count = 0

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

        page_blocks = _parse_ocr_payload_text_blocks(
            payload,
            page_index=page_index,
            scale_x=scale_x,
            scale_y=scale_y,
            coord_space=coord_space,
            provider=provider,
        )
        if page_blocks:
            text_blocks_by_page.setdefault(page_index, []).extend(page_blocks)
            text_count += len(page_blocks)

    return {page: tuple(blocks) for page, blocks in text_blocks_by_page.items()}, text_count


def _predict_ppocrv5_text_blocks(
    *,
    file_uri: str,
    target_page_sizes: Mapping[int, ImageSize | None],
    options: Mapping[str, Any],
) -> tuple[list[JsonDict], dict[int, tuple[TextBlock, ...]], int]:
    predict_kwargs = dict(_coerce_mapping(options.get("paddle_ocr_predict_kwargs")) or {})
    raw_pages: list[JsonDict] = []
    try:
        model = _build_ppocr_v5(options)
        result_iter = model.predict(input=file_uri, **predict_kwargs)
        for result in result_iter:
            raw_pages.append(_jsonable(_extract_result_json(result)))
    except Exception as exc:  # pragma: no cover - depends on Paddle runtime
        raise PaddleTableStructureError(
            f"PaddleOCR PP-OCRv5 recognition failed: {type(exc).__name__}: {exc!r}"
        ) from exc

    text_blocks_by_page, text_count = parse_paddle_ocr_text_blocks(
        raw_pages,
        target_page_sizes=target_page_sizes,
        provider="paddleocr_ppocrv5",
    )
    return raw_pages, text_blocks_by_page, text_count


@dataclass(frozen=True)
class _TextCropCandidate:
    page_index: int
    bounding_box: BoundingBox
    source_block: TextBlock


def _predict_layout_crop_text_blocks_with_ppocrv5(
    *,
    file_uri: str,
    output_dir: Path,
    options: Mapping[str, Any],
    page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
) -> tuple[list[JsonDict], dict[int, tuple[TextBlock, ...]], int]:
    image_path = _local_image_path(file_uri)
    if not image_path.exists() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise PaddleTableStructureError("PaddleOCR layout crop recognition requires a local image source")

    candidates = _text_crop_candidates(page_text_blocks_by_page)
    if not candidates:
        raise PaddleTableStructureError("PaddleOCR layout crop recognition found no text blocks from MinerU layout")

    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as source_image:
            source = ImageOps.exif_transpose(source_image).convert("RGB")
    except Exception as exc:  # pragma: no cover - depends on image runtime
        raise PaddleTableStructureError(
            f"PaddleOCR layout crop source image failed to load: {type(exc).__name__}: {exc!r}"
        ) from exc

    crop_dir = output_dir / "paddle_text_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    scale_x, scale_y = _infer_layout_to_image_scale(
        candidates=candidates,
        image_width=source.width,
        image_height=source.height,
        options=options,
    )
    padding = _coerce_int(options.get("paddle_ocr_crop_padding"), 8)
    max_blocks = _coerce_int(options.get("paddle_ocr_layout_crop_max_blocks"), 120)
    predict_kwargs = dict(_coerce_mapping(options.get("paddle_text_rec_predict_kwargs")) or {})
    raw_results: list[JsonDict] = []
    text_blocks_by_page: dict[int, list[TextBlock]] = {}
    text_count = 0

    try:
        model = _build_text_recognition_model(options)
        for candidate_index, candidate in enumerate(candidates[:max_blocks]):
            crop_box = _scaled_crop_box(
                candidate.bounding_box,
                scale_x=scale_x,
                scale_y=scale_y,
                padding=padding,
                image_width=source.width,
                image_height=source.height,
            )
            if crop_box is None:
                continue
            crop_path = crop_dir / f"p{candidate.page_index:04d}_{candidate_index:04d}.jpg"
            source.crop(crop_box).save(crop_path, "JPEG", quality=95)
            crop_payloads: list[JsonDict] = []
            for result in model.predict(input=str(crop_path), **predict_kwargs):
                crop_payloads.append(_jsonable(_extract_result_json(result)))
            text = _extract_vl_result_text(crop_payloads)
            raw_results.append(
                {
                    "page_index": candidate.page_index,
                    "candidate_index": candidate_index,
                    "image_uri": str(crop_path),
                    "source_bbox": candidate.bounding_box.model_dump(mode="json"),
                    "crop_box": list(crop_box),
                    "result": crop_payloads,
                    "text": text,
                }
            )
            if not text:
                continue
            meta = dict(candidate.source_block.meta or {})
            meta.update(
                {
                    "source": "paddleocr_text_recognition_layout_crop",
                    "paddle_source": "text_recognition_layout_crop",
                    "image_uri": str(crop_path),
                    "coord_space": "mineru_layout",
                    "original_block_type": candidate.source_block.block_type,
                    "layout_to_image_scale_x": scale_x,
                    "layout_to_image_scale_y": scale_y,
                }
            )
            text_blocks_by_page.setdefault(candidate.page_index, []).append(
                TextBlock(
                    text=text,
                    bounding_box=candidate.bounding_box,
                    block_type=candidate.source_block.block_type or "text",
                    confidence=None,
                    meta=meta,
                )
            )
            text_count += 1
    except PaddleTableStructureError:
        raise
    except Exception as exc:  # pragma: no cover - depends on Paddle runtime
        raise PaddleTableStructureError(
            f"PaddleOCR layout crop recognition failed: {type(exc).__name__}: {exc!r}"
        ) from exc

    return raw_results, {page: tuple(blocks) for page, blocks in text_blocks_by_page.items()}, text_count


def _text_crop_candidates(
    page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
) -> list[_TextCropCandidate]:
    candidates: list[_TextCropCandidate] = []
    for page_index, blocks in dict(page_text_blocks_by_page or {}).items():
        for block in blocks:
            if _text_block_has_seal_hint(block):
                continue
            if block.bounding_box.w <= 0 or block.bounding_box.h <= 0:
                continue
            block_type = str(block.block_type or "").lower()
            if block_type in {"table", "table_body", "image"}:
                continue
            candidates.append(
                _TextCropCandidate(
                    page_index=int(page_index),
                    bounding_box=block.bounding_box,
                    source_block=block,
                )
            )
    candidates.sort(key=lambda item: (item.page_index, item.bounding_box.y, item.bounding_box.x))
    return candidates


def _infer_layout_to_image_scale(
    *,
    candidates: Sequence[Any],
    image_width: int,
    image_height: int,
    options: Mapping[str, Any],
) -> tuple[float, float]:
    configured_width = _coerce_float(options.get("paddle_layout_canvas_width"))
    configured_height = _coerce_float(options.get("paddle_layout_canvas_height"))
    if configured_width and configured_height and configured_width > 0 and configured_height > 0:
        return image_width / configured_width, image_height / configured_height

    max_right = max((item.bounding_box.x + item.bounding_box.w for item in candidates), default=0)
    max_bottom = max((item.bounding_box.y + item.bounding_box.h for item in candidates), default=0)
    if max_right <= 0 or max_bottom <= 0 or image_width <= 0 or image_height <= 0:
        return 1.0, 1.0

    aspect = image_width / image_height
    width_margin = _coerce_float(options.get("paddle_layout_canvas_width_margin"))
    if width_margin is None:
        width_margin = 1.06
    height_margin = _coerce_float(options.get("paddle_layout_canvas_height_margin"))
    if height_margin is None:
        height_margin = 1.02
    inferred_width = max(float(max_right) * width_margin, float(max_bottom) * aspect * height_margin)
    inferred_height = inferred_width / aspect if aspect > 0 else float(max_bottom) * height_margin
    if inferred_width <= 0 or inferred_height <= 0:
        return 1.0, 1.0
    return image_width / inferred_width, image_height / inferred_height


def _scaled_crop_box(
    bbox: BoundingBox,
    *,
    scale_x: float,
    scale_y: float,
    padding: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x0 = int(round((bbox.x - padding) * scale_x))
    y0 = int(round((bbox.y - padding) * scale_y))
    x1 = int(round((bbox.x + bbox.w + padding) * scale_x))
    y1 = int(round((bbox.y + bbox.h + padding) * scale_y))
    x0 = max(0, min(image_width, x0))
    y0 = max(0, min(image_height, y0))
    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


@dataclass(frozen=True)
class _SealCropCandidate:
    page_index: int
    image_uri: str
    bounding_box: BoundingBox
    source_block: TextBlock


def _predict_seal_crops_with_paddleocr_vl(
    *,
    output_dir: Path,
    options: Mapping[str, Any],
    page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
) -> tuple[dict[int, tuple[TextBlock, ...]], int, list[JsonDict]]:
    candidates = _seal_crop_candidates(
        output_dir=output_dir,
        page_text_blocks_by_page=page_text_blocks_by_page,
    )
    if not candidates:
        raise PaddleTableStructureError("PaddleOCR-VL seal route found no seal crop image from MinerU layout")

    init_overrides = dict(_coerce_mapping(options.get("paddle_vl_init_kwargs")) or {})
    init_overrides.setdefault("use_layout_detection", False)
    init_overrides.setdefault("use_seal_recognition", True)
    init_overrides.setdefault("use_ocr_for_image_block", True)
    init_overrides.setdefault("use_queues", False)
    init_overrides.setdefault("vl_rec_max_concurrency", 1)
    vl_options = dict(options)
    vl_options["paddle_vl_init_kwargs"] = init_overrides

    model = None
    if not _coerce_bool(options.get("paddle_vl_seal_subprocess"), True):
        try:
            model = _build_paddleocr_vl(vl_options)
        except Exception as exc:  # pragma: no cover - depends on Paddle runtime
            raise PaddleTableStructureError(
                f"PaddleOCR-VL seal crop model failed to load: {type(exc).__name__}: {exc!r}"
            ) from exc

    seal_blocks_by_page: dict[int, list[TextBlock]] = {}
    raw_results: list[JsonDict] = []
    seal_count = 0
    failures: list[str] = []
    for candidate_index, candidate in enumerate(candidates):
        try:
            raw_payload = _predict_one_seal_crop_with_paddleocr_vl(
                model=model,
                image_uri=candidate.image_uri,
                options=options,
            )
        except Exception as exc:  # pragma: no cover - depends on Paddle runtime
            failures.append(
                f"{candidate.image_uri}: {type(exc).__name__}: {exc!r}"
            )
            continue

        text = _extract_vl_result_text(raw_payload)
        raw_results.append(
            {
                "page_index": candidate.page_index,
                "candidate_index": candidate_index,
                "image_uri": candidate.image_uri,
                "source_block": candidate.source_block.model_dump(mode="json"),
                "result": raw_payload,
                "text": text,
            }
        )
        if not text:
            continue
        meta = dict(candidate.source_block.meta or {})
        meta.update(
            {
                "source": "paddleocr_vl_seal_crop",
                "paddle_source": "vl_rec_model",
                "image_uri": candidate.image_uri,
                "coord_space": "mineru_layout",
                "original_block_type": candidate.source_block.block_type,
            }
        )
        seal_blocks_by_page.setdefault(candidate.page_index, []).append(
            TextBlock(
                text=text,
                bounding_box=candidate.bounding_box,
                block_type="seal_text",
                confidence=None,
                meta=meta,
            )
        )
        seal_count += 1

    if seal_count <= 0:
        details = "; ".join(failures[:3])
        suffix = f"; failures: {details}" if details else ""
        raise PaddleTableStructureError(
            f"PaddleOCR-VL seal crop recognition produced no text{suffix}"
        )

    return {page: tuple(blocks) for page, blocks in seal_blocks_by_page.items()}, seal_count, raw_results


def _predict_one_seal_crop_with_paddleocr_vl(
    *,
    model: Any,
    image_uri: str,
    options: Mapping[str, Any],
) -> JsonDict:
    if _coerce_bool(options.get("paddle_vl_seal_subprocess"), True):
        return _predict_one_seal_crop_with_paddleocr_vl_subprocess(
            image_uri=image_uri,
            options=options,
        )

    prompt = _coerce_optional_str(options.get("paddle_vl_seal_prompt")) or "Seal Recognition:"
    direct_model = getattr(getattr(model, "paddlex_pipeline", None), "vl_rec_model", None)
    if direct_model is not None:
        image_path = _local_image_path(image_uri)
        try:
            import numpy as np
            from PIL import Image

            with Image.open(image_path) as image:
                image_array = np.array(image.convert("RGB"))
            raw_items = []
            result_iter = direct_model.predict(
                [{"image": image_array, "query": prompt}],
                min_pixels=_coerce_int(options.get("paddle_vl_seal_min_pixels"), 3136),
                max_pixels=_coerce_int(options.get("paddle_vl_seal_max_pixels"), 50176),
                max_new_tokens=_coerce_int(options.get("paddle_vl_seal_max_new_tokens"), 64),
                use_cache=_coerce_bool(options.get("paddle_vl_seal_use_cache"), True),
            )
            for result in result_iter:
                try:
                    raw_items.append(_jsonable(_extract_result_json(result)))
                except PaddleTableStructureError:
                    raw_items.append(_jsonable(result))
            return {
                "provider": "paddleocr_vl",
                "mode": "direct_vl_rec_model",
                "prompt": prompt,
                "result": raw_items,
            }
        except Exception as exc:
            direct_error = f"{type(exc).__name__}: {exc!r}"
        else:
            direct_error = None
    else:
        direct_error = "vl_rec_model_unavailable"

    predict_kwargs = dict(_coerce_mapping(options.get("paddle_vl_seal_predict_kwargs")) or {})
    predict_kwargs.setdefault("use_queues", False)
    predict_kwargs.setdefault("use_layout_detection", False)
    predict_kwargs.setdefault("use_seal_recognition", True)
    predict_kwargs.setdefault("prompt_label", "seal")
    predict_kwargs.setdefault("max_new_tokens", _coerce_int(options.get("paddle_vl_seal_max_new_tokens"), 64))
    raw_pages: list[JsonDict] = []
    result_iter = model.predict(input=image_uri, **predict_kwargs)
    for result in result_iter:
        raw_pages.append(_jsonable(_extract_result_json(result)))
    return {
        "provider": "paddleocr_vl",
        "mode": "pipeline_predict",
        "prompt": prompt,
        "direct_error": direct_error,
        "result": raw_pages,
    }


def _predict_one_seal_crop_with_paddleocr_vl_subprocess(
    *,
    image_uri: str,
    options: Mapping[str, Any],
) -> JsonDict:
    prompt = _coerce_optional_str(options.get("paddle_vl_seal_prompt")) or "Seal Recognition:"
    timeout_seconds = _coerce_int(options.get("paddle_vl_seal_subprocess_timeout_seconds"), 600)
    device = _resolve_paddle_device(options) or ""
    result_path = str(_local_image_path(image_uri).with_suffix(".vl_result.json"))
    script = r"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCRVL


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def result_json(value):
    payload = getattr(value, "json", None)
    if callable(payload):
        payload = payload()
    if payload is not None:
        return payload
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, dict):
        return value
    return {"repr": repr(value)}


image_uri = sys.argv[1]
prompt = sys.argv[2]
device = sys.argv[3] or None
min_pixels = int(sys.argv[4])
max_pixels = int(sys.argv[5])
max_new_tokens = int(sys.argv[6])
result_path = Path(sys.argv[7])
model_kwargs = {
    "use_layout_detection": False,
    "use_seal_recognition": True,
    "use_ocr_for_image_block": True,
    "use_queues": False,
    "vl_rec_max_concurrency": 1,
}
if device:
    model_kwargs["device"] = device
model = PaddleOCRVL(**model_kwargs)
with Image.open(image_uri) as image:
    image_array = np.array(image.convert("RGB"))
items = []
for result in model.paddlex_pipeline.vl_rec_model.predict(
    [{"image": image_array, "query": prompt}],
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    max_new_tokens=max_new_tokens,
    use_cache=True,
):
    items.append(jsonable(result_json(result)))
payload = {
    "provider": "paddleocr_vl",
    "mode": "subprocess_direct_vl_rec_model",
    "prompt": prompt,
    "result": items,
}
result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
"""
    command = [
        sys.executable,
        "-c",
        script,
        image_uri,
        prompt,
        device,
        str(_coerce_int(options.get("paddle_vl_seal_min_pixels"), 3136)),
        str(_coerce_int(options.get("paddle_vl_seal_max_pixels"), 50176)),
        str(_coerce_int(options.get("paddle_vl_seal_max_new_tokens"), 64)),
        result_path,
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_seconds,
    )
    output_path = Path(result_path)
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    if completed.returncode != 0:
        raise PaddleTableStructureError(
            "PaddleOCR-VL seal subprocess failed: "
            f"returncode={completed.returncode} result_path={result_path}"
        )
    raise PaddleTableStructureError(
        "PaddleOCR-VL seal subprocess did not emit a JSON result: "
        f"result_path={result_path}"
    )


def _seal_crop_candidates(
    *,
    output_dir: Path,
    page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
) -> list[_SealCropCandidate]:
    candidates: list[_SealCropCandidate] = []
    for page_index, blocks in dict(page_text_blocks_by_page or {}).items():
        for block in blocks:
            if not _text_block_has_seal_hint(block):
                continue
            image_uri = _resolve_seal_crop_image_uri(block, output_dir=output_dir)
            if not image_uri:
                continue
            candidates.append(
                _SealCropCandidate(
                    page_index=int(page_index),
                    image_uri=image_uri,
                    bounding_box=block.bounding_box,
                    source_block=block,
                )
            )
    return candidates


def _resolve_seal_crop_image_uri(block: TextBlock, *, output_dir: Path) -> str | None:
    meta = dict(block.meta or {})
    raw_uri = (
        _coerce_optional_str(meta.get("image_uri"))
        or _coerce_optional_str(meta.get("image_path"))
        or _coerce_optional_str(meta.get("img_path"))
    )
    if raw_uri is None:
        return None
    if raw_uri.startswith("file://"):
        path = _local_image_path(raw_uri)
        return str(path) if path.exists() else None
    raw_path = Path(raw_uri)
    if raw_path.is_absolute():
        return str(raw_path) if raw_path.exists() else None
    for candidate in (
        output_dir / raw_path,
        output_dir / "images" / raw_path.name,
        output_dir.parent / raw_path,
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _local_image_path(image_uri: str) -> Path:
    if image_uri.startswith("file://"):
        return Path(image_uri[7:])
    return Path(image_uri)


def _extract_vl_result_text(raw_payload: Any) -> str:
    candidates: list[str] = []
    _collect_vl_text_candidates(raw_payload, candidates)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = _normalize_vl_text(candidate)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return "\n".join(deduped)


def _collect_vl_text_candidates(value: Any, candidates: list[str]) -> None:
    if isinstance(value, str):
        candidates.append(value)
        return
    if isinstance(value, Mapping):
        for key in ("result", "text", "rec_text", "block_content", "content", "markdown", "markdown_text"):
            item = value.get(key)
            if item is not None:
                _collect_vl_text_candidates(item, candidates)
        for key in ("rec_texts", "texts"):
            for item in _as_list(value.get(key)):
                if isinstance(item, str):
                    candidates.append(item)
        for key in ("res", "vl_rec_res", "seal_res", "seal_ocr_res", "ocr_res"):
            item = value.get(key)
            if isinstance(item, (Mapping, list, tuple)):
                _collect_vl_text_candidates(item, candidates)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_vl_text_candidates(item, candidates)


def _normalize_vl_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text


def _merge_text_blocks_by_page(
    *block_groups: Mapping[int, Sequence[TextBlock]],
) -> dict[int, tuple[TextBlock, ...]]:
    merged: dict[int, list[TextBlock]] = {}
    seen_by_page: dict[int, set[tuple[str, int, int, int, int, str | None]]] = {}
    for group in block_groups:
        for page_index, blocks in dict(group or {}).items():
            page = int(page_index)
            seen = seen_by_page.setdefault(page, set())
            for block in blocks:
                key = (
                    _normalize_vl_text(block.text),
                    block.bounding_box.x,
                    block.bounding_box.y,
                    block.bounding_box.w,
                    block.bounding_box.h,
                    block.block_type,
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(page, []).append(block)
    for blocks in merged.values():
        blocks.sort(key=lambda block: (block.bounding_box.y, block.bounding_box.x, block.block_type or ""))
    return {page: tuple(blocks) for page, blocks in merged.items()}


def _text_block_has_seal_hint(block: TextBlock) -> bool:
    if _value_has_seal_hint(block.block_type):
        return True
    meta = block.meta
    return isinstance(meta, Mapping) and _mapping_has_seal_hint(meta)


def _write_table_artifact(
    *,
    output_dir: Path,
    provider: str,
    mode: str,
    raw_results: Sequence[Mapping[str, Any]],
    tables_by_page: Mapping[int, Sequence[TableBlock]],
    text_blocks_by_page: Mapping[int, Sequence[TextBlock]] | None = None,
    table_count: int,
    cell_count: int,
    text_block_count: int = 0,
    source_file: str | None = None,
    replace_text_blocks: bool = False,
) -> PaddleTableStructureResult:
    artifact_path = output_dir / "paddle_table_structure.json"
    artifact_payload = {
        "provider": provider,
        "mode": mode,
        "source_file": source_file,
        "table_count": table_count,
        "cell_count": cell_count,
        "text_block_count": text_block_count,
        "tables_by_page": {
            str(page): [table.model_dump(mode="json") for table in tables]
            for page, tables in tables_by_page.items()
        },
        "text_blocks_by_page": {
            str(page): [block.model_dump(mode="json") for block in blocks]
            for page, blocks in dict(text_blocks_by_page or {}).items()
        },
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
            "text_block_count": text_block_count,
        },
    )
    return PaddleTableStructureResult(
        tables_by_page={page: tuple(tables) for page, tables in tables_by_page.items()},
        text_blocks_by_page={
            page: tuple(blocks) for page, blocks in dict(text_blocks_by_page or {}).items()
        },
        artifact_ref=artifact_ref,
        meta={
            "provider": provider,
            "mode": mode,
            "table_count": table_count,
            "cell_count": cell_count,
            "text_block_count": text_block_count,
            "replace_text_blocks": replace_text_blocks,
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
    device = (
        _coerce_optional_str(options.get("paddle_ppstructurev3_device"))
        or _coerce_optional_str(os.environ.get("MINERU_PPSTRUCTURE_DEVICE"))
        or _resolve_paddle_device(options)
    )
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


def _build_ppocr_v5(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PaddleTableStructureError("PaddleOCR PP-OCRv5 is not installed") from exc

    init_kwargs: JsonDict = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "lang": _coerce_optional_str(options.get("lang")) or "ch",
        "ocr_version": _coerce_optional_str(options.get("paddle_ocr_version")) or "PP-OCRv5",
    }
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_ocr_init_kwargs")) or {})

    return _cached_paddle_object(
        cache_prefix="ppocrv5",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_table_disable_pipeline_cache")),
        factory=lambda kwargs: PaddleOCR(**kwargs),
        fallback_keys={"use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation", "device"},
    )


def _build_text_recognition_model(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import TextRecognition
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PaddleTableStructureError("PaddleOCR TextRecognition is not installed") from exc

    init_kwargs: JsonDict = {
        "model_name": _coerce_optional_str(options.get("paddle_text_rec_model_name")) or "PP-OCRv5_server_rec",
    }
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_text_rec_init_kwargs")) or {})

    return _cached_paddle_object(
        cache_prefix="text_recognition",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_table_disable_pipeline_cache")),
        factory=lambda kwargs: TextRecognition(**kwargs),
        fallback_keys={"device"},
    )


def _build_paddleocr_vl(options: Mapping[str, Any]) -> Any:
    _prepare_paddle_runtime()
    try:
        from paddleocr import PaddleOCRVL
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PaddleTableStructureError("PaddleOCR-VL is not installed") from exc

    init_kwargs: JsonDict = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": True,
        "use_ocr_for_image_block": True,
        "use_queues": False,
        "vl_rec_max_concurrency": 1,
    }
    device = _resolve_paddle_device(options)
    if device:
        init_kwargs["device"] = device
    init_kwargs.update(_coerce_mapping(options.get("paddle_vl_init_kwargs")) or {})

    return _cached_paddle_object(
        cache_prefix="paddleocrvl",
        init_kwargs=init_kwargs,
        disable_cache=_coerce_bool(options.get("paddle_table_disable_pipeline_cache")),
        factory=lambda kwargs: PaddleOCRVL(**kwargs),
        fallback_keys={"use_doc_orientation_classify", "use_doc_unwarping", "device"},
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
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _resolve_paddle_device(options: Mapping[str, Any]) -> str | None:
    configured = (
        _coerce_optional_str(options.get("paddle_device"))
        or _coerce_optional_str(os.environ.get("MINERU_PADDLE_DEVICE"))
    )
    if configured and configured.strip().lower() != "auto":
        return configured

    selected = _select_auto_paddle_gpu_device()
    if selected:
        return selected

    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        return None
    return None


def _select_auto_paddle_gpu_device() -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    rows: list[tuple[int, int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    if not rows:
        return None

    visible_devices = _visible_cuda_device_indexes()
    if visible_devices is not None:
        rows = [(index, free_mb) for index, free_mb in rows if index in visible_devices]
        if not rows:
            return None
        physical_index = max(rows, key=lambda item: item[1])[0]
        return f"gpu:{visible_devices.index(physical_index)}"

    physical_index = max(rows, key=lambda item: item[1])[0]
    return f"gpu:{physical_index}"


def _visible_cuda_device_indexes() -> list[int] | None:
    raw = _coerce_optional_str(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"", "all"}:
        return None
    if normalized in {"none", "void", "-1"}:
        return []
    indexes: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item.isdigit():
            return None
        indexes.append(int(item))
    return indexes


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

    candidate_html = _coerce_optional_str(candidate.get("html")) or _coerce_optional_str(
        candidate.get("table_body")
    )
    html = (
        _coerce_optional_str(payload.get("pred_html"))
        or _structure_to_html(payload.get("structure"))
        or candidate_html
    )
    # Candidate boxes from TableStructureRecognition already represent cells.
    # Keep the original grid order so HTML cell text can align 1:1.
    _assign_grid_indices(candidates)
    if candidate_html:
        _fill_cell_text_from_html(
            candidates,
            candidate_html,
            context_text=_join_text_block_content(page_text_blocks),
        )
    _merge_wrapped_cell_fragments_from_html(candidates, html)
    _sort_candidates_row_major(candidates)
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
    page_text_blocks: Sequence[TextBlock],
) -> TableBlock | None:
    ocr_pred = _coerce_mapping(raw_table.get("table_ocr_pred")) or {}
    texts = [str(value).strip() for value in _as_list(ocr_pred.get("rec_texts"))]
    scores = [_coerce_confidence(value) for value in _as_list(ocr_pred.get("rec_scores"))]
    rec_boxes = _as_list(ocr_pred.get("rec_boxes"))
    cell_boxes = _as_list(raw_table.get("cell_box_list"))

    candidates: list[JsonDict] = []
    if cell_boxes:
        for cell_index, raw_box in enumerate(cell_boxes):
            bbox = _coerce_bbox(
                raw_box,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            if bbox is None:
                continue
            candidates.append(
                {
                    "cell_id": f"p{page_index}-t{table_index}-c{cell_index}",
                    "text": "",
                    "bounding_box": bbox,
                    "confidence": None,
                    "meta": {
                        "source": "paddleocr_table",
                        "paddle_index": cell_index,
                        "bbox_source": "cell_box_list",
                    },
                }
            )
        _assign_grid_indices(candidates)
        _fill_cell_text_from_ocr_fragments(
            candidates,
            texts=texts,
            scores=scores,
            raw_boxes=rec_boxes or cell_boxes,
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
            bbox_source="table_ocr_pred.rec_boxes" if rec_boxes else "cell_box_list",
        )
        if _header_row_is_bottom_heavy(candidates):
            active_bounds = _active_candidate_text_bounds(candidates)
            _reset_candidate_fragment_text(candidates)
            _fill_cell_text_from_ocr_fragments(
                candidates,
                texts=texts,
                scores=scores,
                raw_boxes=rec_boxes or cell_boxes,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
                bbox_source="table_ocr_pred.rec_boxes" if rec_boxes else "cell_box_list",
                box_transform="flip_xy",
                transform_bounds=active_bounds,
            )
    else:
        raw_boxes = rec_boxes
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
                        "bbox_source": "table_ocr_pred.rec_boxes",
                    },
                }
            )

    html = _coerce_optional_str(raw_table.get("pred_html"))
    _assign_grid_indices(candidates)
    _merge_wrapped_cell_fragments_from_html(candidates, html)
    _fill_suspicious_cell_text_from_page_blocks(candidates, page_text_blocks)
    _sort_candidates_row_major(candidates)
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


def _fill_cell_text_from_ocr_fragments(
    candidates: list[JsonDict],
    *,
    texts: Sequence[str],
    scores: Sequence[float | None],
    raw_boxes: Sequence[Any],
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
    bbox_source: str,
    box_transform: str | None = None,
    transform_bounds: BoundingBox | None = None,
) -> None:
    if not candidates or not texts:
        return

    fragments: list[JsonDict] = []
    for index, text in enumerate(texts):
        normalized = _normalize_table_cell_text(text)
        if not normalized:
            continue
        bbox = (
            _coerce_bbox(
                raw_boxes[index],
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            if index < len(raw_boxes)
            else None
        )
        if bbox is None:
            continue
        fragments.append(
            {
                "text": normalized,
                "bounding_box": bbox,
                "confidence": scores[index] if index < len(scores) else None,
                "source_index": index,
            }
        )

    if not fragments:
        return

    if box_transform:
        resolved_transform_bounds = transform_bounds or _union_bounding_boxes(
            [
                box
                for box in (fragment.get("bounding_box") for fragment in fragments)
                if isinstance(box, BoundingBox)
            ]
        )
        if resolved_transform_bounds is not None:
            for fragment in fragments:
                box = fragment.get("bounding_box")
                if isinstance(box, BoundingBox):
                    fragment["bounding_box"] = _transform_fragment_box(
                        box,
                        grid_bounds=resolved_transform_bounds,
                        transform=box_transform,
                    )

    assignments: dict[int, list[JsonDict]] = {index: [] for index in range(len(candidates))}
    for fragment in fragments:
        fragment_box = fragment.get("bounding_box")
        if not isinstance(fragment_box, BoundingBox):
            continue
        best_index: int | None = None
        best_score = 0.0
        for cell_index, candidate in enumerate(candidates):
            cell_box = candidate.get("bounding_box")
            if not isinstance(cell_box, BoundingBox):
                continue
            overlap = _bbox_intersection_ratio(fragment_box, cell_box)
            inside = _bbox_center_inside(fragment_box, cell_box)
            score = overlap + (1.0 if inside else 0.0)
            if score > best_score:
                best_score = score
                best_index = cell_index
        if best_index is not None and best_score >= 0.2:
            assignments[best_index].append(fragment)

    for cell_index, hits in assignments.items():
        if not hits:
            continue
        text = _compose_cell_text_from_hits(hits)
        if not text:
            continue
        candidate = candidates[cell_index]
        candidate["text"] = text
        confidences = [value for value in (item.get("confidence") for item in hits) if value is not None]
        if confidences:
            candidate["confidence"] = float(max(confidences))
        meta = dict(candidate.get("meta") or {})
        meta["text_source"] = "table_ocr_pred.fragments"
        meta["ocr_fragment_count"] = len(hits)
        meta["ocr_fragment_indexes"] = [int(item["source_index"]) for item in hits]
        meta["ocr_fragment_bbox_source"] = bbox_source
        if box_transform:
            meta["ocr_fragment_box_transform"] = box_transform
        candidate["meta"] = meta


def _compose_cell_text_from_hits(hits: Sequence[Mapping[str, Any]]) -> str:
    expanded_hits: list[JsonDict] = []
    for hit in hits:
        expanded_hits.extend(_expand_vertical_cell_fragment(hit))
    expanded_hits.sort(key=lambda item: _cell_fragment_sort_key(item["bounding_box"]))
    return _normalize_table_cell_text("".join(str(item.get("text") or "") for item in expanded_hits))


def _expand_vertical_cell_fragment(hit: Mapping[str, Any]) -> list[JsonDict]:
    text = str(hit.get("text") or "")
    box = hit.get("bounding_box")
    if not isinstance(box, BoundingBox):
        return [dict(hit)]
    if not _should_expand_vertical_fragment(text, box):
        return [dict(hit)]

    char_count = len(text)
    step = max(1.0, float(box.h) / max(1, char_count))
    expanded: list[JsonDict] = []
    for index, char in enumerate(text):
        y0 = int(round(box.y + step * index))
        y1 = int(round(box.y + step * (index + 1)))
        char_box = BoundingBox(
            x=int(box.x),
            y=y0,
            w=int(box.w),
            h=max(1, y1 - y0),
        )
        item = dict(hit)
        item["text"] = char
        item["bounding_box"] = char_box
        expanded.append(item)
    return expanded


def _should_expand_vertical_fragment(text: str, box: BoundingBox) -> bool:
    value = str(text or "").strip()
    if len(value) < 2:
        return False
    if box.w <= 0 or box.h <= 0:
        return False
    avg_char_height = float(box.h) / float(len(value))
    ratio = avg_char_height / max(1.0, float(box.w))
    if ratio < 0.55 or ratio > 1.8:
        return False
    return box.h > box.w * 1.1


def _transform_fragment_box(
    box: BoundingBox,
    *,
    grid_bounds: BoundingBox,
    transform: str,
) -> BoundingBox:
    if transform == "flip_xy":
        return BoundingBox(
            x=int(grid_bounds.x + grid_bounds.w - (box.x - grid_bounds.x) - box.w),
            y=int(grid_bounds.y + grid_bounds.h - (box.y - grid_bounds.y) - box.h),
            w=int(box.w),
            h=int(box.h),
        )
    return box


def _parse_ocr_payload_text_blocks(
    payload: Mapping[str, Any],
    *,
    page_index: int,
    scale_x: float,
    scale_y: float,
    coord_space: str,
    provider: str,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    seen: set[tuple[str, int, int, int, int]] = set()

    def add_block(
        *,
        text: Any,
        raw_bbox: Any,
        confidence: Any = None,
        block_type: str | None = None,
        source: str,
    ) -> None:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return
        bbox = _coerce_bbox(raw_bbox, scale_x=scale_x, scale_y=scale_y, offset_x=0.0, offset_y=0.0)
        if bbox is None:
            return
        key = (normalized_text, bbox.x, bbox.y, bbox.w, bbox.h)
        if key in seen:
            return
        seen.add(key)
        blocks.append(
            TextBlock(
                text=normalized_text,
                bounding_box=bbox,
                block_type=block_type or "text",
                confidence=_coerce_confidence(confidence),
                meta={
                    "source": provider,
                    "paddle_source": source,
                    "coord_space": coord_space,
                },
            )
        )

    for item in _as_mappings(payload.get("parsing_res_list")):
        add_block(
            text=item.get("block_content") or item.get("text") or item.get("content"),
            raw_bbox=item.get("block_bbox") or item.get("bbox") or item.get("poly"),
            confidence=item.get("score") or item.get("confidence"),
            block_type=_coerce_optional_str(item.get("block_label") or item.get("type")),
            source="parsing_res_list",
        )

    for item in _as_mappings(payload.get("layout_res_list")):
        add_block(
            text=item.get("text") or item.get("content") or item.get("block_content"),
            raw_bbox=item.get("bbox") or item.get("block_bbox") or item.get("poly"),
            confidence=item.get("score") or item.get("confidence"),
            block_type=_coerce_optional_str(item.get("label") or item.get("type")),
            source="layout_res_list",
        )

    ocr_payloads = [
        payload,
        *[
            item
            for key in ("ocr_res", "overall_ocr_res", "seal_res", "seal_ocr_res")
            if isinstance((item := payload.get(key)), Mapping)
        ],
    ]
    for ocr_payload in ocr_payloads:
        texts = [str(value).strip() for value in _as_list(ocr_payload.get("rec_texts"))]
        if not texts:
            continue
        scores = [_coerce_confidence(value) for value in _as_list(ocr_payload.get("rec_scores"))]
        raw_boxes = (
            _as_list(ocr_payload.get("rec_polys"))
            or _as_list(ocr_payload.get("dt_polys"))
            or _as_list(ocr_payload.get("rec_boxes"))
            or _as_list(ocr_payload.get("dt_boxes"))
            or _as_list(ocr_payload.get("boxes"))
        )
        for index, text in enumerate(texts):
            raw_bbox = raw_boxes[index] if index < len(raw_boxes) else None
            add_block(
                text=text,
                raw_bbox=raw_bbox,
                confidence=scores[index] if index < len(scores) else None,
                block_type="seal_text" if ocr_payload is not payload and "seal" in str(ocr_payload).lower() else "text",
                source="ocr_res",
            )

    for key in ("layout_parsing_result", "doc_preprocessor_res", "vl_rec_res", "result"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested is not payload:
            blocks.extend(
                block
                for block in _parse_ocr_payload_text_blocks(
                    nested,
                    page_index=page_index,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    coord_space=coord_space,
                    provider=provider,
                )
                if (block.text, block.bounding_box.x, block.bounding_box.y, block.bounding_box.w, block.bounding_box.h)
                not in seen
            )

    markdown_text = _coerce_optional_str(payload.get("markdown")) or _coerce_optional_str(
        payload.get("markdown_text")
    )
    if markdown_text and not blocks:
        page_bbox = _full_page_bbox(payload)
        add_block(
            text=markdown_text,
            raw_bbox=page_bbox,
            block_type="markdown",
            source="markdown",
        )

    return blocks


def _fill_cell_text_from_page_blocks(candidates: list[JsonDict], text_blocks: Sequence[TextBlock]) -> None:
    used_block_indexes: set[int] = set()
    sorted_blocks = sorted(
        [
            (index, block)
            for index, block in enumerate(text_blocks)
            if block.bounding_box is not None
            and block.block_type in {"text", "seal_text"}
            and block.text.strip()
        ],
        key=lambda item: (item[1].bounding_box.y, item[1].bounding_box.x),
    )
    for candidate in candidates:
        if str(candidate.get("text") or "").strip():
            continue
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


def _fill_suspicious_cell_text_from_page_blocks(
    candidates: list[JsonDict],
    text_blocks: Sequence[TextBlock],
) -> None:
    if not candidates or not text_blocks:
        return

    sorted_blocks = sorted(
        [
            (index, block)
            for index, block in enumerate(text_blocks)
            if block.bounding_box is not None and block.block_type != "table_cell" and block.text.strip()
        ],
        key=lambda item: (item[1].bounding_box.y, item[1].bounding_box.x),
    )
    used_block_indexes: set[int] = set()

    for candidate in candidates:
        current_text = _normalize_table_cell_text(str(candidate.get("text") or ""))
        if not _candidate_needs_page_text_fallback(current_text):
            continue
        cell_box = candidate.get("bounding_box")
        if cell_box is None:
            continue
        hits: list[TextBlock] = []
        for block_index, block in sorted_blocks:
            if block_index in used_block_indexes:
                continue
            if _bbox_center_inside(block.bounding_box, cell_box):
                hits.append(block)
        if not hits:
            continue
        fallback_text = _compose_cell_text_from_hits(
            [
                {"text": block.text, "bounding_box": block.bounding_box}
                for block in hits
                if block.bounding_box is not None
            ]
        )
        if not _should_replace_cell_text_with_page_fallback(current_text, fallback_text):
            continue
        for block_index, block in sorted_blocks:
            if block in hits:
                used_block_indexes.add(block_index)
        candidate["text"] = fallback_text
        confidences = [block.confidence for block in hits if block.confidence is not None]
        if confidences:
            candidate["confidence"] = float(mean(confidences))
        meta = dict(candidate.get("meta") or {})
        meta["page_text_fallback"] = True
        meta["page_text_block_count"] = len(hits)
        candidate["meta"] = meta


def _candidate_needs_page_text_fallback(text: str) -> bool:
    normalized = _normalize_table_cell_text_without_date(text)
    if not normalized:
        return True
    if len(normalized) <= 1:
        return True
    if re.search(r"[\u4e00-\u9fff][A-Za-z]$", normalized):
        return True
    return bool(_looks_like_date_fragment(normalized) and _extract_context_date(normalized) is None)


def _should_replace_cell_text_with_page_fallback(current_text: str, fallback_text: str) -> bool:
    if not fallback_text:
        return False
    current = _normalize_table_cell_text(current_text)
    fallback = _normalize_table_cell_text(fallback_text)
    if not fallback or ("<" in fallback and ">" in fallback):
        return False
    if not re.search(r"[\u4e00-\u9fff]", fallback) and not _extract_context_date(fallback):
        digit_count = sum(1 for ch in fallback if ch.isdigit())
        if digit_count < 8:
            return False
    if not current:
        return True
    if len(_normalize_table_cell_text_without_date(current)) <= 1 and len(fallback) > len(current):
        return True
    current_date = _extract_context_date(current)
    fallback_date = _extract_context_date(fallback)
    if fallback_date and fallback_date != current_date:
        return True
    if re.search(r"[\u4e00-\u9fff][A-Za-z]$", current) and len(fallback) <= len(current):
        return True
    return False


def _fill_cell_text_from_html(
    candidates: list[JsonDict],
    html: str,
    *,
    context_text: str | None = None,
) -> None:
    rows = _html_table_rows(html)
    _repair_contextual_table_cells(rows, context_text=context_text)
    cell_texts = [cell for row in rows for cell in row]
    if not cell_texts:
        return
    for index, candidate in enumerate(candidates):
        if index >= len(cell_texts):
            break
        text = cell_texts[index]
        if not text or str(candidate.get("text") or "").strip():
            continue
        candidate["text"] = text
        meta = dict(candidate.get("meta") or {})
        meta.setdefault("text_source", "mineru_table_html")
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


def _header_row_is_bottom_heavy(candidates: Sequence[Mapping[str, Any]]) -> bool:
    rows = _candidate_text_rows(candidates)
    non_empty_rows = [row for row in rows if any(cell for cell in row)]
    if len(non_empty_rows) < 2:
        return False
    first_score = _table_header_row_score(non_empty_rows[0])
    last_score = _table_header_row_score(non_empty_rows[-1])
    return last_score >= first_score + 2


def _candidate_text_rows(candidates: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    row_map: dict[int, dict[int, str]] = defaultdict(dict)
    for candidate in candidates:
        row_index = candidate.get("row_index")
        col_index = candidate.get("col_index")
        if row_index is None or col_index is None:
            continue
        row_map[int(row_index)][int(col_index)] = _normalize_table_cell_text(
            str(candidate.get("text") or "")
        )
    if not row_map:
        return []
    rows: list[list[str]] = []
    for row_index in sorted(row_map):
        cols = row_map[row_index]
        width = max(cols) + 1 if cols else 0
        row = [cols.get(col_index, "") for col_index in range(width)]
        rows.append(row)
    return rows


def _active_candidate_text_bounds(candidates: Sequence[Mapping[str, Any]]) -> BoundingBox | None:
    return _union_bounding_boxes(
        [
            box
            for candidate in candidates
            if str(candidate.get("text") or "").strip()
            for box in [candidate.get("bounding_box")]
            if isinstance(box, BoundingBox)
        ]
    )


def _table_header_row_score(row: Sequence[str]) -> int:
    header_keywords = (
        "姓名",
        "身份",
        "证号",
        "身份证",
        "角色",
        "时间",
        "人像",
        "签字",
        "方式",
        "结果",
    )
    score = 0
    for cell in row:
        text = _normalize_table_cell_text_without_date(cell)
        if not text:
            continue
        score += sum(1 for keyword in header_keywords if keyword in text)
    return score


def _reset_candidate_fragment_text(candidates: Sequence[JsonDict]) -> None:
    for candidate in candidates:
        candidate["text"] = ""
        candidate["confidence"] = None
        meta = dict(candidate.get("meta") or {})
        for key in (
            "text_source",
            "ocr_fragment_count",
            "ocr_fragment_indexes",
            "ocr_fragment_bbox_source",
            "ocr_fragment_box_transform",
        ):
            meta.pop(key, None)
        candidate["meta"] = meta


def _sort_candidates_row_major(candidates: list[JsonDict]) -> None:
    candidates.sort(
        key=lambda item: (
            int(item.get("row_index") or 0),
            int(item.get("col_index") or 0),
        )
    )


def _merge_wrapped_cell_fragments_from_html(candidates: list[JsonDict], html: str | None) -> None:
    if not html or len(candidates) < 2:
        _normalize_candidate_texts(candidates)
        return

    _normalize_candidate_texts(candidates)
    existing_texts = {_normalize_table_cell_text(str(candidate.get("text") or "")) for candidate in candidates}
    targets = [
        text
        for text in _html_cell_texts(html, deduplicate=True)
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


def _html_cell_texts(html: str, *, deduplicate: bool = False) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for row in _html_table_rows(html):
        for text in row:
            if not text:
                continue
            if deduplicate and text in seen:
                continue
            seen.add(text)
            texts.append(text)
    return texts


def _html_table_rows(html: str) -> list[list[str]]:
    parser = _HtmlTableRowsParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return []

    rows: list[list[str]] = []
    for row in parser.rows:
        rows.append([_normalize_table_cell_text(text) for text in row])
    return rows


class _HtmlTableRowsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "tr":
            self._current_row = []
        elif normalized in {"td", "th"}:
            if self._current_row is None:
                self._current_row = []
            self._current_cell = []
            self._current = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._current_cell is not None:
            text = " ".join(self._current).strip()
            if self._current_row is None:
                self._current_row = []
            self._current_row.append(text)
            self._current_cell = None
            self._current = []
        elif normalized == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def _normalize_table_cell_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", normalized)
    return _extract_context_date(normalized) or normalized


def _repair_contextual_table_cells(rows: list[list[str]], *, context_text: str | None) -> None:
    if len(rows) < 2:
        return
    context_date = _extract_context_date(context_text or "")
    if not context_date:
        return
    headers = rows[0]
    time_columns = [
        index
        for index, header in enumerate(headers)
        if "\u8ba4\u8bc1\u65f6\u95f4" in header or "\u5b9e\u540d\u8ba4\u8bc1\u65f6\u95f4" in header
    ]
    for col_index in time_columns:
        values = [row[col_index] for row in rows[1:] if col_index < len(row)]
        if not any(_looks_like_date_fragment(value) for value in values):
            continue
        for row in rows[1:]:
            if col_index >= len(row):
                continue
            cell_date = _extract_context_date(row[col_index])
            if cell_date:
                row[col_index] = cell_date
            elif not row[col_index] or _looks_like_date_fragment(row[col_index]):
                row[col_index] = context_date


def _looks_like_date_fragment(value: str) -> bool:
    text = _normalize_table_cell_text_without_date(value)
    if not text:
        return False
    if _extract_context_date(text):
        return True
    if any(token in text for token in ("\u5e74", "\u6708", "\u65e5")):
        return True
    return bool(re.fullmatch(r"[\dHh/\-. ]{1,10}", text))


def _normalize_table_cell_text_without_date(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
    normalized = re.sub(r"(?<=\d)\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", normalized)
    return normalized


def _extract_context_date(text: str) -> str | None:
    value = str(text or "")
    match = re.search(
        r"((?:19|20)\d{2})\s*[\u5e74/-]\s*(\d{1,2})\s*[\u6708/-]\s*(\d{1,2})\s*\u65e5?",
        value,
    )
    if match:
        return f"{match.group(1)}\u5e74{int(match.group(2))}\u6708{int(match.group(3))}\u65e5"

    serial_match = re.search(
        r"\u4e1a\u52a1\u6d41\u6c34\u53f7[:\uff1a]?\S*?((?:19|20)\d{2})(\d{2})(\d{2})",
        value,
    )
    if serial_match:
        return (
            f"{serial_match.group(1)}\u5e74"
            f"{int(serial_match.group(2))}\u6708{int(serial_match.group(3))}\u65e5"
        )
    exact_match = re.fullmatch(r"\s*((?:19|20)\d{2})(\d{2})(\d{2})\s*", value)
    if exact_match:
        return (
            f"{exact_match.group(1)}\u5e74"
            f"{int(exact_match.group(2))}\u6708{int(exact_match.group(3))}\u65e5"
        )
    return None


def _join_text_block_content(text_blocks: Sequence[TextBlock]) -> str:
    return "\n".join(block.text for block in text_blocks if block.text)


def _horizontal_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    x0 = max(left.x, right.x)
    x1 = min(left.x + left.w, right.x + right.w)
    if x1 <= x0:
        return 0.0
    return float((x1 - x0) / max(1, min(left.w, right.w)))


def _bbox_intersection_ratio(left: BoundingBox, right: BoundingBox) -> float:
    x0 = max(left.x, right.x)
    x1 = min(left.x + left.w, right.x + right.w)
    y0 = max(left.y, right.y)
    y1 = min(left.y + left.h, right.y + right.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap_area = float((x1 - x0) * (y1 - y0))
    left_area = max(1.0, float(left.w * left.h))
    return overlap_area / left_area


def _cell_fragment_sort_key(box: BoundingBox) -> tuple[int, int, int, int]:
    return (
        int(round(box.y / 4.0)),
        int(round(box.x / 4.0)),
        int(box.y),
        int(box.x),
    )


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


def _paddle_input_uri(file_uri: str, options: Mapping[str, Any]) -> str:
    source_input_uri = (
        _coerce_optional_str(options.get("source_input_uri"))
        or _coerce_optional_str(options.get("original_input_uri"))
        or _coerce_optional_str(options.get("source_file_uri"))
    )
    if not source_input_uri:
        return file_uri
    suffix = Path(source_input_uri).suffix.lower()
    if suffix in {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return source_input_uri
    return file_uri


def _has_seal_hints(
    *,
    options: Mapping[str, Any],
    table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
    page_text_blocks_by_page: Mapping[int, Sequence[TextBlock]],
) -> bool:
    forced_mode = _coerce_optional_str(options.get("paddle_force_mode"))
    if forced_mode and forced_mode.strip().lower() in {"vl", "paddleocr_vl", "paddleocrvl"}:
        return True
    if _coerce_bool(options.get("paddle_has_seal")):
        return True

    for page_candidates in table_candidates_by_page.values():
        for candidate in page_candidates:
            if _mapping_has_seal_hint(candidate):
                return True

    for blocks in page_text_blocks_by_page.values():
        for block in blocks:
            if _value_has_seal_hint(block.block_type):
                return True
            if _mapping_has_seal_hint(block.meta):
                return True
            if _coerce_bool(options.get("paddle_seal_text_hint_enable")) and _value_has_seal_hint(block.text):
                return True
    return False


def _should_fallback_ppstructurev3_result(
    result: PaddleTableStructureResult,
    *,
    table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
    options: Mapping[str, Any],
) -> bool:
    if not _coerce_bool(options.get("paddle_ppstructure_quality_fallback"), True):
        return False
    expected_cell_count = _candidate_html_cell_count(table_candidates_by_page)
    if expected_cell_count <= 0:
        return False
    actual_cell_count = sum(
        len(table.cells)
        for tables in result.tables_by_page.values()
        for table in tables
    )
    if actual_cell_count <= 0:
        return True
    coverage = actual_cell_count / expected_cell_count
    min_coverage = _coerce_float(options.get("paddle_ppstructure_min_cell_coverage"))
    if min_coverage is None:
        min_coverage = 0.95
    return coverage < min_coverage


def _candidate_html_cell_count(
    table_candidates_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
) -> int:
    count = 0
    for page_candidates in table_candidates_by_page.values():
        for candidate in page_candidates:
            html = _coerce_optional_str(candidate.get("html")) or _coerce_optional_str(
                candidate.get("table_body")
            )
            if not html:
                continue
            count += sum(len(row) for row in _html_table_rows(html))
    return count


def _mapping_has_seal_hint(value: Mapping[str, Any]) -> bool:
    for key, item in value.items():
        if _value_has_seal_hint(key) or _value_has_seal_hint(item):
            return True
        if isinstance(item, Mapping) and _mapping_has_seal_hint(item):
            return True
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                if isinstance(child, Mapping) and _mapping_has_seal_hint(child):
                    return True
                if _value_has_seal_hint(child):
                    return True
    return False


def _value_has_seal_hint(value: Any) -> bool:
    if value is None or isinstance(value, (int, float, bool)):
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(token in text for token in ("seal", "stamp", "印章", "公章", "盖章"))


def _full_page_bbox(payload: Mapping[str, Any]) -> list[float] | None:
    width = _coerce_float(payload.get("width"))
    height = _coerce_float(payload.get("height"))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return [0.0, 0.0, width, height]


def _coerce_bbox(
    raw_box: Any,
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> BoundingBox | None:
    if hasattr(raw_box, "tolist"):
        raw_box = raw_box.tolist()
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

    while len(raw_box) == 1 and isinstance(raw_box[0], Sequence) and not isinstance(raw_box[0], (str, bytes)):
        raw_box = raw_box[0]

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
