from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..contracts import ImageSize, TextBlock
from ..inference.paddle_table import (
    PaddleTableStructureError,
    PaddleTableStructureService,
    paddle_table_cache_info,
    paddle_table_result_to_payload,
    warmup_paddle_table_models,
)


class PaddleTableExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_uri: str
    file_name: str | None = None
    file_bytes_b64: str | None = None
    output_dir: str
    target_page_sizes: dict[int, dict[str, Any] | None] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    table_candidates_by_page: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)
    page_text_blocks_by_page: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)


class PaddleTableWarmupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modes: list[str] = Field(default_factory=lambda: ["table_structure", "ppstructurev3"])
    options: dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


app = FastAPI(title="Paddle Table API", version="0.1.0")
_SERVICE = PaddleTableStructureService()
_STARTED_AT = time.time()


@app.on_event("startup")
def preload_models() -> None:
    if not _coerce_env_bool("PADDLE_TABLE_API_PRELOAD", default=True):
        return
    modes = [
        item.strip()
        for item in os.environ.get("PADDLE_TABLE_API_PRELOAD_MODES", "table_structure,ppstructurev3").split(",")
        if item.strip()
    ]
    warmup_paddle_table_models(modes=modes)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "paddle-table-api",
        "uptime_seconds": round(time.time() - _STARTED_AT, 3),
        "cache": paddle_table_cache_info(),
    }


@app.post("/warmup", response_model=ApiResponse)
def warmup(request: PaddleTableWarmupRequest) -> ApiResponse:
    try:
        return ApiResponse(
            success=True,
            data=warmup_paddle_table_models(options=request.options, modes=request.modes),
            error=None,
        )
    except PaddleTableStructureError as exc:
        return ApiResponse(
            success=False,
            data=None,
            error={"code": "PADDLE_TABLE_WARMUP_FAILED", "message": str(exc)},
        )


@app.post("/api/v1/paddle/table-extract", response_model=ApiResponse)
def extract_tables(request: PaddleTableExtractRequest) -> ApiResponse:
    try:
        with tempfile.TemporaryDirectory(prefix="paddle-table-api-") as temp_dir:
            file_uri = request.file_uri
            output_dir = Path(request.output_dir)
            table_candidates_by_page = request.table_candidates_by_page
            if request.file_bytes_b64:
                temp_root = Path(temp_dir)
                file_uri = str(_write_b64_file(
                    request.file_bytes_b64,
                    directory=temp_root,
                    file_name=request.file_name or Path(request.file_uri).name or "input.pdf",
                    default_suffix=".pdf",
                ))
                output_dir = temp_root / "output"
                table_candidates_by_page = _materialize_candidate_images(table_candidates_by_page, temp_root)

            result = _SERVICE.extract_tables(
                file_uri=file_uri,
                output_dir=output_dir,
                target_page_sizes=_decode_target_page_sizes(request.target_page_sizes),
                options=request.options,
                table_candidates_by_page=table_candidates_by_page,
                page_text_blocks_by_page=_decode_text_blocks(request.page_text_blocks_by_page),
            )
            response_data = paddle_table_result_to_payload(result)
    except PaddleTableStructureError as exc:
        return ApiResponse(
            success=False,
            data=None,
            error={"code": "PADDLE_TABLE_EXTRACT_FAILED", "message": str(exc)},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApiResponse(success=True, data=response_data, error=None)


def _decode_target_page_sizes(raw: dict[int, dict[str, Any] | None]) -> dict[int, ImageSize | None]:
    return {
        int(page_index): ImageSize.model_validate(payload) if payload is not None else None
        for page_index, payload in raw.items()
    }


def _decode_text_blocks(raw: dict[int, list[dict[str, Any]]]) -> dict[int, list[TextBlock]]:
    return {
        int(page_index): [TextBlock.model_validate(item) for item in items]
        for page_index, items in raw.items()
    }


def _materialize_candidate_images(
    raw: dict[int, list[dict[str, Any]]],
    temp_root: Path,
) -> dict[int, list[dict[str, Any]]]:
    materialized: dict[int, list[dict[str, Any]]] = {}
    image_dir = temp_root / "candidate_images"
    for page_index, candidates in raw.items():
        page_items: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            item = dict(candidate)
            image_bytes_b64 = item.pop("image_bytes_b64", None)
            image_name = item.pop("image_name", None)
            if isinstance(image_bytes_b64, str) and image_bytes_b64:
                item["image_uri"] = str(_write_b64_file(
                    image_bytes_b64,
                    directory=image_dir,
                    file_name=str(image_name or f"page_{page_index}_table_{index}.png"),
                    default_suffix=".png",
                ))
            page_items.append(item)
        materialized[int(page_index)] = page_items
    return materialized


def _write_b64_file(
    value: str,
    *,
    directory: Path,
    file_name: str,
    default_suffix: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_file_name(file_name, default_suffix=default_suffix)
    path = directory / safe_name
    try:
        content = base64.b64decode(value.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise PaddleTableStructureError("Invalid base64 file payload") from exc
    path.write_bytes(content)
    return path


def _safe_file_name(value: str, *, default_suffix: str) -> str:
    name = Path(value).name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name:
        name = f"input{default_suffix}"
    if not Path(name).suffix:
        name = f"{name}{default_suffix}"
    return name


def _coerce_env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
