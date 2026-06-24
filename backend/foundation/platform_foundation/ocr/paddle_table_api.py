from __future__ import annotations

import os
from pathlib import Path
import threading
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
    output_dir: str
    target_page_sizes: dict[int, dict[str, Any] | None] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    table_candidates_by_page: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)
    page_text_blocks_by_page: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)


class PaddleTableWarmupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modes: list[str] = Field(default_factory=lambda: ["table_structure"])
    options: dict[str, Any] = Field(default_factory=dict)


class ApiResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


def _coerce_optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    parsed = int(stripped)
    if parsed <= 0:
        raise ValueError("PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS must be positive when set")
    return parsed


app = FastAPI(title="Paddle Table API", version="0.1.0")
_SERVICE = PaddleTableStructureService()
_STARTED_AT = time.time()
_MAX_CONCURRENT_EXTRACTS = _coerce_optional_positive_int(
    os.environ.get("PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS")
)
_EXTRACT_SEMAPHORE = (
    threading.BoundedSemaphore(_MAX_CONCURRENT_EXTRACTS)
    if _MAX_CONCURRENT_EXTRACTS is not None
    else None
)
_ACTIVE_EXTRACTS = 0
_ACTIVE_EXTRACTS_LOCK = threading.Lock()


@app.on_event("startup")
def preload_models() -> None:
    if not _coerce_env_bool("PADDLE_TABLE_API_PRELOAD", default=True):
        return
    modes = [
        item.strip()
        for item in os.environ.get("PADDLE_TABLE_API_PRELOAD_MODES", "table_structure").split(",")
        if item.strip()
    ]
    warmup_paddle_table_models(modes=modes)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "paddle-table-api",
        "uptime_seconds": round(time.time() - _STARTED_AT, 3),
        "max_concurrent_extracts": _MAX_CONCURRENT_EXTRACTS,
        "active_extracts": _ACTIVE_EXTRACTS,
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
        with _extract_slot():
            result = _SERVICE.extract_tables(
                file_uri=request.file_uri,
                output_dir=Path(request.output_dir),
                target_page_sizes=_decode_target_page_sizes(request.target_page_sizes),
                options=request.options,
                table_candidates_by_page=request.table_candidates_by_page,
                page_text_blocks_by_page=_decode_text_blocks(request.page_text_blocks_by_page),
            )
    except PaddleTableStructureError as exc:
        return ApiResponse(
            success=False,
            data=None,
            error={"code": "PADDLE_TABLE_EXTRACT_FAILED", "message": str(exc)},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApiResponse(success=True, data=paddle_table_result_to_payload(result), error=None)


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


def _coerce_env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class _extract_slot:
    def __enter__(self) -> None:
        global _ACTIVE_EXTRACTS
        if _EXTRACT_SEMAPHORE is not None:
            _EXTRACT_SEMAPHORE.acquire()
        with _ACTIVE_EXTRACTS_LOCK:
            _ACTIVE_EXTRACTS += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        global _ACTIVE_EXTRACTS
        with _ACTIVE_EXTRACTS_LOCK:
            _ACTIVE_EXTRACTS -= 1
        if _EXTRACT_SEMAPHORE is not None:
            _EXTRACT_SEMAPHORE.release()
