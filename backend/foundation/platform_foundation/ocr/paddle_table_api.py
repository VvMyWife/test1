from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..contracts import ImageSize, TextBlock
from .pure_mineru import extract_pdf_with_paddle
from ..inference.paddle_document import paddle_document_cache_info, warmup_paddle_document_models
from ..inference.paddle_table import (
    PaddleTableStructureError,
    PaddleTableStructureService,
    paddle_table_cache_info,
    paddle_runtime_info,
    paddle_table_result_to_payload,
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

    modes: list[str] = Field(default_factory=lambda: ["layout", "ocr", "table"])
    options: dict[str, Any] = Field(default_factory=dict)


class PaddleDocumentExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_uri: str
    output_dir: str | None = None
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
        for item in os.environ.get("PADDLE_TABLE_API_PRELOAD_MODES", "layout,ocr,table").split(",")
        if item.strip()
    ]
    modes = [mode for mode in modes if mode.lower() not in {"vl", "vl1.5", "vl1.6"}]
    warmup_paddle_document_models(modes=modes)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "paddle-table-api",
        "uptime_seconds": round(time.time() - _STARTED_AT, 3),
        "cache": paddle_table_cache_info(),
        "document_cache": paddle_document_cache_info(),
        "runtime": paddle_runtime_info(),
    }


@app.post("/warmup", response_model=ApiResponse)
def warmup(request: PaddleTableWarmupRequest) -> ApiResponse:
    try:
        return ApiResponse(
            success=True,
            data=warmup_paddle_document_models(options=request.options, modes=request.modes),
            error=None,
        )
    except PaddleTableStructureError as exc:
        return ApiResponse(
            success=False,
            data=None,
            error={
                "code": getattr(exc, "code", "PADDLE_TABLE_WARMUP_FAILED"),
                "message": str(exc),
                "detail": getattr(exc, "detail", None),
            },
        )


@app.post("/api/v1/paddle/table-extract", response_model=ApiResponse)
def extract_tables(request: PaddleTableExtractRequest) -> ApiResponse:
    try:
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
            error={
                "code": getattr(exc, "code", "PADDLE_TABLE_EXTRACT_FAILED"),
                "message": str(exc),
                "detail": getattr(exc, "detail", None),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApiResponse(success=True, data=paddle_table_result_to_payload(result), error=None)


@app.post("/api/v1/paddle/document-extract", response_model=ApiResponse)
def extract_document(request: PaddleDocumentExtractRequest) -> ApiResponse:
    try:
        options = dict(request.options)
        if request.output_dir:
            options.setdefault("output_dir", request.output_dir)
        options.setdefault("table_engine", "paddle")
        result = extract_pdf_with_paddle(
            request.file_uri,
            options=options,
        )
    except PaddleTableStructureError as exc:
        return ApiResponse(
            success=False,
            data=None,
            error={
                "code": getattr(exc, "code", "PADDLE_DOCUMENT_EXTRACT_FAILED"),
                "message": str(exc),
                "detail": getattr(exc, "detail", None),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ApiResponse(success=True, data=result.model_dump(mode="json"), error=None)


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
