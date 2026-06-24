from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..deps import get_mineru_layout_service
from ..responses import success_response
from ...config import load_settings_from_env
from ...schemas.mineru_layout import (
    MinerULayoutBatchDocumentInput,
    MinerULayoutBatchRequest,
    MinerULayoutPathRequest,
)
from ...services.mineru_layout_service import MinerULayoutService, PlatformApiError

router = APIRouter(prefix="/api/v1/operators/mineru", tags=["mineru"])


@router.post("/layout-extract")
def layout_extract(
    service: Annotated[MinerULayoutService, Depends(get_mineru_layout_service)],
    files: Annotated[list[UploadFile], File(...)],
    output_dir: Annotated[str | None, Form()] = None,
    timeout_seconds: Annotated[float | None, Form()] = None,
    mineru_options_json: Annotated[str | None, Form()] = None,
):
    if not files:
        raise PlatformApiError(
            code="EMPTY_UPLOAD",
            message="At least one file must be uploaded",
            status_code=400,
        )

    mineru_options = _parse_json_dict_field(
        mineru_options_json,
        field_name="mineru_options_json",
    )

    settings = load_settings_from_env()
    output_root = _resolve_output_dir(output_dir, workspace_root=settings.workspace_root)
    upload_root = output_root / "_uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    documents: list[MinerULayoutBatchDocumentInput] = []
    for index, upload in enumerate(files):
        target_name = _build_target_filename(index=index, upload=upload)
        target_path = upload_root / target_name
        with target_path.open("wb") as handle:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, handle)

        documents.append(
            MinerULayoutBatchDocumentInput(
                file_uri=str(target_path),
                source_file_name=upload.filename or target_name,
            )
        )

    result = service.extract_layout_batch(
        MinerULayoutBatchRequest(
            documents=documents,
            output_dir=str(output_root),
            mineru_options=mineru_options,
            timeout_seconds=timeout_seconds,
        )
    )

    return success_response(result.model_dump(mode="json"))


@router.post("/layout-extract-path")
def layout_extract_path(
    request: MinerULayoutPathRequest,
    service: Annotated[MinerULayoutService, Depends(get_mineru_layout_service)],
):
    result = service.extract_layout_path(request)
    return success_response(result.model_dump(mode="json"))


def _parse_json_dict_field(raw_value: str | None, *, field_name: str) -> dict:
    if raw_value is None or raw_value.strip() == "":
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise PlatformApiError(
            code="INVALID_JSON_FIELD",
            message=f"{field_name} must be a valid JSON object string",
            status_code=400,
        ) from exc

    if not isinstance(parsed, dict):
        raise PlatformApiError(
            code="INVALID_JSON_FIELD",
            message=f"{field_name} must be a JSON object",
            status_code=400,
        )
    return parsed


def _build_target_filename(*, index: int, upload: UploadFile) -> str:
    original_name = Path(upload.filename or f"upload-{index + 1}.pdf").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        original_name = f"{Path(original_name).stem or f'upload-{index + 1}'}.pdf"
    return f"{index + 1:03d}-{original_name}"


def _resolve_output_dir(raw_output_dir: str | None, *, workspace_root: str) -> Path:
    if raw_output_dir is not None and raw_output_dir.strip():
        return Path(raw_output_dir).expanduser().resolve()
    return (Path(workspace_root).expanduser().resolve() / "output").resolve()
