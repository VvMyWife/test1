from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from platform_foundation.ocr import paddle_table_api  # noqa: E402
from platform_foundation.ocr.pure_mineru import MinerUPdfPage, MinerUPdfResult  # noqa: E402
from platform_foundation.contracts import ParsedPdf  # noqa: E402
from platform_foundation.inference.paddle_document import PaddleDocumentVLError  # noqa: E402


def test_paddle_table_api_health_has_no_server_side_extract_limit() -> None:
    payload = paddle_table_api.health()

    assert payload["status"] == "healthy"
    assert "max_concurrent_extracts" not in payload
    assert "active_extracts" not in payload
    assert "runtime" in payload
    assert "document_cache" in payload


def test_paddle_document_extract_endpoint_returns_success_envelope(monkeypatch) -> None:
    monkeypatch.setattr(
        paddle_table_api,
        "extract_pdf_with_paddle",
        lambda file_uri, options: MinerUPdfResult(
            source_pdf=file_uri,
            source_file_name="demo.pdf",
            page_count=1,
            coord_space="image_pixels",
            artifacts=[],
            parsed_pdf=ParsedPdf(pdf_path=file_uri, total_pages=1, pages=[]),
            pages=[MinerUPdfPage(page_index=0, text="ok")],
        ),
    )

    request = paddle_table_api.PaddleDocumentExtractRequest(
        file_uri="/tmp/demo.pdf",
        output_dir="/tmp/out",
        options={},
    )

    payload = paddle_table_api.extract_document(request)

    assert payload.success is True
    assert payload.data["coord_space"] == "image_pixels"


def test_preload_models_drops_vl_from_startup_modes(monkeypatch) -> None:
    called: dict[str, object] = {}

    monkeypatch.setenv("PADDLE_TABLE_API_PRELOAD", "true")
    monkeypatch.setenv("PADDLE_TABLE_API_PRELOAD_MODES", "layout,ocr,table,vl,vl1.5,vl1.6")
    monkeypatch.setattr(
        paddle_table_api,
        "warmup_paddle_document_models",
        lambda **kwargs: called.update(kwargs),
    )

    paddle_table_api.preload_models()

    assert called["modes"] == ["layout", "ocr", "table"]


def test_paddle_document_extract_endpoint_returns_structured_vl_error(monkeypatch) -> None:
    monkeypatch.setattr(
        paddle_table_api,
        "extract_pdf_with_paddle",
        lambda file_uri, options: (_ for _ in ()).throw(
            PaddleDocumentVLError(
                "vl failed",
                detail={"stage": "vl", "reason": "empty_pages", "resolved_device": "gpu:1"},
            )
        ),
    )

    payload = paddle_table_api.extract_document(
        paddle_table_api.PaddleDocumentExtractRequest(
            file_uri="/tmp/demo.pdf",
            output_dir="/tmp/out",
            options={"enable_paddle_vl": True},
        )
    )

    assert payload.success is False
    assert payload.error["code"] == "PADDLE_VL_REQUIRED_FAILED"
    assert payload.error["detail"]["stage"] == "vl"
