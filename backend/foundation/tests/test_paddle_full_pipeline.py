from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from platform_foundation.inference import paddle_document
from platform_foundation.contracts import ParsedPdf
from platform_foundation.ocr import pure_mineru
from platform_foundation.ocr.pure_mineru import PaddleDocumentAPIError
from platform_foundation.ocr.pure_mineru import MinerUPdfPage, MinerUPdfResult, extract_pdf


def _fake_paddle_result(path: str | Path) -> MinerUPdfResult:
    resolved = Path(path)
    return MinerUPdfResult(
        source_pdf=str(resolved),
        source_file_name=resolved.name,
        page_count=1,
        coord_space="image_pixels",
        parsed_pdf=ParsedPdf(pdf_path=str(resolved), total_pages=1, pages=[]),
        pages=[MinerUPdfPage(page_index=0, text="paddle")],
    )


def test_extract_pdf_routes_paddle_to_local_full_paddle_service(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.delenv("PADDLE_TABLE_API_URL", raising=False)
    monkeypatch.setattr(
        pure_mineru,
        "extract_pdf_with_paddle",
        lambda path, options: _fake_paddle_result(path),
    )

    result = extract_pdf(pdf_path, table_engine="paddle")

    assert result.pages[0].text == "paddle"
    assert result.coord_space == "image_pixels"


def test_extract_pdf_routes_paddle_to_document_api_when_configured(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setenv("PADDLE_TABLE_API_URL", "http://127.0.0.1:8200")
    monkeypatch.setattr(
        pure_mineru,
        "_extract_pdf_via_paddle_api",
        lambda path, options, api_url: _fake_paddle_result(path),
    )

    result = extract_pdf(pdf_path, table_engine="paddle")

    assert result.pages[0].text == "paddle"


def test_build_mineru_options_defaults_paddle_vl_version() -> None:
    options = pure_mineru._build_mineru_options(
        output_dir=None,
        api_url=None,
        timeout_seconds=10.0,
        parse_method="auto",
        backend="pipeline",
        lang="ch",
        extra_args=None,
        table_engine="paddle",
        paddle_table_mode="ppstructurev3",
        paddle_device=None,
        mineru_options=None,
    )

    assert options["table_engine"] == "paddle"
    assert options["paddle_vl_version"] == "1.6"


def test_build_vl_pipeline_sets_requested_pipeline_version(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(paddle_document, "_prepare_paddle_runtime", lambda: None)
    monkeypatch.setattr(paddle_document, "_resolve_paddle_device", lambda options: "gpu:2")
    monkeypatch.setattr(
        paddle_document,
        "_cached_document_object",
        lambda **kwargs: kwargs["factory"](kwargs["init_kwargs"]),
    )
    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCRVL = FakePaddleOCRVL
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    paddle_document._build_vl_pipeline({"paddle_vl_version": "1.6"})

    assert captured["pipeline_version"] == "v1"
    assert captured["device"] == "gpu:2"


def test_predict_pages_with_vl_raises_when_predict_fails(monkeypatch, tmp_path: Path) -> None:
    class FakePipeline:
        def predict(self, *, input: str, **kwargs):
            raise RuntimeError("boom")

    image_path = tmp_path / "demo.jpg"
    image_path.write_bytes(b"demo")
    monkeypatch.setattr(paddle_document, "_build_vl_pipeline", lambda options: FakePipeline())
    monkeypatch.setattr(paddle_document, "_resolve_paddle_device", lambda options: "gpu:3")
    monkeypatch.setattr(
        paddle_document,
        "paddle_document_cache_info",
        lambda: {"runtime": {"resolved_device": "gpu:3", "cuda_device_count": 1}},
    )

    with pytest.raises(paddle_document.PaddleDocumentVLError) as exc_info:
        paddle_document._predict_pages_with_vl(image_path, {})

    assert exc_info.value.code == "PADDLE_VL_REQUIRED_FAILED"
    assert exc_info.value.detail["stage"] == "vl"
    assert exc_info.value.detail["reason"] == "predict_failed"
    assert exc_info.value.detail["resolved_device"] == "gpu:3"
    assert "gpu:3" in str(exc_info.value)


def test_predict_pages_with_vl_raises_when_result_pages_empty(monkeypatch, tmp_path: Path) -> None:
    class FakePipeline:
        def predict(self, *, input: str, **kwargs):
            return []

    image_path = tmp_path / "demo.jpg"
    image_path.write_bytes(b"demo")
    monkeypatch.setattr(paddle_document, "_build_vl_pipeline", lambda options: FakePipeline())
    monkeypatch.setattr(paddle_document, "_resolve_paddle_device", lambda options: None)
    monkeypatch.setattr(
        paddle_document,
        "paddle_document_cache_info",
        lambda: {"runtime": {"resolved_device": None, "cuda_device_count": 0}},
    )

    with pytest.raises(paddle_document.PaddleDocumentVLError) as exc_info:
        paddle_document._predict_pages_with_vl(image_path, {"paddle_vl_version": "1.6"})

    assert exc_info.value.detail["reason"] == "empty_pages"
    assert exc_info.value.detail["pipeline_version"] == "v1"
    assert exc_info.value.detail["runtime"]["cuda_device_count"] == 0


def test_extract_pdf_via_paddle_api_preserves_structured_error(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"success": false, "error": {"code": "PADDLE_VL_REQUIRED_FAILED", '
                b'"message": "vl failed", "detail": {"stage": "vl", "reason": "build_failed"}}}'
            )

    monkeypatch.setattr(pure_mineru.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(PaddleDocumentAPIError) as exc_info:
        pure_mineru._extract_pdf_via_paddle_api(
            pdf_path,
            options={"timeout_seconds": 10},
            api_url="http://127.0.0.1:8200",
        )

    assert exc_info.value.code == "PADDLE_VL_REQUIRED_FAILED"
    assert exc_info.value.detail["stage"] == "vl"
