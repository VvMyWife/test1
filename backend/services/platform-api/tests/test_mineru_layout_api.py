from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import get_mineru_layout_service
from app.main import create_app
from app.schemas.mineru_layout import MinerUGeneratedJson, MinerULayoutBatchResult
from app.services.mineru_layout_service import PlatformApiError


def _request_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("files", ("doc-1.pdf", b"%PDF-1.4 test-1", "application/pdf")),
        ("files", ("doc-2.pdf", b"%PDF-1.4 test-2", "application/pdf")),
    ]


def test_layout_extract_endpoint_returns_success_envelope_for_multiple_files() -> None:
    app = create_app()

    class _FakeService:
        def extract_layout_batch(self, request):  # noqa: ANN001
            assert len(request.documents) == 2
            assert request.output_dir
            return MinerULayoutBatchResult(
                document_count=2,
                output_dir=request.output_dir,
                documents=[
                    MinerUGeneratedJson(
                        source_pdf=request.documents[0].file_uri,
                        source_file_name=request.documents[0].source_file_name,
                        json_path=f"{request.output_dir}/doc-1.json",
                        artifact_dir=f"{request.output_dir}/doc-1",
                        page_count=1,
                        text_block_count=3,
                        table_block_count=0,
                        elapsed_seconds=0.1,
                    ),
                    MinerUGeneratedJson(
                        source_pdf=request.documents[1].file_uri,
                        source_file_name=request.documents[1].source_file_name,
                        json_path=f"{request.output_dir}/doc-2.json",
                        artifact_dir=f"{request.output_dir}/doc-2",
                        page_count=1,
                        text_block_count=4,
                        table_block_count=0,
                        elapsed_seconds=0.2,
                    ),
                ],
            )

    app.dependency_overrides[get_mineru_layout_service] = lambda: _FakeService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/operators/mineru/layout-extract",
        data={"output_dir": "/tmp/mineru-output"},
        files=_request_files(),
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["document_count"] == 2
    assert body["data"]["documents"][0]["source_file_name"] == "doc-1.pdf"
    assert body["data"]["documents"][1]["json_path"].endswith("/doc-2.json")
    assert "pages" not in body["data"]["documents"][0]


def test_layout_extract_endpoint_returns_wrapped_service_error() -> None:
    app = create_app()

    class _FailingService:
        def extract_layout_batch(self, request):  # noqa: ANN001
            raise PlatformApiError(
                code="MINERU_TIMEOUT",
                message="MinerU timed out while extracting layout",
                status_code=503,
            )

    app.dependency_overrides[get_mineru_layout_service] = lambda: _FailingService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/operators/mineru/layout-extract",
        files=_request_files(),
    )
    body = response.json()

    assert response.status_code == 503
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "MINERU_TIMEOUT"


def test_layout_extract_path_endpoint_returns_generated_json_summary() -> None:
    app = create_app()

    class _FakeService:
        def extract_layout_path(self, request):  # noqa: ANN001
            return MinerUGeneratedJson(
                source_pdf=request.pdf_path,
                source_file_name="local.pdf",
                json_path=f"{request.output_dir}/local.json",
                artifact_dir=f"{request.output_dir}/local",
                page_count=2,
                text_block_count=10,
                table_block_count=1,
                elapsed_seconds=1.5,
            )

    app.dependency_overrides[get_mineru_layout_service] = lambda: _FakeService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/operators/mineru/layout-extract-path",
        json={"pdf_path": "/data/input/local.pdf", "output_dir": "/data/output"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["data"]["json_path"] == "/data/output/local.json"
    assert body["data"]["page_count"] == 2


def test_layout_extract_endpoint_returns_wrapped_validation_error() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/operators/mineru/layout-extract",
    )
    body = response.json()

    assert response.status_code == 422
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_ERROR"
