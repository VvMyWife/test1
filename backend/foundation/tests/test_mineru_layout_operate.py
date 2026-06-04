from __future__ import annotations

import json
from pathlib import Path

from platform_foundation.contracts import (
    ArtifactRef,
    BoundingBox,
    DocumentItem,
    PageItem,
    TableBlock,
    TableCell,
    TextBlock,
)
from platform_foundation.ocr.mineru_layout import operate
from platform_foundation.ocr import pdf_extract as pdf_extract_module
from platform_foundation.ocr.pdf_extract import (
    DEFAULT_OCR_OPERATOR_MAX_INFLIGHT,
    DEFAULT_PADDLE_OPERATOR_MAX_INFLIGHT,
    MinerUPdfFileOperator,
    extract_pdf_dir,
    extract_pdf_file,
)
from platform_foundation.ocr.pure_mineru import dump_pure_mineru_json, extract_pdf


def test_operate_is_importable_foundation_entrypoint() -> None:
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")
    text_block = TextBlock(
        text="hello",
        bounding_box=BoundingBox(x=10, y=20, w=30, h=40),
        block_type="text",
    )
    table_block = TableBlock(
        table_id="p0-t0",
        page_index=0,
        cells=[
            TableCell(
                cell_id="p0-t0-c0",
                text="cell",
                bounding_box=BoundingBox(x=12, y=22, w=8, h=6),
                row_index=0,
                col_index=0,
            )
        ],
    )

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            assert document["meta"]["mineru_options"]["backend"] == "pipeline"
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="hello",
                text_blocks=[text_block],
                table_blocks=[table_block],
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "width": 100,
                    "height": 200,
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    document = DocumentItem(
        archive_id="archive-1",
        archive_owner_user_id="owner-1",
        triggered_by_user_id="user-1",
        doc_id="doc-1",
        file_uri="/tmp/doc.pdf",
    )

    result = operate(
        document,
        trace_id="trace-1",
        run_id="run-1",
        mineru_options={"backend": "pipeline"},
        operator_factory=_FakeOperator,
    )

    assert result.trace_id == "trace-1"
    assert result.run_id == "run-1"
    assert result.page_count == 1
    assert result.artifacts[0].kind == "middle_json"
    assert result.parsed_pdf.pdf_path == "/tmp/doc.pdf"
    assert result.parsed_pdf.pages[0].text_blocks[0].bounding_box.w == 30
    assert result.parsed_pdf.pages[0].table_blocks[0].cells[0].text == "cell"
    assert result.pages[0].text_blocks[0].text == "hello"


def test_extract_pdf_returns_business_field_free_result() -> None:
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="hello",
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    result = extract_pdf(
        "/tmp/demo.pdf",
        api_url="http://127.0.0.1:8000",
        operator_factory=_FakeOperator,
    )

    assert result.source_file_name == "demo.pdf"
    assert result.page_count == 1
    assert result.pages[0].text == "hello"
    dumped = result.model_dump(mode="json")
    assert "archive_id" not in dumped["pages"][0]
    assert "triggered_by_user_id" not in dumped["pages"][0]
    output_payload = dump_pure_mineru_json(result)
    assert '"parsed_pdf"' not in output_payload


def test_extract_pdf_file_persists_json_and_metrics(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input" / "demo.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF-1.4\n")
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="hello",
                table_blocks=[
                    TableBlock(
                        table_id="p0-t0",
                        page_index=0,
                        cells=[TableCell(cell_id="p0-t0-c0", text="cell")],
                    )
                ],
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    result = extract_pdf_file(
        pdf_path,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
    )

    assert result.success is True
    assert result.json_path is not None
    assert Path(result.json_path).exists()
    assert Path(result.artifact_dir).name == "demo"
    assert result.page_count == 1
    assert result.table_block_count == 1
    assert result.table_cell_count == 1
    payload = Path(result.json_path).read_text(encoding="utf-8")
    assert '"parsed_pdf"' not in payload


def test_extract_pdf_file_accepts_image_and_converts_to_pdf(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "input" / "photo.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 24), (255, 255, 255)).save(image_path)
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            assert document["file_uri"].endswith(".converted.pdf")
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="image text",
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    result = extract_pdf_file(
        image_path,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
    )

    assert result.success is True
    assert result.input_type == "image"
    assert result.converted_pdf_path is not None
    assert Path(result.converted_pdf_path).exists()
    assert Path(result.json_path or "").name == "photo.json"
    payload = json.loads(Path(result.json_path or "").read_text(encoding="utf-8"))
    assert payload["source_pdf"] == str(image_path.resolve())
    assert payload["source_file_name"] == "photo.jpg"
    assert any(artifact["kind"] == "converted_pdf" for artifact in payload["artifacts"])


def test_extract_pdf_file_writes_error_json_for_invalid_input(tmp_path: Path) -> None:
    result = extract_pdf_file(
        tmp_path / "missing.pdf",
        output_dir=tmp_path / "output",
    )

    assert result.success is False
    assert result.error_report is not None
    assert Path(result.error_report).exists()
    assert result.error_type == "FileNotFoundError"


def test_extract_pdf_file_can_export_page_screenshots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "input" / "demo.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF-1.4\n")
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            yield PageItem(
                archive_id="archive",
                archive_owner_user_id="owner",
                triggered_by_user_id="user",
                doc_id="doc",
                page_index=0,
                text="hello",
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    def _fake_render(pdf_path, *, output_dir, page_count, dpi):  # noqa: ANN001
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / "page_0001.png"
        screenshot_path.write_bytes(b"png")
        manifest_path = output_dir / "page_manifest.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "page_index": 0,
                    "page_number": 1,
                    "image_path": str(screenshot_path),
                    "source_pdf": str(pdf_path),
                    "dpi": dpi,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return ArtifactRef(
            kind="page_screenshots_manifest",
            uri=str(manifest_path),
            meta={"page_count": page_count, "dpi": dpi},
        )

    monkeypatch.setattr(pdf_extract_module, "_render_pdf_page_screenshots", _fake_render)

    result = extract_pdf_file(
        pdf_path,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
        enable_page_screenshots=True,
        page_screenshot_dpi=96,
    )

    assert result.success is True
    assert result.page_screenshots_manifest is not None
    assert Path(result.page_screenshots_manifest).exists()
    payload = json.loads(Path(result.json_path or "").read_text(encoding="utf-8"))
    assert any(artifact["kind"] == "page_screenshots_manifest" for artifact in payload["artifacts"])


def test_file_operator_does_not_limit_inflight_by_default() -> None:
    operator = MinerUPdfFileOperator()

    ocr_operator = operator._operator_factory_for("ocr")()
    paddle_operator = operator._operator_factory_for("paddle")()

    assert ocr_operator.max_inflight == DEFAULT_OCR_OPERATOR_MAX_INFLIGHT
    assert ocr_operator.recommended_max_concurrency == DEFAULT_OCR_OPERATOR_MAX_INFLIGHT
    assert paddle_operator.max_inflight == DEFAULT_PADDLE_OPERATOR_MAX_INFLIGHT
    assert paddle_operator.recommended_max_concurrency == DEFAULT_PADDLE_OPERATOR_MAX_INFLIGHT


def test_file_operator_enables_mineru_table_predict_for_ocr_mode() -> None:
    operator = MinerUPdfFileOperator()

    assert operator._default_extra_args("ocr") == ["--formula", "false", "--table", "true"]
    assert operator._default_extra_args("paddle") == ["--formula", "false", "--table", "true"]


def test_extract_pdf_dir_writes_batch_reports_and_skips_existing_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf_path = input_dir / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="hello",
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    first = extract_pdf_dir(
        input_dir,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
        concurrency=1,
    )
    first_report = json.loads(Path(first.batch_report_path).read_text(encoding="utf-8"))
    second = extract_pdf_dir(
        input_dir,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
        concurrency=1,
    )

    assert first.success_count == 1
    assert first.failure_count == 0
    assert first.page_count == 1
    assert first.pages_per_second > 0
    assert Path(first.batch_report_path).exists()
    assert Path(first.batch_report_csv_path).exists()
    assert first_report["page_count"] == 1
    assert first_report["pages_per_second"] == first.pages_per_second
    assert second.skipped_count == 1
    assert second.items[0].skipped is True
    assert second.page_count == 1


def test_extract_pdf_dir_scans_images_and_pdfs(tmp_path: Path) -> None:
    from PIL import Image

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "demo.pdf").write_bytes(b"%PDF-1.4\n")
    Image.new("RGB", (24, 24), (255, 255, 255)).save(input_dir / "photo.jpg")
    (input_dir / "ignore.txt").write_text("ignore", encoding="utf-8")
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text=Path(document["file_uri"]).name,
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    report = extract_pdf_dir(
        input_dir,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
        concurrency=1,
        overwrite=True,
    )

    assert report.success_count == 2
    assert report.failure_count == 0
    assert report.pdf_count == 2
    assert {Path(item.json_path or "").name for item in report.items} == {"demo.json", "photo.json"}
    image_item = next(item for item in report.items if item.source_file_name == "photo.jpg")
    assert image_item.input_type == "image"
    assert image_item.converted_pdf_path is not None
