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


def test_extract_pdf_file_can_export_field_coordinates_and_annotation_pdf(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001
) -> None:
    pdf_path = tmp_path / "input" / "demo.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF-1.4\n")
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    def _fake_annotation(input_pdf_path, output_pdf_path, *, matches, parsed_pdf):  # noqa: ANN001
        del input_pdf_path, parsed_pdf
        output_pdf_path.write_bytes(b"%PDF-1.4\n")
        for match in matches:
            match["pdf_bounding_box"] = {"x": 10.0, "y": 20.0, "w": 30.0, "h": 12.0}
            match["pdf_quad_points"] = [
                {"x": 10.0, "y": 20.0},
                {"x": 40.0, "y": 20.0},
                {"x": 40.0, "y": 32.0},
                {"x": 10.0, "y": 32.0},
            ]

    monkeypatch.setattr(pdf_extract_module, "_write_field_annotation_pdf", _fake_annotation)

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="身份证号 123456",
                table_blocks=[
                    TableBlock(
                        table_id="p0-t0",
                        page_index=0,
                        provider="paddleocr_ppstructurev3",
                        coord_space="mineru_layout",
                        cells=[
                            TableCell(
                                cell_id="p0-t0-c0",
                                text="身份证号",
                                bounding_box=BoundingBox(x=100, y=120, w=80, h=24),
                                row_index=0,
                                col_index=0,
                            )
                        ],
                    )
                ],
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "width": 600,
                    "height": 800,
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    result = extract_pdf_file(
        pdf_path,
        output_dir=tmp_path / "output",
        operator_factory=_FakeOperator,
        table_engine="paddle",
        field_keywords=["身份证号"],
    )

    assert result.success is True
    assert result.field_match_count == 1
    assert result.field_coordinates_path is not None
    assert result.field_annotation_pdf_path is not None
    assert Path(result.field_coordinates_path).exists()
    assert Path(result.field_annotation_pdf_path).exists()
    coordinates = json.loads(Path(result.field_coordinates_path).read_text(encoding="utf-8"))
    assert coordinates["field_keywords"] == ["身份证号"]
    assert coordinates["match_count"] == 1
    assert coordinates["matches"][0]["bounding_box"] == {"x": 100.0, "y": 120.0, "w": 80.0, "h": 24.0}
    assert coordinates["matches"][0]["quad_points"][2] == {"x": 180.0, "y": 144.0}
    payload = json.loads(Path(result.json_path or "").read_text(encoding="utf-8"))
    artifact_kinds = {artifact["kind"] for artifact in payload["artifacts"]}
    assert "field_coordinates_json" in artifact_kinds
    assert "field_annotation_pdf" in artifact_kinds


def test_extract_pdf_file_accepts_image_and_passes_directly(tmp_path: Path) -> None:
    """Images are now passed directly to MinerU — no intermediate PDF conversion."""
    from PIL import Image

    image_path = tmp_path / "input" / "photo.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 24), (255, 255, 255)).save(image_path)
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            # Image is passed directly — file_uri should be the original .jpg
            assert document["file_uri"].endswith(".jpg")
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
    assert result.converted_pdf_path is None  # No longer converted
    assert Path(result.json_path or "").name == "photo.json"
    payload = json.loads(Path(result.json_path or "").read_text(encoding="utf-8"))
    assert payload["source_pdf"] == str(image_path.resolve())
    assert payload["source_file_name"] == "photo.jpg"
    # converted_pdf artifact is no longer emitted for images
    assert not any(artifact["kind"] == "converted_pdf" for artifact in payload["artifacts"])


def test_extract_pdf_file_flattens_mineru_pdf_artifact_dir(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input" / "1.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            output_dir = Path(document["meta"]["mineru_options"]["output_dir"])
            mineru_auto_dir = output_dir / "1" / "auto"
            mineru_auto_dir.mkdir(parents=True)
            middle_json = mineru_auto_dir / "1_middle.json"
            middle_json.write_text("{}", encoding="utf-8")
            artifact = ArtifactRef(kind="middle_json", uri=str(middle_json))
            yield PageItem(
                archive_id=document["archive_id"],
                archive_owner_user_id=document["archive_owner_user_id"],
                triggered_by_user_id=document["triggered_by_user_id"],
                doc_id=document["doc_id"],
                page_index=0,
                text="pdf text",
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

    artifact_dir = Path(result.artifact_dir)
    assert result.success is True
    assert Path(result.json_path or "") == artifact_dir / "1.json"
    assert (artifact_dir / "1_middle.json").exists()
    assert not (artifact_dir / "1").exists()
    assert not (artifact_dir / "auto").exists()
    payload = json.loads(Path(result.json_path or "").read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["uri"] == str(artifact_dir / "1_middle.json")
    assert payload["pages"][0]["layout_ref"]["uri"] == str(artifact_dir / "1_middle.json")


def test_extract_pdf_file_flattens_image_artifact_dir_directly(tmp_path: Path) -> None:
    """Image inputs no longer go through .converted.pdf — files use original stem."""
    from PIL import Image

    image_path = tmp_path / "input" / "10_01.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 24), (255, 255, 255)).save(image_path)

    class _FakeOperator:
        def process(self, ctx, items, path):  # noqa: ANN001
            document = next(items)
            output_dir = Path(document["meta"]["mineru_options"]["output_dir"])
            mineru_stem = Path(document["file_uri"]).stem
            # Image passed directly — stem is the original filename stem
            assert mineru_stem == "10_01"
            mineru_auto_dir = output_dir / mineru_stem / "auto"
            mineru_auto_dir.mkdir(parents=True)
            middle_json = mineru_auto_dir / "10_01_middle.json"
            middle_json.write_text("{}", encoding="utf-8")
            markdown = mineru_auto_dir / "10_01.md"
            markdown.write_text(
                "# Demo\n\n<table><tr><td>姓名</td><td>结果</td></tr></table>\n",
                encoding="utf-8",
            )
            content_list = mineru_auto_dir / "10_01_content_list.json"
            content_list.write_text(
                json.dumps(
                    [
                        {"type": "header", "text": "实名认证信息表", "bbox": [30, 1, 92, 15]},
                        {"type": "header", "text": "页码，1/1", "bbox": [321, -1, 360, 13]},
                        {
                            "type": "footer",
                            "text": "http://example.local/print 2025/5/14",
                            "bbox": [34, 542, 363, 557],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            artifact = ArtifactRef(kind="middle_json", uri=str(middle_json))
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

    artifact_dir = Path(result.artifact_dir)
    assert result.success is True
    assert Path(result.json_path or "") == artifact_dir / "10_01.json"
    assert result.converted_pdf_path is None  # No longer converted
    assert (artifact_dir / "10_01_middle.json").exists()
    markdown = artifact_dir / "10_01.md"
    assert markdown.exists()
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "<table" not in markdown_text
    assert "姓名 | 结果" in markdown_text
    assert "实名认证信息表 | 页码，1/1" in markdown_text
    assert markdown_text.rstrip().endswith("http://example.local/print 2025/5/14")
    assert not (artifact_dir / "10_01").exists()
    assert not (artifact_dir / "auto").exists()
    payload = json.loads(Path(result.json_path or "").read_text(encoding="utf-8"))
    assert payload["artifacts"][0]["uri"] == str(artifact_dir / "10_01_middle.json")


def test_markdown_html_table_normalization_preserves_grid_and_repairs_dates() -> None:
    markdown = (
        "\u4e1a\u52a1\u6d41\u6c34\u53f7:510604Y3202505140001\n\n"
        "<table>"
        "<tr><td>\u59d3\u540d</td><td>\u8eab\u4efd\u8bc1\u53f7\u7801</td>"
        "<td>\u89d2\u8272</td><td>\u5b9e\u540d\u8ba4\u8bc1\u65f6\u95f4</td>"
        "<td>\u7ed3\u679c</td></tr>"
        "<tr><td>\u5411\u4fca\u5b66</td><td>510626199209193753</td>"
        "<td>\u6cd5\u5b9a\u4ee3\u8868\u4eba</td><td></td><td>\u6210\u529f</td></tr>"
        "<tr><td>\u5411\u4fca\u5b66</td><td>510626199209193753</td>"
        "<td>\u80a1\u4e1c</td><td>2\u65e5</td><td>\u6210\u529f</td></tr>"
        "</table>\n"
    )

    normalized = pdf_extract_module._replace_html_tables_with_text(markdown)

    assert "<table" not in normalized
    assert "| --- | --- | --- | --- | --- |" in normalized
    assert normalized.count("\u5411\u4fca\u5b66") == 2
    assert normalized.count("510626199209193753") == 2
    assert normalized.count("2025\u5e745\u670814\u65e5") == 2


def test_markdown_normalization_appends_seal_section_last(tmp_path: Path) -> None:
    markdown_path = tmp_path / "demo.converted.md"
    mineru_seal_text = "\u5fb7\u963b\u5efa\u946b\u5e02\u653f\u8bbe\u65bd\u7ba1\u7406\u6709\u9650\u8d23\u4efb\u516c\u53f8"
    markdown_path.write_text(
        f"\u6b63\u6587\n\n![](images/seal.jpg)  \n{mineru_seal_text}\n",
        encoding="utf-8",
    )
    markdown_path.with_name("demo.converted_content_list.json").write_text(
        json.dumps(
            [
                {
                    "type": "footer",
                    "text": "\u9875\u811a",
                    "bbox": [0, 900, 100, 920],
                    "page_idx": 0,
                },
                {
                    "type": "seal",
                    "text": mineru_seal_text,
                    "img_path": "images/seal.jpg",
                    "bbox": [637, 598, 846, 744],
                    "page_idx": 0,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "paddle_table_structure.json").write_text(
        json.dumps(
            {
                "text_blocks_by_page": {
                    "0": [
                        {
                            "text": "\u7edf\u4e00\u793e\u4f1a\u4fe1\u7528\u4ee3\u7801 91510604MA6ABCDE1X",
                            "block_type": "seal_text",
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pdf_extract_module._normalize_markdown_artifacts(tmp_path)
    pdf_extract_module._normalize_markdown_artifacts(tmp_path)

    normalized = markdown_path.read_text(encoding="utf-8")
    assert normalized.count("\u5370\u7ae0\uff1a") == 1
    assert "![](images/seal.jpg)" not in normalized
    assert normalized.count(mineru_seal_text) == 0
    assert normalized.index("\u9875\u811a") < normalized.index("\u5370\u7ae0\uff1a")
    assert normalized.rstrip().endswith(
        "\u5370\u7ae0\uff1a\n\n"
        "\u7edf\u4e00\u793e\u4f1a\u4fe1\u7528\u4ee3\u7801 91510604MA6ABCDE1X"
    )


def test_markdown_normalization_rebuilds_seal_mode_from_paddle_blocks(tmp_path: Path) -> None:
    markdown_path = tmp_path / "demo.converted.md"
    markdown_path.write_text("\u6b63\u6587\u4e22\u5931\n\n\u76d6\u7387\n", encoding="utf-8")
    (tmp_path / "paddle_table_structure.json").write_text(
        json.dumps(
            {
                "provider": "paddleocr_ppocrv5_paddleocr_vl_seal",
                "mode": "seal_vl_crops_ppocrv5",
                "text_blocks_by_page": {
                    "0": [
                        {
                            "text": "\u5168\u4f53\u6295\u8d44\u4eba\u7b7e\u5b57\uff08\u76d6\u7ae0\uff09\u7f57\u5efa",
                            "block_type": "text",
                            "bounding_box": {"x": 100, "y": 400, "w": 300, "h": 20},
                        },
                        {
                            "text": "\u5fb7\u9633\u5efa\u946b\u5e02\u653f\u8bbe\u65bd\u7ba1\u7406\u6709\u9650\u8d23\u4efb\u516c\u53f8",
                            "block_type": "seal_text",
                            "bounding_box": {"x": 637, "y": 598, "w": 209, "h": 146},
                        },
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pdf_extract_module._normalize_markdown_artifacts(tmp_path)

    normalized = markdown_path.read_text(encoding="utf-8")
    assert "\u6b63\u6587\u4e22\u5931" not in normalized
    assert "\u76d6\u7387" not in normalized
    assert "\u5168\u4f53\u6295\u8d44\u4eba\u7b7e\u5b57\uff08\u76d6\u7ae0\uff09\u7f57\u5efa" in normalized
    assert normalized.rstrip().endswith(
        "\u5370\u7ae0\uff1a\n\n"
        "\u5fb7\u9633\u5efa\u946b\u5e02\u653f\u8bbe\u65bd\u7ba1\u7406\u6709\u9650\u8d23\u4efb\u516c\u53f8"
    )


def test_markdown_normalization_rebuilds_markdown_table_from_paddle_cells(tmp_path: Path) -> None:
    markdown_path = tmp_path / "0077.md"
    markdown_path.write_text(
        (
            "# \u80a1\u4e1c\u5b9e\u540d\u8ba4\u8bc1\u4fe1\u606f\n\n"
            "| \u59d3\u540d | \u8eab\u4efd\u8bc1\u53f7\u7801 | \u89d2\u8272 | \u5b9e\u540d\u8ba4\u8bc1\u65f6\u95f4 |\n"
            "| --- | --- | --- | --- |\n"
            "| \u9648\u827a\u7476 | 510603199605255966 | \u4eba\u8868\uff09 |  |\n"
            "| \u59da\u6d2a\u83ca | 510602197307030982 | \u7231 |  |\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "paddle_table_structure.json").write_text(
        json.dumps(
            {
                "provider": "paddleocr_ppstructurev3",
                "mode": "ppstructurev3",
                "tables_by_page": {
                    "0": [
                        {
                            "bounding_box": {"x": 10, "y": 20, "w": 300, "h": 200},
                            "cells": [
                                {"text": "\u59d3\u540d", "row_index": 0, "col_index": 0, "bounding_box": {"x": 10, "y": 20}},
                                {"text": "\u8eab\u4efd\u8bc1\u53f7\u7801", "row_index": 0, "col_index": 1, "bounding_box": {"x": 40, "y": 20}},
                                {"text": "\u89d2\u8272", "row_index": 0, "col_index": 2, "bounding_box": {"x": 100, "y": 20}},
                                {"text": "\u5b9e\u540d\u8ba4\u8bc1\u65f6\u95f4", "row_index": 0, "col_index": 2, "bounding_box": {"x": 140, "y": 20}},
                                {"text": "\u9648\u827a\u7476", "row_index": 1, "col_index": 0, "bounding_box": {"x": 10, "y": 60}},
                                {"text": "510603199605255966", "row_index": 1, "col_index": 1, "bounding_box": {"x": 40, "y": 60}},
                                {"text": "\u6267\u884c\u4e8b\u52a1\u5408\u4f19\u4eba\uff08\u59d4\u6d3e\u4ee3\u8868\uff09", "row_index": 1, "col_index": 2, "bounding_box": {"x": 100, "y": 60}},
                                {"text": "2025\u5e746\u670819\u65e5", "row_index": 1, "col_index": 2, "bounding_box": {"x": 140, "y": 60}},
                                {"text": "\u59da\u6d2a\u83ca", "row_index": 2, "col_index": 0, "bounding_box": {"x": 10, "y": 100}},
                                {"text": "510602197307030982", "row_index": 2, "col_index": 1, "bounding_box": {"x": 40, "y": 100}},
                                {"text": "\u59d4\u6258\u4ee3\u7406\u4eba", "row_index": 2, "col_index": 2, "bounding_box": {"x": 100, "y": 100}},
                                {"text": "2025\u5e746\u670819\u65e5", "row_index": 2, "col_index": 2, "bounding_box": {"x": 140, "y": 100}},
                            ],
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pdf_extract_module._normalize_markdown_artifacts(tmp_path)

    normalized = markdown_path.read_text(encoding="utf-8")
    assert "\u4eba\u8868\uff09" not in normalized
    assert "| \u9648\u827a\u7476 | 510603199605255966 | \u6267\u884c\u4e8b\u52a1\u5408\u4f19\u4eba\uff08\u59d4\u6d3e\u4ee3\u8868\uff09 | 2025\u5e746\u670819\u65e5 |" in normalized
    assert "| \u59da\u6d2a\u83ca | 510602197307030982 | \u59d4\u6258\u4ee3\u7406\u4eba | 2025\u5e746\u670819\u65e5 |" in normalized


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

    def _fake_render(input_path, *, output_dir, page_count, dpi, input_type="pdf"):  # noqa: ANN001
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
                    "source_pdf": str(input_path),
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

    monkeypatch.setattr(pdf_extract_module, "_render_input_page_screenshots", _fake_render)

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
    assert first.seconds_per_page > 0
    assert Path(first.batch_report_path).exists()
    assert Path(first.batch_report_csv_path).exists()
    assert first_report["page_count"] == 1
    assert first_report["seconds_per_page"] == first.seconds_per_page
    assert "pages_per_second" not in first_report
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
    assert image_item.converted_pdf_path is None  # No longer converted


def test_extract_pdf_dir_preserves_recursive_relative_directories(tmp_path: Path) -> None:
    from PIL import Image

    input_dir = tmp_path / "input"
    nested_dir = input_dir / "3-WS-001"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (24, 24), (255, 255, 255)).save(nested_dir / "0189.jpg")
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
        recursive=True,
        overwrite=True,
    )

    item = report.items[0]
    assert item.relative_input_path == "3-WS-001/0189.jpg"
    assert item.output_relative_dir == "3-WS-001"
    assert Path(item.artifact_dir) == tmp_path / "output" / "3-WS-001" / "0189"
    assert Path(item.json_path or "") == tmp_path / "output" / "3-WS-001" / "0189" / "0189.json"
    assert item.converted_pdf_path is None  # No longer converted


def test_extract_pdf_dir_preserves_direct_business_folder_name(tmp_path: Path) -> None:
    input_dir = tmp_path / "3-WS-001"
    input_dir.mkdir()
    (input_dir / "0190.pdf").write_bytes(b"%PDF-1.4\n")
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

    item = report.items[0]
    assert item.relative_input_path == "0190.pdf"
    assert item.output_relative_dir == "3-WS-001"
    assert Path(item.artifact_dir) == tmp_path / "output" / "3-WS-001" / "0190"
    assert Path(item.json_path or "") == tmp_path / "output" / "3-WS-001" / "0190" / "0190.json"


def test_extract_pdf_dir_collapses_duplicate_parent_and_file_stem(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    duplicate_dir = input_dir / "1"
    duplicate_dir.mkdir(parents=True)
    (duplicate_dir / "1.pdf").write_bytes(b"%PDF-1.4\n")
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
        recursive=True,
        overwrite=True,
    )

    item = report.items[0]
    assert item.relative_input_path == "1/1.pdf"
    assert item.output_relative_dir is None
    assert Path(item.artifact_dir) == tmp_path / "output" / "1"
    assert Path(item.json_path or "") == tmp_path / "output" / "1" / "1.json"
