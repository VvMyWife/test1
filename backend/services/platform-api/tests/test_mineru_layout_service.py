from __future__ import annotations

import json
from pathlib import Path

from app.schemas.mineru_layout import MinerULayoutExtractRequest
from app.services.mineru_layout_service import MinerULayoutService
from platform_foundation.contracts import ArtifactRef, BoundingBox, PageItem, TextBlock


def test_extract_layout_writes_pure_mineru_json_and_returns_summary(tmp_path: Path) -> None:
    artifact = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")
    text_block = TextBlock(
        text="hello",
        bounding_box=BoundingBox(x=10, y=20, w=30, h=40),
        block_type="text",
    )

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
                text_blocks=[text_block],
                layout_ref=artifact,
                page_meta={
                    "coord_space": "mineru_layout",
                    "width": 100,
                    "height": 200,
                    "mineru_artifacts": [artifact.model_dump(mode="python")],
                },
            ).model_dump(mode="python")

    service = MinerULayoutService(operator_factory=_FakeOperator)
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    result = service.extract_layout(
        MinerULayoutExtractRequest(
            file_uri=str(tmp_path / "doc.pdf"),
            output_dir=str(tmp_path / "output"),
        )
    )

    assert result.source_file_name == "doc.pdf"
    assert result.page_count == 1
    assert result.text_block_count == 1
    assert result.table_block_count == 0
    assert Path(result.json_path).exists()
    assert Path(result.artifact_dir).name == "doc"

    payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
    assert "parsed_pdf" not in payload
    assert payload["pages"][0]["text_blocks"][0]["bounding_box"]["w"] == 30
    assert payload["pages"][0]["text"] == "hello"
    assert "archive_id" not in payload["pages"][0]
    assert "triggered_by_user_id" not in payload["pages"][0]
