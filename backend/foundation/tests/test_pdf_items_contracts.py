from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from platform_foundation.contracts import (
    ArchiveJobItem,
    ArtifactRef,
    CoordSpace,
    DocumentItem,
    DocumentResultItem,
    PageItem,
    ProvenanceMin,
    SensitiveSpanItem,
)


def test_archive_job_item_roundtrip() -> None:
    raw = {
        "archive_id": "a1",
        "job_id": "j1",
        "triggered_by_user_id": "u_trigger",
        "triggered_at": "2026-01-15T10:00:00Z",
        "source_paths": ["/data/x.pdf"],
        "options": {"ocr": True},
        "config_version": "v1",
    }
    m = ArchiveJobItem.model_validate(raw)
    out = m.model_dump(mode="json")
    assert out["triggered_by_user_id"] == "u_trigger"
    assert "tenant_id" not in out


def test_document_page_span_result_chain() -> None:
    owner = "u_owner"
    trigger = "u_trigger"
    doc = DocumentItem(
        archive_id="a1",
        archive_owner_user_id=owner,
        triggered_by_user_id=trigger,
        doc_id="d1",
        file_uri="s3://b/x.pdf",
        num_pages=2,
    )
    page = PageItem(
        archive_id=doc.archive_id,
        archive_owner_user_id=owner,
        triggered_by_user_id=trigger,
        doc_id=doc.doc_id,
        page_index=0,
        text="hello",
    )
    assert page.archive_id == doc.archive_id
    now = datetime.now(UTC)
    span = SensitiveSpanItem(
        archive_id=doc.archive_id,
        archive_owner_user_id=owner,
        triggered_by_user_id=trigger,
        doc_id=doc.doc_id,
        page_index=0,
        span_id="s1",
        text="hello",
        risk_type="secret",
        score=0.9,
        bbox=(0.0, 0.0, 10.0, 10.0),
        coord_space=CoordSpace.MINERU_LAYOUT,
        provenance=ProvenanceMin(
            source="mineru_span",
            page_index=0,
            char_start=0,
            char_end=5,
        ),
        detected_at=now,
    )
    result = DocumentResultItem(
        doc_id=doc.doc_id,
        archive_id=doc.archive_id,
        archive_owner_user_id=owner,
        triggered_by_user_id=trigger,
        summary={"risk": "high", "hits": 1},
        spans=[span],
    )
    data = result.model_dump(mode="json")
    restored = DocumentResultItem.model_validate(data)
    assert restored.spans[0].span_id == "s1"
    assert restored.spans[0].bbox == (0.0, 0.0, 10.0, 10.0)
    assert restored.spans[0].coord_space == CoordSpace.MINERU_LAYOUT


def test_sensitive_span_bbox_requires_coord_space() -> None:
    with pytest.raises(ValidationError):
        SensitiveSpanItem(
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            doc_id="d1",
            page_index=0,
            span_id="s1",
            text="x",
            risk_type="secret",
            score=0.5,
            bbox=(0.0, 0.0, 1.0, 1.0),
            coord_space=None,
        )


def test_document_result_spans_must_match_doc_id() -> None:
    span = SensitiveSpanItem(
        archive_id="a1",
        archive_owner_user_id="o1",
        triggered_by_user_id="t1",
        doc_id="other",
        page_index=0,
        span_id="s1",
        text="x",
        risk_type="secret",
        score=0.5,
    )
    with pytest.raises(ValidationError):
        DocumentResultItem(
            doc_id="d1",
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            summary={},
            spans=[span],
        )


def test_document_result_spans_must_match_archive_id() -> None:
    span = SensitiveSpanItem(
        archive_id="wrong",
        archive_owner_user_id="o1",
        triggered_by_user_id="t1",
        doc_id="d1",
        page_index=0,
        span_id="s1",
        text="x",
        risk_type="secret",
        score=0.5,
    )
    with pytest.raises(ValidationError):
        DocumentResultItem(
            doc_id="d1",
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            summary={},
            spans=[span],
        )


def test_document_result_spans_must_match_owner_and_trigger() -> None:
    span = SensitiveSpanItem(
        archive_id="a1",
        archive_owner_user_id="o_span",
        triggered_by_user_id="t1",
        doc_id="d1",
        page_index=0,
        span_id="s1",
        text="x",
        risk_type="secret",
        score=0.5,
    )
    with pytest.raises(ValidationError):
        DocumentResultItem(
            doc_id="d1",
            archive_id="a1",
            archive_owner_user_id="o_result",
            triggered_by_user_id="t1",
            summary={},
            spans=[span],
        )


def test_archive_job_rejects_empty_source_path() -> None:
    with pytest.raises(ValidationError):
        ArchiveJobItem(
            archive_id="a1",
            job_id="j1",
            triggered_by_user_id="u1",
            source_paths=["/ok.pdf", "   "],
        )


def test_artifact_ref_uri_xor_inline() -> None:
    ArtifactRef(kind="json", uri="s3://b/x.json")
    ArtifactRef(kind="json", inline={"a": 1})
    with pytest.raises(ValidationError):
        ArtifactRef(kind="json", uri="u", inline={})
    with pytest.raises(ValidationError):
        ArtifactRef(kind="json")


def test_provenance_char_range() -> None:
    with pytest.raises(ValidationError):
        ProvenanceMin(source="s", page_index=0, char_start=5, char_end=3)


def test_sensitive_span_provenance_page_mismatch() -> None:
    with pytest.raises(ValidationError):
        SensitiveSpanItem(
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            doc_id="d1",
            page_index=0,
            span_id="s1",
            text="x",
            risk_type="secret",
            score=0.5,
            provenance=ProvenanceMin(source="s", page_index=1, char_start=0, char_end=1),
        )


def test_num_pages_when_set_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        DocumentItem(
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            doc_id="d1",
            file_uri="/x.pdf",
            num_pages=0,
        )


def test_confirmation_must_be_paired() -> None:
    with pytest.raises(ValidationError):
        DocumentResultItem(
            doc_id="d1",
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            summary={},
            confirmed_by_user_id="u_conf",
            confirmed_at=None,
        )


def test_detection_actor_user_requires_detected_by() -> None:
    with pytest.raises(ValidationError):
        SensitiveSpanItem(
            archive_id="a1",
            archive_owner_user_id="o1",
            triggered_by_user_id="t1",
            doc_id="d1",
            page_index=0,
            span_id="s1",
            text="x",
            risk_type="secret",
            score=0.5,
            detection_actor="user",
            detected_by_user_id=None,
        )
