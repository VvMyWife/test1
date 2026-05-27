from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from platform_foundation.contracts import (
    ArtifactRef,
    BoundingBox,
    CoordSpace,
    DocumentItem,
    ImageSize,
    PageItem,
    TableBlock,
    TableCell,
    TextBlock,
)
from platform_foundation.inference import (
    InlineMinerUDocumentService,
    MinerUServiceError,
    parse_mineru_content_list_json,
    parse_mineru_middle_json,
    parse_mineru_table_candidates_json,
)
from platform_foundation.inference.mineru import (
    MinerUDocumentParseResult,
    MinerUPageResult,
    _coerce_local_file_path,
    _merge_table_result,
)
from platform_foundation.inference.paddle_table import PaddleTableStructureResult, parse_paddle_structure_tables
from platform_foundation.ocr.pure_mineru import _build_mineru_options
from platform_foundation.operators import OperatorContext, OperatorError
from platform_foundation.operators.layout_extract_mineru import LayoutExtractMinerUOperator
from platform_foundation.operators.layout_extract_mineru_paddle_table import (
    LayoutExtractMinerUPaddleTableOperator,
)


def _sample_middle_json() -> dict:
    return {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [595.0, 842.0],
                "para_blocks": [
                    {
                        "type": "title",
                        "bbox": [10, 20, 110, 50],
                        "lines": [{"spans": [{"content": "Archive Title"}]}],
                    },
                    {
                        "type": "text",
                        "bbox": [10, 60, 210, 100],
                        "lines": [{"spans": [{"content": "第一页正文"}]}],
                    },
                ],
            },
            {
                "page_idx": 1,
                "page_size": [595.0, 842.0],
                "preproc_blocks": [
                    {"bbox": [5, 5, 105, 35], "lines": [{"spans": [{"content": "Second page"}]}]},
                ],
            },
        ]
    }


def _sample_table_middle_json() -> dict:
    return {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [595.0, 842.0],
                "para_blocks": [
                    {
                        "type": "table",
                        "bbox": [10, 20, 210, 120],
                        "score": 0.98,
                        "blocks": [
                            {
                                "type": "table_body",
                                "bbox": [10, 20, 210, 120],
                                "lines": [
                                    {
                                        "spans": [
                                            {
                                                "type": "table",
                                                "html": (
                                                    "<table><tr><td>A</td><td>B</td></tr>"
                                                    "<tr><td>1</td><td>2</td></tr></table>"
                                                ),
                                            }
                                        ]
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_parse_mineru_middle_json_extracts_pages() -> None:
    parsed = parse_mineru_middle_json(
        _sample_middle_json(),
        middle_json_uri="file:///tmp/example_middle.json",
        parser_version="mineru-2.7.6",
    )

    assert parsed.page_count == 2
    assert parsed.coord_space == CoordSpace.MINERU_LAYOUT
    assert parsed.middle_json_ref.kind == "middle_json"
    assert parsed.pages[0].page_index == 0
    assert parsed.pages[0].text == "Archive Title\n第一页正文"
    assert parsed.pages[0].text_blocks[0].text == "Archive Title"
    assert parsed.pages[0].text_blocks[0].bounding_box.w == 100
    assert parsed.parsed_pdf is not None
    assert parsed.parsed_pdf.pages[0].text_blocks[1].text == "第一页正文"
    assert parsed.pages[1].text == "Second page"


def test_parse_mineru_content_list_json_extracts_daft_text_blocks() -> None:
    blocks_by_page = parse_mineru_content_list_json(
        [
            {
                "type": "text",
                "text": "第一条",
                "bbox": [125, 266, 883, 394],
                "page_idx": 0,
                "score": 0.99,
            },
            {
                "type": "page_number",
                "text": "3",
                "bbox": [840, 913, 863, 929],
                "page_idx": 0,
            },
        ]
    )

    assert blocks_by_page[0][0].text == "第一条"
    assert blocks_by_page[0][0].bounding_box.x == 125
    assert blocks_by_page[0][0].bounding_box.w == 758
    assert blocks_by_page[0][1].block_type == "page_number"


def test_parse_mineru_table_candidates_json_extracts_table_crops(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "table.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"fake")

    candidates_by_page = parse_mineru_table_candidates_json(
        [
            {
                "type": "table",
                "bbox": [10, 20, 110, 220],
                "page_idx": 2,
                "img_path": "images/table.jpg",
                "table_caption": ["caption"],
            }
        ],
        base_dir=tmp_path,
    )

    candidate = candidates_by_page[2][0]
    assert candidate["bbox"]["w"] == 100
    assert candidate["image_uri"] == str(image_path.resolve())
    assert candidate["caption"] == "caption"


def test_parse_mineru_middle_json_preserves_mineru_table_output() -> None:
    parsed = parse_mineru_middle_json(
        _sample_table_middle_json(),
        middle_json_uri="file:///tmp/table_middle.json",
    )

    page = parsed.pages[0]
    assert page.page_meta["mineru_has_table"] is True
    assert page.page_meta["mineru_table_output"]["provider"] == "mineru"
    assert len(page.table_blocks) == 1

    table = page.table_blocks[0]
    assert table.provider == "mineru"
    assert table.bounding_box is not None
    assert table.bounding_box.w == 200
    assert table.html is not None
    assert table.cells == []
    assert table.meta["source"] == "mineru_middle_json_table"
    assert table.meta["raw_table"]["type"] == "table"
    assert parsed.parsed_pdf is not None
    assert parsed.parsed_pdf.pages[0].table_blocks[0].provider == "mineru"


def test_parse_mineru_middle_json_preserves_empty_mineru_table_output() -> None:
    parsed = parse_mineru_middle_json(
        {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [595.0, 842.0],
                    "para_blocks": [
                        {
                            "type": "table",
                            "bbox": [10, 20, 210, 120],
                            "blocks": [],
                        }
                    ],
                }
            ]
        },
        middle_json_uri="file:///tmp/empty_table_middle.json",
    )

    page = parsed.pages[0]
    assert page.page_meta["mineru_has_table"] is True
    assert page.page_meta["mineru_table_output"]["table_count"] == 1
    assert len(page.table_blocks) == 1
    assert page.table_blocks[0].cells == []
    assert page.table_blocks[0].html is None


def test_parse_paddle_structure_tables_extracts_scaled_cells() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 200,
                "height": 100,
                "table_res_list": [
                    {
                        "pred_html": "<table><tr><td>A</td><td>B</td></tr></table>",
                        "table_ocr_pred": {
                            "rec_texts": ["A", "B"],
                            "rec_scores": [0.95, 0.9],
                            "rec_boxes": [[0, 0, 50, 20], [60, 0, 110, 20]],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=100, height=50)},
    )

    table = parsed.tables_by_page[0][0]
    assert table.provider == "paddleocr_ppstructurev3"
    assert table.coord_space == "mineru_layout"
    assert table.cells[0].text == "A"
    assert table.cells[0].bounding_box is not None
    assert table.cells[0].bounding_box.w == 25
    assert table.cells[1].col_index == 1
    assert table.html is None
    assert parsed.meta["cell_count"] == 2


def test_parse_paddle_structure_tables_merges_wrapped_html_cell_text() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 500,
                "height": 600,
                "table_res_list": [
                    {
                        "pred_html": (
                            "<table><tr><td>28</td><td>房地产行业监 管处</td>"
                            "<td>陶姜</td><td>处长</td></tr></table>"
                        ),
                        "table_ocr_pred": {
                            "rec_texts": ["28", "房地产行业监", "陶姜", "处长", "29", "管处"],
                            "rec_boxes": [
                                [80, 100, 100, 120],
                                [140, 100, 245, 120],
                                [260, 100, 300, 120],
                                [360, 100, 410, 120],
                                [80, 132, 100, 152],
                                [140, 132, 180, 152],
                            ],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=500, height=600)},
    )

    table = parsed.tables_by_page[0][0]
    texts = [cell.text for cell in table.cells]
    assert "房地产行业监管处" in texts
    assert "房地产行业监" not in texts
    assert "管处" not in texts
    merged = next(cell for cell in table.cells if cell.text == "房地产行业监管处")
    assert merged.row_span == 1
    assert merged.col_span == 1
    assert merged.meta["merge_source"] == "paddle_pred_html"
    assert merged.meta["merge_type"] == "wrapped_cell_text"
    assert "merged_from_cell_ids" not in merged.meta
    assert "merged_text_fragments" not in merged.meta


def test_parse_paddle_structure_tables_merges_grid_split_html_cell_text() -> None:
    enter = "\u5165"
    party = "\u515a"
    time_a = "\u65f6"
    time_b = "\u95f4"
    join_party_time = "\u5165\u515a\u65f6\u95f4"
    work_time = "\u53c2\u52a0\u5de5\u4f5c\u65f6\u95f4"

    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 500,
                "height": 600,
                "table_res_list": [
                    {
                        "pred_html": (
                            f"<table><tr><td>{enter} {party} {time_a} {time_b}</td>"
                            "<td>1995.05</td><td>\u53c2\u52a0\u5de5 \u4f5c\u65f6\u95f4</td>"
                            "<td>1995.07</td></tr></table>"
                        ),
                        "table_ocr_pred": {
                            "rec_texts": [
                                enter,
                                party,
                                "\u53c2\u52a0\u5de5",
                                "1995.05",
                                "1995.07",
                                time_a,
                                time_b,
                                "\u4f5c\u65f6\u95f4",
                            ],
                            "rec_boxes": [
                                [79, 167, 98, 185],
                                [104, 166, 123, 187],
                                [194, 166, 239, 184],
                                [137, 175, 184, 191],
                                [251, 175, 298, 190],
                                [78, 181, 108, 201],
                                [103, 181, 122, 200],
                                [194, 181, 239, 201],
                            ],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=500, height=600)},
    )

    table = parsed.tables_by_page[0][0]
    texts = [cell.text for cell in table.cells]
    assert join_party_time in texts
    assert work_time in texts
    for old_fragment in [
        enter,
        party,
        time_a,
        time_b,
        "\u53c2\u52a0\u5de5",
        "\u4f5c\u65f6\u95f4",
    ]:
        assert old_fragment not in texts

    join_party_cell = next(cell for cell in table.cells if cell.text == join_party_time)
    assert join_party_cell.row_span == 1
    assert join_party_cell.col_span == 1
    assert join_party_cell.meta["merge_source"] == "paddle_pred_html"
    assert join_party_cell.meta["merge_type"] == "wrapped_cell_text"


def test_merge_table_result_replaces_flat_table_text_by_default() -> None:
    middle_ref = ArtifactRef(kind="middle_json", uri="/tmp/doc_middle.json")
    table_bbox = BoundingBox(x=10, y=10, w=100, h=80)
    outside_block = TextBlock(
        text="outside",
        bounding_box=BoundingBox(x=10, y=120, w=50, h=20),
        block_type="text",
        meta={"source": "middle_json"},
    )
    inside_block = TextBlock(
        text="old table text",
        bounding_box=BoundingBox(x=20, y=20, w=40, h=20),
        block_type="text",
        meta={"source": "middle_json"},
    )
    fallback_block = TextBlock(
        text="fallback table text",
        bounding_box=BoundingBox(x=20, y=50, w=40, h=20),
        block_type="table_cell",
        meta={"source": "mineru_ocr_table_fallback"},
    )
    parsed = MinerUDocumentParseResult(
        pages=(
            MinerUPageResult(
                page_index=0,
                text="outside\nold table text\nfallback table text",
                text_blocks=(outside_block, inside_block, fallback_block),
                table_blocks=(
                    TableBlock(
                        table_id="p0-ocr-t0",
                        page_index=0,
                        provider="mineru_ocr",
                        bounding_box=table_bbox,
                        cells=[],
                    ),
                ),
                image_size=ImageSize(width=200, height=200),
            ),
        ),
        middle_json_ref=middle_ref,
        page_count=1,
    )
    paddle_table = TableBlock(
        table_id="p0-t0",
        page_index=0,
        provider="paddleocr_ppstructurev3",
        bounding_box=table_bbox,
        cells=[
            TableCell(
                cell_id="p0-t0-c0",
                text="new table text",
                bounding_box=BoundingBox(x=20, y=20, w=40, h=20),
                row_index=0,
                col_index=0,
            )
        ],
    )

    merged = _merge_table_result(
        parsed,
        PaddleTableStructureResult(
            tables_by_page={0: (paddle_table,)},
            meta={"provider": "paddleocr_ppstructurev3"},
        ),
        emit_text_blocks=False,
        replace_existing_table_blocks=True,
    )

    page = merged.pages[0]
    assert [block.text for block in page.text_blocks] == ["outside"]
    assert page.text == "outside"
    assert [table.provider for table in page.table_blocks] == ["paddleocr_ppstructurev3"]
    assert page.table_blocks[0].cells[0].text == "new table text"


def test_layout_extract_mineru_operator_fans_out_document_into_pages() -> None:
    parsed = parse_mineru_middle_json(
        _sample_middle_json(),
        middle_json_uri="file:///tmp/example_middle.json",
    )
    op = LayoutExtractMinerUOperator(service=InlineMinerUDocumentService(result=parsed))
    ctx = OperatorContext(trace_id="trace-1", run_id="run-1", config_version="cfg-v1")
    document = DocumentItem(
        archive_id="archive-1",
        archive_owner_user_id="owner-1",
        triggered_by_user_id="user-1",
        doc_id="doc-1",
        file_uri="file:///tmp/example.pdf",
        num_pages=2,
    )

    out = list(op.process(ctx, iter([document.model_dump(mode="python")]), path="item"))

    assert len(out) == 2
    page0 = PageItem.model_validate(out[0])
    page1 = PageItem.model_validate(out[1])
    assert page0.doc_id == "doc-1"
    assert page0.page_index == 0
    assert page0.page_meta["layout_provider"] == "mineru"
    assert page0.page_meta["config_version"] == "cfg-v1"
    assert page0.page_meta["operator_safety"]["timeout_seconds"] == 1800.0
    assert page0.page_meta["operator_safety"]["max_inflight"] is None
    assert page0.layout_ref is not None
    assert page0.text_blocks[0].text == "Archive Title"
    assert page1.page_index == 1
    assert page1.text == "Second page"


def test_layout_extract_mineru_operator_waits_for_inflight_slot_by_default() -> None:
    LayoutExtractMinerUOperator._semaphores.clear()
    op = LayoutExtractMinerUOperator(max_inflight=1)
    ctx = OperatorContext(trace_id="trace-1", run_id="run-1", config_version="cfg-v1")
    document = DocumentItem(
        archive_id="archive-1",
        archive_owner_user_id="owner-1",
        triggered_by_user_id="user-1",
        doc_id="doc-1",
        file_uri="file:///tmp/example.pdf",
    )
    completed = threading.Event()
    errors: list[BaseException] = []

    def wait_for_slot() -> None:
        try:
            with op._inflight_slot(ctx, document, {}, ctx.logger):  # noqa: SLF001
                completed.set()
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    with op._inflight_slot(ctx, document, {}, ctx.logger):  # noqa: SLF001
        thread = threading.Thread(target=wait_for_slot)
        thread.start()
        time.sleep(0.05)
        assert not completed.is_set()
        assert errors == []

    thread.join(timeout=1)

    assert completed.is_set()
    assert errors == []


def test_layout_extract_mineru_operator_injects_timeout_and_preserves_override() -> None:
    parsed = parse_mineru_middle_json(
        _sample_middle_json(),
        middle_json_uri="file:///tmp/example_middle.json",
    )

    class _CapturingService:
        options: dict | None = None

        def parse_document(self, *, file_uri, mime_type=None, options=None):  # noqa: ANN001
            self.options = dict(options or {})
            return parsed

    service = _CapturingService()
    op = LayoutExtractMinerUOperator(service=service)
    ctx = OperatorContext(trace_id="trace-1", run_id="run-1")
    document = DocumentItem(
        archive_id="archive-1",
        archive_owner_user_id="owner-1",
        triggered_by_user_id="user-1",
        doc_id="doc-1",
        file_uri="file:///tmp/example.pdf",
    )

    out = list(op.process(ctx, iter([document.model_dump(mode="python")]), path="item"))

    assert service.options is not None
    assert service.options["timeout_seconds"] == 1800.0
    assert PageItem.model_validate(out[0]).page_meta["operator_safety"]["timeout_seconds"] == 1800.0

    document_with_override = document.model_copy(
        update={"meta": {"mineru_options": {"timeout_seconds": 45}}}
    )

    list(op.process(ctx, iter([document_with_override.model_dump(mode="python")]), path="item"))

    assert service.options["timeout_seconds"] == 45.0


def test_layout_extract_mineru_paddle_table_operator_injects_defaults() -> None:
    parsed = parse_mineru_middle_json(
        _sample_middle_json(),
        middle_json_uri="file:///tmp/example_middle.json",
    )

    class _CapturingService:
        options: dict | None = None

        def parse_document(self, *, file_uri, mime_type=None, options=None):  # noqa: ANN001
            self.options = dict(options or {})
            return parsed

    service = _CapturingService()
    op = LayoutExtractMinerUPaddleTableOperator(service=service)
    ctx = OperatorContext(trace_id="trace-1", run_id="run-1")
    document = DocumentItem(
        archive_id="archive-1",
        archive_owner_user_id="owner-1",
        triggered_by_user_id="user-1",
        doc_id="doc-1",
        file_uri="file:///tmp/example.pdf",
    )

    list(op.process(ctx, iter([document.model_dump(mode="python")]), path="item"))

    assert service.options is not None
    assert service.options["enable_table_cell_refine"] is True
    assert service.options["table_cell_refine_when_tables_present"] is True
    assert service.options["table_cell_refine_fail_open"] is False
    assert service.options["paddle_table_mode"] == "ppstructurev3"
    assert service.options["timeout_seconds"] == 1800.0


def test_pure_mineru_options_default_to_ocr_table_engine() -> None:
    options = _build_mineru_options(
        output_dir=None,
        api_url="http://127.0.0.1:8000",
        timeout_seconds=1800.0,
        parse_method="auto",
        backend="pipeline",
        lang="ch",
        extra_args=None,
        table_engine="ocr",
        paddle_table_mode="ppstructurev3",
        paddle_device=None,
        mineru_options=None,
    )

    assert options["table_engine"] == "ocr"
    assert options["extra_args"] == ["--formula", "false", "--table", "true"]
    assert options["enable_table_cell_refine"] is False
    assert options["enable_paddle_table_refine"] is False
    assert "paddle_table_mode" not in options


class _FailingMinerUService:
    def parse_document(self, **_: object) -> object:
        raise MinerUServiceError(
            "MinerU temporary unavailable",
            code="MINERU_TIMEOUT",
            retryable=True,
        )


def test_layout_extract_mineru_operator_maps_service_errors_and_records_failure(
    tmp_path: Path,
) -> None:
    op = LayoutExtractMinerUOperator(service=_FailingMinerUService())
    ctx = OperatorContext(trace_id="trace-1", run_id="run-1")
    document = DocumentItem(
        archive_id="archive-1",
        archive_owner_user_id="owner-1",
        triggered_by_user_id="user-1",
        doc_id="doc-1",
        file_uri="file:///tmp/example.pdf",
        meta={"mineru_options": {"output_dir": str(tmp_path)}},
    )

    with pytest.raises(OperatorError) as exc_info:
        list(op.process(ctx, iter([document.model_dump(mode="python")]), path="item"))

    assert exc_info.value.code == "MINERU_TIMEOUT"
    assert exc_info.value.retryable is True
    failure_record = tmp_path / "operate_error.json"
    assert failure_record.exists()
    assert exc_info.value.details["failure_record"] == str(failure_record.resolve())
    payload = json.loads(failure_record.read_text(encoding="utf-8"))
    assert payload["document"]["doc_id"] == "doc-1"
    assert payload["error"]["code"] == "MINERU_TIMEOUT"
    assert payload["service_options"]["timeout_seconds"] == 1800.0


def test_coerce_local_file_path_accepts_native_absolute_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    resolved = _coerce_local_file_path(str(pdf_path))

    assert resolved == pdf_path.resolve()


def test_coerce_local_file_path_accepts_file_uri(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    resolved = _coerce_local_file_path(pdf_path.as_uri())

    assert resolved == pdf_path.resolve()
