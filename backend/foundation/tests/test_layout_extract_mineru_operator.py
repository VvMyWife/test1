from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any

import pytest

import platform_foundation.inference.paddle_table as paddle_table_module
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
    _cleanup_mineru_api_task_dirs,
    _merge_table_result,
    _should_cleanup_mineru_api_task_dirs,
)
from platform_foundation.inference.paddle_table import (
    PaddleTableStructureResult,
    PaddleTableStructureService,
    _build_paddleocr_vl,
    _parse_table_structure_module_result,
    parse_paddle_ocr_text_blocks,
    parse_paddle_structure_tables,
)
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


def test_parse_mineru_table_candidates_json_preserves_table_body_html(tmp_path: Path) -> None:
    candidates_by_page = parse_mineru_table_candidates_json(
        [
            {
                "type": "table",
                "bbox": [0, 0, 100, 50],
                "page_idx": 0,
                "img_path": "images/table.jpg",
                "table_body": "<table><tr><td>姓名</td><td>结果</td></tr></table>",
            }
        ],
        base_dir=tmp_path,
    )

    assert candidates_by_page[0][0]["html"] == "<table><tr><td>姓名</td><td>结果</td></tr></table>"


def test_table_structure_fallback_fills_cell_text_from_mineru_html() -> None:
    table = _parse_table_structure_module_result(
        {"bbox": [[0, 0, 10, 10], [10, 0, 20, 10]], "structure_score": 0.9},
        page_index=0,
        table_index=0,
        candidate={
            "bbox": {"x": 0, "y": 0, "w": 20, "h": 10},
            "html": "<table><tr><td>姓名</td><td>结果</td></tr></table>",
        },
        page_text_blocks=(),
    )

    assert table is not None
    assert [cell.text for cell in table.cells] == ["姓名", "结果"]
    assert table.cells[0].meta["text_source"] == "mineru_table_html"


def test_table_structure_fallback_preserves_duplicate_cells_and_repairs_dates() -> None:
    headers = [
        "\u59d3\u540d",
        "\u8eab\u4efd\u8bc1\u53f7\u7801",
        "\u89d2\u8272",
        "\u5b9e\u540d\u8ba4\u8bc1\u65f6\u95f4",
        "\u4eba\u50cf\u4fe1\u606f",
        "\u7b7e\u5b57\u4fe1\u606f",
        "\u8ba4\u8bc1\u65b9\u5f0f",
        "\u7ed3\u679c",
    ]
    rows = [
        headers,
        [
            "\u5411\u4fca\u5b66",
            "510626199209193753",
            "\u6cd5\u5b9a\u4ee3\u8868\u4eba",
            "",
            "",
            "",
            "\u603b\u5c40\u8ba4\u8bc1",
            "\u6210\u529f",
        ],
        [
            "\u5411\u4fca\u5b66",
            "510626199209193753",
            "\u80a1\u4e1c",
            "2\u65e5",
            "",
            "",
            "\u603b\u5c40\u8ba4\u8bc1",
            "\u6210\u529f",
        ],
    ]
    html = "<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    ) + "</table>"
    boxes = [
        [col * 10, row * 10, (col + 1) * 10, (row + 1) * 10]
        for row in range(len(rows))
        for col in range(len(headers))
    ]

    table = _parse_table_structure_module_result(
        {"bbox": boxes, "structure_score": 0.9},
        page_index=0,
        table_index=0,
        candidate={"bbox": {"x": 0, "y": 0, "w": 80, "h": 30}, "html": html},
        page_text_blocks=(
            TextBlock(
                text="\u4e1a\u52a1\u6d41\u6c34\u53f7:510604Y3202505140001",
                bounding_box=BoundingBox(x=0, y=40, w=80, h=10),
            ),
        ),
    )

    assert table is not None
    texts = [cell.text for cell in table.cells]
    assert texts.count("\u5411\u4fca\u5b66") == 2
    assert texts.count("510626199209193753") == 2
    assert texts[11] == "2025\u5e745\u670814\u65e5"
    assert texts[19] == "2025\u5e745\u670814\u65e5"


def test_ppstructurev3_artifact_includes_normalized_tables(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    class _FakePipeline:
        def predict(self, *, input, **kwargs):  # noqa: ANN001
            del input, kwargs
            return [
                {
                    "page_index": 0,
                    "width": 100,
                    "height": 100,
                    "table_res_list": [
                        {
                            "cell_box_list": [[0, 0, 20, 10], [20, 0, 40, 10]],
                            "table_ocr_pred": {
                                "rec_texts": ["\u59d3\u540d", "\u7ed3\u679c"],
                                "rec_scores": [0.98, 0.97],
                            },
                        }
                    ],
                }
            ]

    monkeypatch.setattr(
        "platform_foundation.inference.paddle_table._build_ppstructure_v3",
        lambda options: _FakePipeline(),
    )

    result = PaddleTableStructureService()._extract_tables_with_ppstructurev3(
        file_uri="/tmp/0062.jpg",
        output_dir=tmp_path,
        target_page_sizes={0: ImageSize(width=200, height=200)},
        options={},
    )

    payload = json.loads((tmp_path / "paddle_table_structure.json").read_text(encoding="utf-8"))
    assert payload["provider"] == "paddleocr_ppstructurev3"
    assert payload["table_count"] == 1
    assert payload["cell_count"] == 2
    assert payload["tables_by_page"]["0"][0]["cells"][0]["text"] == "\u59d3\u540d"
    assert payload["tables_by_page"]["0"][0]["cells"][1]["text"] == "\u7ed3\u679c"
    assert payload["raw_results"][0]["table_res_list"][0]["table_ocr_pred"]["rec_texts"] == [
        "\u59d3\u540d",
        "\u7ed3\u679c",
    ]
    assert result.artifact_ref is not None


def test_paddle_auto_falls_back_when_ppstructure_misses_html_cells(monkeypatch) -> None:  # noqa: ANN001
    service = PaddleTableStructureService()
    pp_table = TableBlock(
        table_id="p0-t0",
        page_index=0,
        provider="paddleocr_ppstructurev3",
        cells=[TableCell(cell_id="p0-t0-c0", text="\u59d3\u540d")],
    )
    fallback_table = TableBlock(
        table_id="p0-t0",
        page_index=0,
        provider="paddleocr_table_structure",
        cells=[
            TableCell(cell_id="p0-t0-c0", text="\u59d3\u540d"),
            TableCell(cell_id="p0-t0-c1", text="\u7ed3\u679c"),
        ],
    )

    def fake_ppstructure(self, **kwargs):  # noqa: ANN001
        del self, kwargs
        return PaddleTableStructureResult(
            tables_by_page={0: (pp_table,)},
            meta={"provider": "paddleocr_ppstructurev3", "table_count": 1, "cell_count": 1},
        )

    def fake_mineru_crops(self, **kwargs):  # noqa: ANN001
        del self, kwargs
        return PaddleTableStructureResult(
            tables_by_page={0: (fallback_table,)},
            meta={"provider": "paddleocr_table_structure", "table_count": 1, "cell_count": 2},
        )

    monkeypatch.setattr(PaddleTableStructureService, "_extract_tables_with_ppstructurev3", fake_ppstructure)
    monkeypatch.setattr(PaddleTableStructureService, "_extract_tables_from_mineru_crops", fake_mineru_crops)

    result = service._extract_with_paddle3_auto(
        file_uri="/tmp/0062.jpg",
        output_dir=Path("/tmp/out"),
        target_page_sizes={},
        options={},
        table_candidates_by_page={
            0: [
                {
                    "html": "<table><tr><td>\u59d3\u540d</td><td>\u7ed3\u679c</td></tr></table>",
                }
            ]
        },
        page_text_blocks_by_page={},
    )

    assert result.tables_by_page[0][0].provider == "paddleocr_table_structure"
    assert result.meta["mode"] == "auto_ppstructurev3_quality_fallback_mineru_crops"
    assert result.meta["fallback_reason"] == "cell_coverage_below_mineru_html"


def test_paddleocr_vl_builder_disables_queue_worker(monkeypatch) -> None:  # noqa: ANN001
    captured: dict = {}

    class _FakePaddleOCRVL:
        def __init__(self, **kwargs):  # noqa: ANN001
            captured.update(kwargs)

    def fake_cached_paddle_object(**kwargs):  # noqa: ANN001
        factory = kwargs["factory"]
        return factory(kwargs["init_kwargs"])

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(PaddleOCRVL=_FakePaddleOCRVL),
    )
    monkeypatch.setattr(
        "platform_foundation.inference.paddle_table._cached_paddle_object",
        fake_cached_paddle_object,
    )
    monkeypatch.setattr(
        "platform_foundation.inference.paddle_table._resolve_paddle_device",
        lambda options: "gpu:0",
    )

    _build_paddleocr_vl({})

    assert captured["use_queues"] is False
    assert captured["vl_rec_max_concurrency"] == 1
    assert captured["device"] == "gpu:0"


def test_paddleocr_vl_predict_defaults_disable_queues_and_limit_tokens(
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _FakePaddleOCRVL:
        def predict(self, input, **kwargs):  # noqa: ANN001
            captured["input"] = input
            captured["kwargs"] = kwargs
            return [
                {
                    "page_index": 0,
                    "seal_res": {
                        "rec_texts": ["demo seal"],
                        "rec_boxes": [[0, 0, 10, 10]],
                    },
                }
            ]

    monkeypatch.setattr(
        paddle_table_module,
        "_build_paddleocr_vl",
        lambda options: _FakePaddleOCRVL(),
    )

    result = PaddleTableStructureService()._extract_with_paddleocr_vl(
        file_uri="/workspace/input/demo.jpg",
        output_dir=tmp_path,
        target_page_sizes={0: ImageSize(width=100, height=100)},
        options={},
        mode="paddleocr_vl",
    )

    assert captured["input"] == "/workspace/input/demo.jpg"
    assert captured["kwargs"]["use_queues"] is False
    assert captured["kwargs"]["max_new_tokens"] == 2048
    assert result.meta["provider"] == "paddleocr_vl"


def test_paddle_auto_uses_ppocr_and_vl_seal_crop(
    monkeypatch,  # noqa: ANN001
    tmp_path: Path,
) -> None:
    ocr_block = TextBlock(
        text="\u5168\u4f53\u6295\u8d44\u4eba\u7b7e\u5b57\uff08\u76d6\u7ae0\uff09\u7f57\u5efa",
        bounding_box=BoundingBox(x=120, y=430, w=360, h=28),
        block_type="text",
        meta={"source": "paddleocr_ppocrv5"},
    )
    vl_block = TextBlock(
        text="\u5fb7\u9633\u5efa\u946b\u5e02\u653f\u8bbe\u65bd\u7ba1\u7406\u6709\u9650\u8d23\u4efb\u516c\u53f8",
        bounding_box=BoundingBox(x=637, y=598, w=209, h=146),
        block_type="seal_text",
        meta={"source": "paddleocr_vl_seal_crop", "image_uri": str(tmp_path / "images" / "seal.jpg")},
    )

    seal_block = TextBlock(
        text="",
        bounding_box=BoundingBox(x=637, y=598, w=209, h=146),
        block_type="seal",
        meta={"img_path": "images/seal.jpg"},
    )

    def fake_predict_layout_crop_text_blocks_with_ppocrv5(**kwargs):  # noqa: ANN001
        return [{"page_index": 0}], {0: (ocr_block,)}, 1

    def fake_predict_seal_crops_with_paddleocr_vl(**kwargs):  # noqa: ANN001
        return {0: (vl_block,)}, 1, [{"page_index": 0, "text": vl_block.text}]

    monkeypatch.setattr(
        paddle_table_module,
        "_predict_layout_crop_text_blocks_with_ppocrv5",
        fake_predict_layout_crop_text_blocks_with_ppocrv5,
    )
    monkeypatch.setattr(
        paddle_table_module,
        "_predict_seal_crops_with_paddleocr_vl",
        fake_predict_seal_crops_with_paddleocr_vl,
    )

    result = PaddleTableStructureService().extract_tables(
        file_uri="/workspace/output/demo.converted.pdf",
        output_dir=tmp_path,
        target_page_sizes={0: ImageSize(width=900, height=1200)},
        options={
            "paddle_table_mode": "auto",
            "source_input_uri": "/workspace/input/demo.jpg",
        },
        page_text_blocks_by_page={0: [seal_block]},
    )

    payload = json.loads((tmp_path / "paddle_table_structure.json").read_text(encoding="utf-8"))
    assert result.meta["provider"] == "paddleocr_ppocrv5_paddleocr_vl_seal"
    assert result.meta["replace_text_blocks"] is True
    assert payload["mode"] == "seal_vl_crops_ppocrv5"
    assert payload["text_blocks_by_page"]["0"][0]["text"] == ocr_block.text
    assert payload["text_blocks_by_page"]["0"][1]["block_type"] == "seal_text"
    assert payload["text_blocks_by_page"]["0"][1]["text"] == vl_block.text


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
    assert table.meta["raw_source"] == "table"
    assert table.meta["raw_block_count"] == 1
    assert table.meta["has_html"] is True
    assert "raw_table" not in table.meta
    assert json.dumps(table.model_dump(mode="json"), ensure_ascii=False).count("<table>") == 1
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


def test_parse_paddle_structure_tables_uses_cell_boxes_as_true_cells() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 200,
                "height": 160,
                "table_res_list": [
                    {
                        "cell_box_list": [
                            [0, 0, 100, 80],
                            [100, 0, 200, 80],
                        ],
                        "table_ocr_pred": {
                            "rec_texts": ["执行", "事务", "合伙人", "2025", "年6", "月19日"],
                            "rec_scores": [0.99, 0.99, 0.99, 0.98, 0.98, 0.98],
                            "rec_boxes": [
                                [8, 10, 32, 30],
                                [34, 10, 58, 30],
                                [60, 10, 92, 30],
                                [112, 10, 138, 30],
                                [140, 10, 162, 30],
                                [164, 10, 194, 30],
                            ],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=200, height=160)},
    )

    table = parsed.tables_by_page[0][0]
    assert len(table.cells) == 2
    assert table.cells[0].text == "执行事务合伙人"
    assert table.cells[0].meta["bbox_source"] == "cell_box_list"
    assert table.cells[0].meta["text_source"] == "table_ocr_pred.fragments"
    assert table.cells[0].meta["ocr_fragment_count"] == 3
    assert table.cells[1].text == "2025年6月19日"
    assert table.cells[1].meta["ocr_fragment_count"] == 3
    assert parsed.meta["cell_count"] == 2


def test_parse_paddle_structure_tables_interleaves_vertical_fragment_columns() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 200,
                "height": 240,
                "table_res_list": [
                    {
                        "cell_box_list": [
                            [0, 0, 80, 140],
                        ],
                        "table_ocr_pred": {
                            "rec_texts": ["ace", "bdf", "gh", "ij", "klm"],
                            "rec_scores": [0.99, 0.99, 0.98, 0.98, 0.98],
                            "rec_boxes": [
                                [10, 10, 34, 82],
                                [36, 10, 60, 82],
                                [18, 84, 52, 106],
                                [14, 108, 56, 130],
                                [10, 132, 62, 154],
                            ],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=200, height=240)},
    )

    table = parsed.tables_by_page[0][0]
    assert len(table.cells) == 1
    assert table.cells[0].text == "abcdefghijklm"


def test_parse_paddle_structure_tables_flips_180_degree_fragment_boxes() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 180,
                "height": 120,
                "table_res_list": [
                    {
                        "cell_box_list": [
                            [0, 0, 60, 40],
                            [60, 0, 120, 40],
                            [120, 0, 180, 40],
                            [0, 40, 60, 120],
                            [60, 40, 120, 120],
                            [120, 40, 180, 120],
                        ],
                        "table_ocr_pred": {
                            "rec_texts": [
                                "\u59d3\u540d",
                                "\u8eab\u4efd\u8bc1\u53f7\u7801",
                                "\u7ed3\u679c",
                                "\u4f55\u4f1f",
                                "510602197808294054",
                                "\u6210\u529f",
                            ],
                            "rec_scores": [0.99] * 6,
                            "rec_boxes": [
                                [120, 80, 180, 120],
                                [60, 80, 120, 120],
                                [0, 80, 60, 120],
                                [120, 0, 180, 40],
                                [60, 0, 120, 40],
                                [0, 0, 60, 40],
                            ],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=180, height=120)},
    )

    table = parsed.tables_by_page[0][0]
    cells = {(cell.row_index, cell.col_index): cell for cell in table.cells}
    assert cells[(0, 0)].text == "\u59d3\u540d"
    assert cells[(0, 1)].text == "\u8eab\u4efd\u8bc1\u53f7\u7801"
    assert cells[(0, 2)].text == "\u7ed3\u679c"
    assert cells[(1, 0)].text == "\u4f55\u4f1f"
    assert cells[(1, 1)].text == "510602197808294054"
    assert cells[(1, 2)].text == "\u6210\u529f"
    assert cells[(0, 0)].meta["ocr_fragment_box_transform"] == "flip_xy"


def test_parse_paddle_structure_tables_flips_when_last_row_is_empty() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 180,
                "height": 180,
                "table_res_list": [
                    {
                        "cell_box_list": [
                            [0, 0, 60, 40],
                            [60, 0, 120, 40],
                            [120, 0, 180, 40],
                            [0, 40, 60, 120],
                            [60, 40, 120, 120],
                            [120, 40, 180, 120],
                            [0, 120, 60, 180],
                            [60, 120, 120, 180],
                            [120, 120, 180, 180],
                        ],
                        "table_ocr_pred": {
                            "rec_texts": [
                                "\u59d3\u540d",
                                "\u8eab\u4efd\u8bc1\u53f7\u7801",
                                "\u7ed3\u679c",
                                "\u4f55\u4f1f",
                                "510602197808294054",
                                "\u6210\u529f",
                            ],
                            "rec_scores": [0.99] * 6,
                            "rec_boxes": [
                                [120, 60, 180, 120],
                                [60, 60, 120, 120],
                                [0, 60, 60, 120],
                                [120, 0, 180, 40],
                                [60, 0, 120, 40],
                                [0, 0, 60, 40],
                            ],
                        },
                    }
                ],
            }
        ],
        target_page_sizes={0: ImageSize(width=180, height=180)},
    )

    table = parsed.tables_by_page[0][0]
    cells = {(cell.row_index, cell.col_index): cell for cell in table.cells}
    assert cells[(0, 0)].text == "\u59d3\u540d"
    assert cells[(0, 1)].text == "\u8eab\u4efd\u8bc1\u53f7\u7801"
    assert cells[(0, 2)].text == "\u7ed3\u679c"
    assert cells[(1, 0)].text == "\u4f55\u4f1f"
    assert cells[(1, 1)].text == "510602197808294054"
    assert cells[(1, 2)].text == "\u6210\u529f"
    assert all(cells[(2, col)].text == "" for col in range(3))


def test_parse_paddle_structure_tables_fills_suspicious_cells_from_page_ocr() -> None:
    parsed = parse_paddle_structure_tables(
        [
            {
                "page_index": 0,
                "width": 180,
                "height": 120,
                "table_res_list": [
                    {
                        "cell_box_list": [
                            [0, 0, 60, 40],
                            [60, 0, 120, 40],
                            [120, 0, 180, 40],
                            [0, 40, 60, 120],
                            [60, 40, 120, 120],
                            [120, 40, 180, 120],
                        ],
                        "table_ocr_pred": {
                            "rec_texts": [
                                "\u59d3\u540d",
                                "\u8eab\u4efd\u8bc1\u53f7\u7801",
                                "\u7ed3\u679c",
                                "",
                                "510602197808294054",
                                "X",
                            ],
                            "rec_scores": [0.99] * 6,
                            "rec_boxes": [
                                [0, 0, 60, 40],
                                [60, 0, 120, 40],
                                [120, 0, 180, 40],
                                [0, 40, 60, 120],
                                [60, 40, 120, 120],
                                [120, 40, 180, 120],
                            ],
                        },
                    }
                ],
                "overall_ocr_res": {
                    "rec_texts": ["\u4f55\u4f1f", "\u6210\u529f"],
                    "rec_boxes": [
                        [5, 45, 55, 80],
                        [125, 45, 175, 80],
                    ],
                    "rec_scores": [0.98, 0.98],
                },
            }
        ],
        target_page_sizes={0: ImageSize(width=180, height=120)},
    )

    table = parsed.tables_by_page[0][0]
    cells = {(cell.row_index, cell.col_index): cell for cell in table.cells}
    assert cells[(1, 0)].text == "\u4f55\u4f1f"
    assert cells[(1, 2)].text == "\u6210\u529f"
    assert cells[(1, 0)].meta["page_text_fallback"] is True
    assert cells[(1, 2)].meta["page_text_fallback"] is True


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


def test_parse_paddle_ocr_text_blocks_extracts_ppocrv5_recognition() -> None:
    blocks_by_page, count = parse_paddle_ocr_text_blocks(
        [
            {
                "res": {
                    "page_index": 0,
                    "width": 100,
                    "height": 100,
                    "rec_texts": ["plain text"],
                    "rec_scores": [0.92],
                    "rec_polys": [[[10, 20], [50, 20], [50, 40], [10, 40]]],
                }
            }
        ],
        target_page_sizes={0: ImageSize(width=200, height=200)},
        provider="paddleocr_ppocrv5",
    )

    assert count == 1
    block = blocks_by_page[0][0]
    assert block.text == "plain text"
    assert block.bounding_box == BoundingBox(x=20, y=40, w=80, h=40)
    assert block.meta["source"] == "paddleocr_ppocrv5"


def test_merge_table_result_can_replace_text_blocks_without_tables() -> None:
    middle_ref = ArtifactRef(kind="middle_json", uri="file:///tmp/middle.json")
    original_block = TextBlock(
        text="mineru old",
        bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
    )
    paddle_block = TextBlock(
        text="paddle new",
        bounding_box=BoundingBox(x=5, y=5, w=20, h=10),
        meta={"source": "paddleocr_ppocrv5"},
    )
    parsed = MinerUDocumentParseResult(
        pages=(
            MinerUPageResult(
                page_index=0,
                text="mineru old",
                text_blocks=(original_block,),
                table_blocks=(),
                image_size=ImageSize(width=100, height=100),
            ),
        ),
        middle_json_ref=middle_ref,
        page_count=1,
    )

    merged = _merge_table_result(
        parsed,
        PaddleTableStructureResult(
            tables_by_page={},
            text_blocks_by_page={0: (paddle_block,)},
            meta={"provider": "paddleocr_ppocrv5", "replace_text_blocks": True},
        ),
        emit_text_blocks=False,
        replace_existing_table_blocks=True,
    )

    assert [block.text for block in merged.pages[0].text_blocks] == ["paddle new"]
    assert merged.pages[0].text == "paddle new"
    assert merged.meta["table_cell_refine"]["provider"] == "paddleocr_ppocrv5"


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
    assert service.options["paddle_table_mode"] == "auto"
    assert service.options["timeout_seconds"] == 1800.0


def test_layout_extract_mineru_paddle_operator_respects_ocr_engine() -> None:
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
        meta={"mineru_options": {"table_engine": "ocr"}},
    )

    list(op.process(ctx, iter([document.model_dump(mode="python")]), path="item"))

    assert service.options is not None
    assert service.options["table_engine"] == "ocr"
    assert service.options["enable_table_cell_refine"] is False
    assert service.options["enable_paddle_table_refine"] is False
    assert "paddle_table_mode" not in service.options


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


def test_cleanup_mineru_api_task_dirs_removes_matching_uuid_upload(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    final_output_dir = output_root / "business" / "doc"
    final_output_dir.mkdir(parents=True)
    input_path = final_output_dir / "doc.converted.pdf"
    input_path.write_bytes(b"%PDF-1.4\nconverted")

    matching_task_dir = output_root / "1b985c48-168c-49bc-ba15-f39598a9d16e"
    matching_upload_dir = matching_task_dir / "uploads"
    matching_upload_dir.mkdir(parents=True)
    (matching_upload_dir / input_path.name).write_bytes(input_path.read_bytes())
    (matching_task_dir / "doc.converted" / "auto").mkdir(parents=True)

    other_task_dir = output_root / "fdbc4967-d456-48bf-bd16-b5b88e3d44f4"
    other_upload_dir = other_task_dir / "uploads"
    other_upload_dir.mkdir(parents=True)
    (other_upload_dir / input_path.name).write_bytes(b"different size")

    cleaned = _cleanup_mineru_api_task_dirs(
        input_path=input_path,
        output_dir=final_output_dir,
        started_at=0,
    )

    assert cleaned == (matching_task_dir,)
    assert not matching_task_dir.exists()
    assert other_task_dir.exists()
    assert final_output_dir.exists()
    assert input_path.exists()


def test_cleanup_mineru_api_task_dirs_only_enabled_for_http_api() -> None:
    assert _should_cleanup_mineru_api_task_dirs(options={}, api_url="http://127.0.0.1:8000")
    assert not _should_cleanup_mineru_api_task_dirs(options={}, api_url=None)
    assert not _should_cleanup_mineru_api_task_dirs(options={}, api_url="")
    assert not _should_cleanup_mineru_api_task_dirs(options={}, api_url="file:///tmp/input.pdf")
    assert not _should_cleanup_mineru_api_task_dirs(
        options={"cleanup_api_task_dirs": False},
        api_url="http://127.0.0.1:8000",
    )


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
