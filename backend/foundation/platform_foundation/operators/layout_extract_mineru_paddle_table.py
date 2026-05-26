from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..contracts import DocumentItem
from ..inference import MinerUCliDocumentService, MinerUDocumentService
from .contracts import Item, OperatorContext, OperatorError
from .layout_extract_mineru import LayoutExtractMinerUOperator


@dataclass(frozen=True)
class LayoutExtractMinerUPaddleTableOperator(LayoutExtractMinerUOperator):
    """MinerU layout extraction plus PaddleOCR table-cell refinement.

    The base MinerU operator remains the stable single-PDF OCR primitive. This
    enhanced operator injects table refinement defaults, while still delegating
    the actual fan-out and error mapping to the base operator.
    """

    op_name: str = "layout_extract_mineru_paddle_table"
    op_version: str = "0.3.0"

    service: MinerUDocumentService = MinerUCliDocumentService()

    def process_item(self, ctx: OperatorContext, item: Item) -> Item | list[Item]:
        try:
            document = DocumentItem.model_validate(dict(item))
        except ValidationError as exc:
            raise OperatorError(
                "layout_extract_mineru_paddle_table expects a valid DocumentItem input",
                code="INVALID_DOCUMENT_ITEM",
                retryable=False,
                details={"errors": exc.errors(include_url=False)},
            ) from exc

        meta = dict(document.meta)
        mineru_options: dict[str, Any] = {}
        raw_options = meta.get("mineru_options")
        if isinstance(raw_options, Mapping):
            mineru_options.update(raw_options)

        table_engine = str(mineru_options.get("table_engine") or "paddle").strip().lower()
        if table_engine == "ocr":
            mineru_options.setdefault("enable_table_cell_refine", False)
            mineru_options.setdefault("enable_paddle_table_refine", False)
        else:
            mineru_options.setdefault("table_engine", "paddle")
            mineru_options.setdefault("enable_table_cell_refine", True)
            mineru_options.setdefault("enable_paddle_table_refine", True)
            mineru_options.setdefault("table_cell_refine_fail_open", True)
            mineru_options.setdefault("paddle_table_mode", "ppstructurev3")
        mineru_options.setdefault("table_cell_refine_when_tables_present", True)
        mineru_options.setdefault("emit_table_cells_as_text_blocks", False)
        mineru_options.setdefault("paddle_table_structure_init_kwargs", {"model_name": "SLANet_plus"})

        meta["mineru_options"] = mineru_options
        enhanced_document = document.model_copy(update={"meta": meta})
        return super().process_item(ctx, enhanced_document.model_dump(mode="python"))
