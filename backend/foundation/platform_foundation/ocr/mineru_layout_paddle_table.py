from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import DocumentItem
from ..operators import LayoutExtractMinerUPaddleTableOperator
from .mineru_layout import MinerULayoutOperateResult, operate as _operate_mineru_layout


def operate(
    document: DocumentItem | Mapping[str, Any],
    *,
    trace_id: str | None = None,
    run_id: str | None = None,
    config_version: str | None = None,
    tags: Mapping[str, str] | None = None,
    mineru_options: Mapping[str, Any] | None = None,
    operator_factory: Callable[[], LayoutExtractMinerUPaddleTableOperator] = (
        LayoutExtractMinerUPaddleTableOperator
    ),
) -> MinerULayoutOperateResult:
    """Extract OCR/layout data and refine detected tables into cells.

    This enhanced entrypoint keeps the original MinerU output contract, but
    defaults to PaddleOCR table refinement when MinerU detects table regions.
    Paddle models are cached in-process by the inference adapter, so batch
    callers should reuse one Python process or platform API worker.
    """

    return _operate_mineru_layout(
        document,
        trace_id=trace_id,
        run_id=run_id,
        config_version=config_version,
        tags=tags,
        mineru_options=mineru_options,
        operator_factory=operator_factory,
    )
