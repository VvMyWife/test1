from __future__ import annotations

from platform_foundation.operators import OperatorContext
from platform_foundation.operators.examples.normalize_ocr_text import NormalizeOcrTextOperator


def test_normalize_ocr_text_item_path() -> None:
    op = NormalizeOcrTextOperator()
    ctx = OperatorContext(trace_id="t1", run_id="r1")
    items = iter([{"text": "  ＡＢＣ　 123  \n"}, {"text": "·foo  \tbar·"}])

    out = list(op.process(ctx, items, path="item"))
    assert out[0]["text"] == "ABC 123"
    assert out[1]["text"] == "foo bar"

