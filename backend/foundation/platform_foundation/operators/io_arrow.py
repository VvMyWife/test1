from __future__ import annotations

from typing import Any, Dict, Iterator, List

from .contracts import HAS_ARROW, Item, ItemStream, OperatorError

if HAS_ARROW:  # pragma: no cover
    import pyarrow as pa  # type: ignore


def require_arrow() -> None:
    if not HAS_ARROW:
        raise OperatorError(
            "pyarrow is not installed; Arrow utilities are unavailable",
            code="ARROW_NOT_INSTALLED",
            retryable=False,
        )


def items_to_recordbatches(
    items: ItemStream,
    *,
    schema: "pa.Schema",
    batch_size: int = 1024,
) -> Iterator["pa.RecordBatch"]:
    require_arrow()
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    buf: List[Dict[str, Any]] = []
    for item in items:
        buf.append(dict(item))
        if len(buf) >= batch_size:
            yield pa.RecordBatch.from_pylist(buf, schema=schema)
            buf.clear()
    if buf:
        yield pa.RecordBatch.from_pylist(buf, schema=schema)


def recordbatch_to_items(batch: "pa.RecordBatch") -> Iterator[Item]:
    require_arrow()
    # This converts Arrow vectors to Python objects (not zero-copy). Use only when necessary.
    for row in batch.to_pylist():
        yield row


def table_to_recordbatches(table: "pa.Table") -> Iterator["pa.RecordBatch"]:
    require_arrow()
    for batch in table.to_batches():
        yield batch

