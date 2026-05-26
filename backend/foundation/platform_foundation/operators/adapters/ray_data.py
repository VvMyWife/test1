from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..contracts import HAS_ARROW, OperatorContext

if HAS_ARROW:  # pragma: no cover
    import pyarrow as pa  # type: ignore


def to_ray_map_batches_fn(
    op: Any,
    *,
    ctx: OperatorContext,
) -> Callable[[Any], Any]:
    """Wrap an operator into a Ray Data `map_batches` callable.

    Notes:
    - This function does not import ray at module import time.
    - Ray Data may pass batches as `pyarrow.Table`, `pyarrow.RecordBatch`, or pandas DataFrame
      depending on dataset format. Here we focus on Arrow and raise on unsupported types.
    """

    if not HAS_ARROW:
        raise RuntimeError("pyarrow is required for Ray Data Arrow adapters")

    def _fn(batch: Any) -> Any:
        if isinstance(batch, pa.Table):
            batches = batch.to_batches()
            out_batches = [op.process_arrow_batch(ctx, b) for b in batches]
            return pa.Table.from_batches(out_batches)
        if isinstance(batch, pa.RecordBatch):
            return op.process_arrow_batch(ctx, batch)
        raise TypeError(f"Unsupported Ray batch type: {type(batch)!r}")

    return _fn

