from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..contracts import HAS_ARROW, OperatorContext

if HAS_ARROW:  # pragma: no cover
    import pyarrow as pa  # type: ignore


def to_daft_map_partitions_fn(
    op: Any,
    *,
    ctx: OperatorContext,
) -> Callable[[Any], Any]:
    """Wrap an operator into a Daft `map_partitions` style callable.

    Daft's public API can evolve; this adapter intentionally stays minimal and
    operates on Arrow tables/batches when available.
    """

    if not HAS_ARROW:
        raise RuntimeError("pyarrow is required for Daft Arrow adapters")

    def _fn(partition: Any) -> Any:
        if isinstance(partition, pa.Table):
            out_batches = [op.process_arrow_batch(ctx, b) for b in partition.to_batches()]
            return pa.Table.from_batches(out_batches)
        if isinstance(partition, pa.RecordBatch):
            return op.process_arrow_batch(ctx, partition)
        raise TypeError(f"Unsupported Daft partition type: {type(partition)!r}")

    return _fn

