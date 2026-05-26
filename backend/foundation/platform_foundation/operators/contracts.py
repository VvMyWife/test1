from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    runtime_checkable,
)

try:
    import pyarrow as pa  # type: ignore

    HAS_ARROW = True
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    HAS_ARROW = False

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore


JsonDict = Dict[str, Any]
Item = Mapping[str, Any]
ItemStream = Iterator[Item]

if HAS_ARROW:
    ArrowBatch = "pa.RecordBatch"
    ArrowStream = Iterator["pa.RecordBatch"]
else:  # pragma: no cover
    ArrowBatch = Any
    ArrowStream = Any


@runtime_checkable
class Logger(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class OperatorContext:
    trace_id: str
    run_id: str
    batch_id: Optional[str] = None
    item_id: Optional[str] = None
    tenant_id: Optional[str] = None
    config_version: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logger: Optional[Logger] = None


class OperatorError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "OPERATOR_ERROR",
        retryable: bool = False,
        details: Optional[JsonDict] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff: Literal["fixed", "exponential"] = "exponential"
    base_delay_seconds: float = 1.0
    max_delay_seconds: Optional[float] = None
    jitter: bool = True


OperatorSchemaKind = Literal["item", "arrow"]


@dataclass(frozen=True)
class OperatorSchema:
    kind: OperatorSchemaKind
    schema: Any
    description: Optional[str] = None

    @staticmethod
    def item(model: Type[BaseModel], *, description: Optional[str] = None) -> "OperatorSchema":
        return OperatorSchema(kind="item", schema=model, description=description)

    @staticmethod
    def arrow(schema: "pa.Schema", *, description: Optional[str] = None) -> "OperatorSchema":
        if not HAS_ARROW:  # pragma: no cover
            raise RuntimeError("pyarrow is not installed; cannot create Arrow OperatorSchema")
        return OperatorSchema(kind="arrow", schema=schema, description=description)


TransformKind = Literal["none", "item_to_arrow", "arrow_to_item", "custom"]


@dataclass(frozen=True)
class TransformPlan:
    kind: TransformKind
    description: str = ""
    transform: Optional[Callable[[Any], Any]] = None


T = TypeVar("T")


def iter_chunks(seq: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]

