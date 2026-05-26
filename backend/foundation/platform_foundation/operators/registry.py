from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Type

from .base import BaseOperator

_lock = threading.Lock()
_inproc_registry: Dict[tuple[str, str], Type[BaseOperator]] = {}


@dataclass(frozen=True)
class OperatorKey:
    op_name: str
    op_version: str


def register_operator(op_cls: Type[BaseOperator]) -> None:
    key = (op_cls.op_name, op_cls.op_version)
    with _lock:
        _inproc_registry[key] = op_cls


@lru_cache(maxsize=1)
def _entry_point_registry() -> Dict[tuple[str, str], Type[BaseOperator]]:
    # Static discovery via Python packaging entry points.
    # This avoids "forgot to import so it isn't registered" pitfalls in distributed workers.
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover
        return {}

    eps = entry_points()
    group = eps.select(group="platform_foundation.operators")  # type: ignore[attr-defined]

    discovered: Dict[tuple[str, str], Type[BaseOperator]] = {}
    for ep in group:
        obj = ep.load()
        if not isinstance(obj, type) or not issubclass(obj, BaseOperator):
            continue
        discovered[(obj.op_name, obj.op_version)] = obj
    return discovered


def get_operator_cls(op_name: str, op_version: str) -> Type[BaseOperator]:
    key = (op_name, op_version)

    # 1) Prefer statically discovered operators.
    ep_map = _entry_point_registry()
    if key in ep_map:
        return ep_map[key]

    # 2) Fall back to in-process registration (tests/local).
    with _lock:
        op_cls = _inproc_registry.get(key)
    if op_cls is not None:
        return op_cls

    raise KeyError(f"Operator not found: {op_name}@{op_version}")

