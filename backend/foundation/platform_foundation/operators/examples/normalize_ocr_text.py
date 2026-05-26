from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..base import BaseOperator
from ..contracts import Item, OperatorContext


_whitespace_re = re.compile(r"\s+")


def normalize_and_lightly_denoise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = "".join(ch for ch in t if ch.isprintable() or ch in "\n\t")
    t = _whitespace_re.sub(" ", t).strip()
    t = t.strip("·•●◆■★☆※*~`'\"|")
    return t


@dataclass(frozen=True)
class NormalizeOcrTextConfig:
    input_key: str = "text"
    output_key: str = "text"


@dataclass(frozen=True)
class NormalizeOcrTextOperator(BaseOperator):
    op_name: str = "normalize_ocr_text"
    op_version: str = "0.1.0"

    config: NormalizeOcrTextConfig = NormalizeOcrTextConfig()

    def process_item(self, ctx: OperatorContext, item: Item) -> Item:
        raw = item.get(self.config.input_key, "")
        text = raw if isinstance(raw, str) else ""
        cleaned = normalize_and_lightly_denoise(text)

        if ctx.logger is not None:
            ctx.logger.debug(
                "normalize_ocr_text",
                extra={
                    "trace_id": ctx.trace_id,
                    "run_id": ctx.run_id,
                    "op_name": self.op_name,
                    "op_version": self.op_version,
                },
            )

        out = dict(item)
        out[self.config.output_key] = cleaned
        return out

    @classmethod
    def from_config(cls, config: dict) -> "NormalizeOcrTextOperator":
        # config example:
        # {"config": {"input_key": "text", "output_key": "clean_text"}}
        cfg = config.get("config") or {}
        return cls(config=NormalizeOcrTextConfig(**cfg))

