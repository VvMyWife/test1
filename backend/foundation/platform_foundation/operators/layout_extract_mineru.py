from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, ClassVar, Iterator

from pydantic import ValidationError

from ..contracts import DocumentItem, PageItem
from ..inference import MinerUCliDocumentService, MinerUDocumentService, MinerUServiceError
from .base import BaseOperator
from .contracts import Item, Logger, OperatorContext, OperatorError, OperatorSchema


_LOGGER = logging.getLogger(__name__)
_DEFAULT_FAILURE_WORKSPACE = "."
_SENSITIVE_OPTION_PARTS = ("api_key", "authorization", "password", "secret", "token")


@dataclass(frozen=True)
class LayoutExtractMinerUOperator(BaseOperator):
    op_name: str = "layout_extract_mineru"
    op_version: str = "0.1.0"

    service: MinerUDocumentService = MinerUCliDocumentService()
    default_timeout_seconds: float = 1800.0
    max_inflight: int | None = None
    concurrency_acquire_timeout_seconds: float | None = None
    recommended_max_concurrency: int | None = None
    failure_record_dir: str | None = None

    _semaphores: ClassVar[dict[tuple[str, int], threading.BoundedSemaphore]] = {}
    _semaphores_lock: ClassVar[threading.Lock] = threading.Lock()

    def input_schema(self) -> OperatorSchema:
        return OperatorSchema.item(DocumentItem, description="Document-level PDF input")

    def output_schema(self) -> OperatorSchema:
        return OperatorSchema.item(PageItem, description="Per-page output enriched by MinerU")

    def process_item(self, ctx: OperatorContext, item: Item) -> Item | list[Item]:
        try:
            document = DocumentItem.model_validate(dict(item))
        except ValidationError as exc:
            raise OperatorError(
                "layout_extract_mineru expects a valid DocumentItem input",
                code="INVALID_DOCUMENT_ITEM",
                retryable=False,
                details={"errors": exc.errors(include_url=False)},
            ) from exc

        logger = ctx.logger or _LOGGER
        service_options = self._build_service_options(ctx, document)
        timeout_seconds = self._resolve_timeout_seconds(service_options)
        requested_concurrency = self._extract_requested_concurrency(service_options)
        safety_meta = self._build_safety_meta(
            timeout_seconds=timeout_seconds,
            requested_concurrency=requested_concurrency,
        )
        if (
            requested_concurrency is not None
            and self.recommended_max_concurrency is not None
            and requested_concurrency > self.recommended_max_concurrency
        ):
            self._log(
                logger,
                "warning",
                (
                    f"{self.op_name} requested_concurrency={requested_concurrency} "
                    f"exceeds recommended_max_concurrency={self.recommended_max_concurrency}; "
                    f"operator max_inflight={self.max_inflight}"
                ),
            )

        start = time.perf_counter()
        self._log(
            logger,
            "info",
            (
                f"{self.op_name} start doc_id={document.doc_id} trace_id={ctx.trace_id} "
                f"timeout_seconds={timeout_seconds} max_inflight={self.max_inflight}"
            ),
        )
        try:
            with self._inflight_slot(ctx, document, service_options, logger):
                parsed = self.service.parse_document(
                    file_uri=document.file_uri,
                    mime_type=document.mime_type,
                    options=service_options,
                )
        except MinerUServiceError as exc:
            failure_record = self._write_failure_record(
                ctx,
                document=document,
                service_options=service_options,
                error=exc,
                code=exc.code,
                retryable=exc.retryable,
                details=exc.details,
                logger=logger,
            )
            details = dict(exc.details)
            if failure_record is not None:
                details["failure_record"] = failure_record
            self._log(
                logger,
                "error",
                (
                    f"{self.op_name} failed doc_id={document.doc_id} trace_id={ctx.trace_id} "
                    f"code={exc.code} retryable={exc.retryable} failure_record={failure_record}"
                ),
            )
            raise OperatorError(
                str(exc),
                code=exc.code,
                retryable=exc.retryable,
                details=details,
            ) from exc
        except OperatorError:
            raise
        except Exception as exc:
            failure_record = self._write_failure_record(
                ctx,
                document=document,
                service_options=service_options,
                error=exc,
                code="MINERU_OPERATOR_UNEXPECTED_ERROR",
                retryable=False,
                details={"error_type": type(exc).__name__},
                logger=logger,
            )
            details: dict[str, Any] = {"error_type": type(exc).__name__}
            if failure_record is not None:
                details["failure_record"] = failure_record
            self._log(
                logger,
                "error",
                (
                    f"{self.op_name} unexpected_error doc_id={document.doc_id} "
                    f"trace_id={ctx.trace_id} error_type={type(exc).__name__} "
                    f"failure_record={failure_record}"
                ),
            )
            raise OperatorError(
                "Unexpected error while extracting MinerU layout",
                code="MINERU_OPERATOR_UNEXPECTED_ERROR",
                retryable=False,
                details=details,
            ) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._log(
            logger,
            "info",
            (
                f"{self.op_name} success doc_id={document.doc_id} trace_id={ctx.trace_id} "
                f"pages={parsed.page_count} elapsed_ms={elapsed_ms:.2f}"
            ),
        )

        def emit_pages() -> list[Item]:
            page_items: list[Item] = []
            for page in parsed.pages:
                page_meta: dict[str, Any] = dict(page.page_meta)
                page_meta.setdefault("layout_provider", "mineru")
                page_meta.setdefault("coord_space", parsed.coord_space.value)
                page_meta.setdefault("page_count", parsed.page_count)
                if ctx.config_version is not None:
                    page_meta.setdefault("config_version", ctx.config_version)
                if parsed.meta.get("parser_version") is not None:
                    page_meta.setdefault("parser_version", parsed.meta["parser_version"])
                if page.image_size is not None:
                    page_meta.setdefault("image_size", page.image_size.model_dump(mode="python"))
                if parsed.artifacts:
                    page_meta.setdefault(
                        "mineru_artifacts",
                        [artifact.model_dump(mode="python") for artifact in parsed.artifacts],
                    )
                page_meta.setdefault("operator_safety", dict(safety_meta))

                page_item = PageItem(
                    archive_id=document.archive_id,
                    archive_owner_user_id=document.archive_owner_user_id,
                    triggered_by_user_id=document.triggered_by_user_id,
                    doc_id=document.doc_id,
                    page_index=page.page_index,
                    text=page.text,
                    text_blocks=list(page.text_blocks),
                    table_blocks=list(page.table_blocks),
                    page_meta=page_meta,
                    layout_ref=parsed.middle_json_ref,
                )
                page_items.append(page_item.model_dump(mode="python"))
            return page_items

        return emit_pages()

    def _build_service_options(
        self, ctx: OperatorContext, document: DocumentItem
    ) -> dict[str, Any]:
        service_options: dict[str, Any] = {}
        raw_service_options = document.meta.get("mineru_options")
        if isinstance(raw_service_options, Mapping):
            service_options.update(raw_service_options)
        if ctx.config_version is not None:
            service_options.setdefault("config_version", ctx.config_version)
        service_options.setdefault("timeout_seconds", self.default_timeout_seconds)
        return service_options

    def _resolve_timeout_seconds(self, service_options: dict[str, Any]) -> float:
        timeout_seconds = self._coerce_positive_float(
            service_options.get("timeout_seconds"),
            option_name="timeout_seconds",
        )
        service_options["timeout_seconds"] = timeout_seconds
        return timeout_seconds

    def _coerce_positive_float(self, value: Any, *, option_name: str) -> float:
        if isinstance(value, bool):
            raise OperatorError(
                f"{option_name} must be a positive number",
                code="INVALID_OPERATOR_OPTION",
                retryable=False,
                details={"option": option_name, "value": value},
            )
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise OperatorError(
                f"{option_name} must be a positive number",
                code="INVALID_OPERATOR_OPTION",
                retryable=False,
                details={"option": option_name, "value": value},
            ) from exc
        if parsed <= 0:
            raise OperatorError(
                f"{option_name} must be a positive number",
                code="INVALID_OPERATOR_OPTION",
                retryable=False,
                details={"option": option_name, "value": value},
            )
        return parsed

    def _extract_requested_concurrency(self, service_options: Mapping[str, Any]) -> int | None:
        for key in ("caller_concurrency", "max_concurrency", "concurrency"):
            value = service_options.get(key)
            if value is None or isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return None

    def _build_safety_meta(
        self,
        *,
        timeout_seconds: float,
        requested_concurrency: int | None,
    ) -> dict[str, Any]:
        safety_meta: dict[str, Any] = {
            "timeout_seconds": timeout_seconds,
            "max_inflight": self.max_inflight,
            "concurrency_acquire_timeout_seconds": self.concurrency_acquire_timeout_seconds,
            "recommended_max_concurrency": self.recommended_max_concurrency,
            "failure_record_enabled": True,
        }
        if requested_concurrency is not None:
            safety_meta["requested_concurrency"] = requested_concurrency
            if self.recommended_max_concurrency is not None:
                safety_meta["requested_concurrency_exceeds_recommendation"] = (
                    requested_concurrency > self.recommended_max_concurrency
                )
        return safety_meta

    @contextmanager
    def _inflight_slot(
        self,
        ctx: OperatorContext,
        document: DocumentItem,
        service_options: Mapping[str, Any],
        logger: Logger | logging.Logger,
    ) -> Iterator[None]:
        if self.max_inflight is None:
            yield
            return
        if self.max_inflight <= 0:
            raise OperatorError(
                "max_inflight must be positive or None",
                code="INVALID_OPERATOR_OPTION",
                retryable=False,
                details={"option": "max_inflight", "value": self.max_inflight},
            )

        semaphore = self._get_inflight_semaphore(self.max_inflight)
        acquired = (
            semaphore.acquire()
            if self.concurrency_acquire_timeout_seconds is None
            else semaphore.acquire(timeout=self.concurrency_acquire_timeout_seconds)
        )
        if not acquired:
            error = OperatorError(
                "Timed out while waiting for MinerU operator concurrency slot",
                code="MINERU_OPERATOR_CONCURRENCY_LIMIT",
                retryable=True,
                details={
                    "max_inflight": self.max_inflight,
                    "concurrency_acquire_timeout_seconds": self.concurrency_acquire_timeout_seconds,
                },
            )
            failure_record = self._write_failure_record(
                ctx,
                document=document,
                service_options=service_options,
                error=error,
                code=error.code,
                retryable=error.retryable,
                details=error.details,
                logger=logger,
            )
            if failure_record is not None:
                error.details["failure_record"] = failure_record
            raise error

        try:
            yield
        finally:
            semaphore.release()

    def _get_inflight_semaphore(self, limit: int) -> threading.BoundedSemaphore:
        key = (self.op_name, limit)
        with self._semaphores_lock:
            semaphore = self._semaphores.get(key)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(limit)
                self._semaphores[key] = semaphore
            return semaphore

    def _write_failure_record(
        self,
        ctx: OperatorContext,
        *,
        document: DocumentItem,
        service_options: Mapping[str, Any],
        error: BaseException,
        code: str,
        retryable: bool,
        details: Mapping[str, Any] | None,
        logger: Logger | logging.Logger,
    ) -> str | None:
        try:
            path = self._failure_record_path(ctx, document, service_options)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "op_name": self.op_name,
                "op_version": self.op_version,
                "context": {
                    "trace_id": ctx.trace_id,
                    "run_id": ctx.run_id,
                    "batch_id": ctx.batch_id,
                    "item_id": ctx.item_id,
                    "tenant_id": ctx.tenant_id,
                    "config_version": ctx.config_version,
                    "tags": dict(ctx.tags),
                },
                "document": {
                    "archive_id": document.archive_id,
                    "archive_owner_user_id": document.archive_owner_user_id,
                    "triggered_by_user_id": document.triggered_by_user_id,
                    "doc_id": document.doc_id,
                    "file_uri": document.file_uri,
                    "mime_type": document.mime_type,
                    "file_hash": document.file_hash,
                    "num_pages": document.num_pages,
                },
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "code": code,
                    "retryable": retryable,
                    "details": _json_safe(dict(details or {})),
                },
                "service_options": self._sanitize_options(service_options),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            return str(path)
        except Exception as exc:  # pragma: no cover - failure logging must be best effort
            self._log(
                logger,
                "warning",
                (
                    f"{self.op_name} could_not_write_failure_record doc_id={document.doc_id} "
                    f"error_type={type(exc).__name__} error={exc}"
                ),
            )
            return None

    def _failure_record_path(
        self,
        ctx: OperatorContext,
        document: DocumentItem,
        service_options: Mapping[str, Any],
    ) -> Path:
        output_dir = self._non_empty_str(service_options.get("output_dir"))
        if output_dir is not None:
            return Path(output_dir).expanduser().resolve() / "operate_error.json"

        configured_dir = (
            self._non_empty_str(service_options.get("failure_record_dir"))
            or self.failure_record_dir
            or os.environ.get("MINERU_OPERATOR_FAILURE_DIR")
        )
        if configured_dir is None:
            workspace = (
                os.environ.get("MINERU_WORKSPACE")
                or os.environ.get("WORKSPACE")
                or os.environ.get("WORKSPACE_ROOT")
                or _DEFAULT_FAILURE_WORKSPACE
            )
            configured_dir = str(Path(workspace) / "logs" / "operator_failures")

        safe_doc_id = _safe_filename(document.doc_id)
        safe_trace_id = _safe_filename(ctx.trace_id)
        return (
            Path(configured_dir).expanduser().resolve()
            / f"{safe_doc_id}-{safe_trace_id}-operate_error.json"
        )

    def _sanitize_options(self, service_options: Mapping[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in service_options.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_OPTION_PARTS):
                sanitized[key_text] = "***"
            else:
                sanitized[key_text] = _json_safe(value)
        return sanitized

    def _non_empty_str(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    def _log(self, logger: Logger | logging.Logger, level: str, message: str) -> None:
        try:
            getattr(logger, level)(message)
        except Exception:  # pragma: no cover
            return


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unknown"
