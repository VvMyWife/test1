from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platform_foundation.ocr import DEFAULT_TIMEOUT_SECONDS, extract_pdf_file
from platform_foundation.operators import LayoutExtractMinerUOperator, OperatorError

from ..schemas.mineru_layout import (
    MinerUGeneratedJson,
    MinerULayoutBatchRequest,
    MinerULayoutBatchResult,
    MinerULayoutExtractRequest,
    MinerULayoutPathRequest,
)


class PlatformApiError(Exception):
    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class MinerULayoutService:
    operator_factory: Callable[[], LayoutExtractMinerUOperator] = LayoutExtractMinerUOperator

    def extract_layout_path(self, request: MinerULayoutPathRequest) -> MinerUGeneratedJson:
        return self.extract_layout(
            MinerULayoutExtractRequest(
                file_uri=request.pdf_path,
                output_dir=request.output_dir,
                source_file_name=Path(request.pdf_path).name,
                mineru_options=dict(request.mineru_options),
                timeout_seconds=request.timeout_seconds,
            )
        )

    def extract_layout_batch(self, request: MinerULayoutBatchRequest) -> MinerULayoutBatchResult:
        documents: list[MinerUGeneratedJson] = []
        for document in request.documents:
            documents.append(
                self.extract_layout(
                    MinerULayoutExtractRequest(
                        file_uri=document.file_uri,
                        output_dir=request.output_dir,
                        source_file_name=document.source_file_name,
                        mineru_options=dict(request.mineru_options),
                        timeout_seconds=request.timeout_seconds,
                    )
                )
            )

        return MinerULayoutBatchResult(
            document_count=len(documents),
            output_dir=str(Path(request.output_dir).expanduser().resolve()),
            documents=documents,
        )

    def extract_layout(self, request: MinerULayoutExtractRequest) -> MinerUGeneratedJson:
        source_path = Path(request.file_uri).expanduser().resolve()
        source_file_name = request.source_file_name or source_path.name
        output_root = Path(request.output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            result = extract_pdf_file(
                source_path,
                output_dir=output_root,
                timeout_seconds=request.timeout_seconds or DEFAULT_TIMEOUT_SECONDS,
                mineru_options=dict(request.mineru_options),
                source_file_name=source_file_name,
                operator_factory=self.operator_factory,
            )
        except OperatorError as exc:
            raise self._map_operator_error(exc) from exc

        if not result.success:
            raise self._map_extract_result_error(result.error_code, result.error)

        return MinerUGeneratedJson(
            source_pdf=str(source_path),
            source_file_name=source_file_name,
            json_path=str(result.json_path),
            artifact_dir=result.artifact_dir,
            page_count=result.page_count,
            text_block_count=result.text_block_count,
            table_block_count=result.table_block_count,
            elapsed_seconds=result.elapsed_seconds or 0.0,
        )

    def _map_operator_error(self, error: OperatorError) -> PlatformApiError:
        status_code = {
            "INVALID_DOCUMENT_ITEM": 400,
            "MINERU_INPUT_NOT_FOUND": 400,
            "MINERU_UNSUPPORTED_INPUT": 400,
            "MINERU_COMMAND_NOT_FOUND": 503,
            "MINERU_TIMEOUT": 503,
            "MINERU_COMMAND_FAILED": 502,
            "MINERU_INVALID_CONTENT_LIST": 502,
            "MINERU_INVALID_MIDDLE_JSON": 502,
            "MINERU_OUTPUT_NOT_FOUND": 502,
            "MINERU_AMBIGUOUS_OUTPUT": 502,
            "PADDLE_TABLE_REFINEMENT_FAILED": 502,
            "TABLE_CELL_REFINEMENT_PROVIDER_UNSUPPORTED": 400,
        }.get(error.code, 500)

        message = str(error)
        if error.code == "MINERU_TIMEOUT":
            message = "MinerU timed out while extracting layout"
        elif error.code == "MINERU_INPUT_NOT_FOUND":
            message = "The requested PDF file was not found"
        elif error.code == "MINERU_UNSUPPORTED_INPUT":
            message = "MinerU currently supports only local PDF paths or file:// URIs"
        elif error.code == "MINERU_COMMAND_NOT_FOUND":
            message = "MinerU is not installed or not available in PATH on this host"
        elif error.code == "MINERU_COMMAND_FAILED":
            message = "MinerU failed while parsing the uploaded PDF"
        elif error.code == "PADDLE_TABLE_REFINEMENT_FAILED":
            message = "PaddleOCR failed while refining table cells"
        elif error.code == "TABLE_CELL_REFINEMENT_PROVIDER_UNSUPPORTED":
            message = "The requested table cell refinement provider is not supported"
        elif error.code in {
            "MINERU_INVALID_CONTENT_LIST",
            "MINERU_INVALID_MIDDLE_JSON",
            "MINERU_OUTPUT_NOT_FOUND",
            "MINERU_AMBIGUOUS_OUTPUT",
        }:
            message = "MinerU completed but returned an invalid or incomplete output payload"

        return PlatformApiError(code=error.code, message=message, status_code=status_code)

    def _map_extract_result_error(self, error_code: str | None, error: str | None) -> PlatformApiError:
        return self._map_operator_error(
            OperatorError(
                error or "MinerU failed while extracting layout",
                code=error_code or "MINERU_EXTRACT_FAILED",
                retryable=False,
            )
        )
