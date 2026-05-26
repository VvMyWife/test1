"""OCR/layout extraction entrypoints.

This package exposes importable foundation capabilities. Web services and batch
engines should call these functions instead of depending on a FastAPI ``main``.
"""

from .mineru_layout import MinerULayoutOperateResult, operate
from .mineru_layout_paddle_table import operate as operate_with_paddle_tables
from .pdf_extract import (
    MinerUPdfDirBatchOperator,
    MinerUPdfFileOperator,
    PdfDirExtractReport,
    PdfFileExtractResult,
    extract_pdf_dir,
    extract_pdf_file,
)
from .pure_mineru import (
    DEFAULT_MINERU_API_URL,
    DEFAULT_TIMEOUT_SECONDS,
    MinerUPdfPage,
    MinerUPdfResult,
    dump_pure_mineru_json,
    extract_pdf,
)

__all__ = [
    "DEFAULT_MINERU_API_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "MinerUPdfDirBatchOperator",
    "MinerUPdfFileOperator",
    "MinerULayoutOperateResult",
    "MinerUPdfPage",
    "MinerUPdfResult",
    "PdfDirExtractReport",
    "PdfFileExtractResult",
    "dump_pure_mineru_json",
    "extract_pdf",
    "extract_pdf_dir",
    "extract_pdf_file",
    "operate",
    "operate_with_paddle_tables",
]
