from __future__ import annotations

from functools import lru_cache

from platform_foundation.inference import MinerUCliDocumentService
from platform_foundation.operators import LayoutExtractMinerUPaddleTableOperator

from ..config import load_settings_from_env
from ..services.mineru_layout_service import MinerULayoutService


@lru_cache(maxsize=1)
def get_mineru_layout_service() -> MinerULayoutService:
    settings = load_settings_from_env()
    mineru_document_service = MinerUCliDocumentService(config=settings.mineru_cli)
    return MinerULayoutService(
        operator_factory=lambda: LayoutExtractMinerUPaddleTableOperator(service=mineru_document_service)
    )
