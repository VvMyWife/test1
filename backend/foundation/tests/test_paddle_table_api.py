from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from platform_foundation.ocr import paddle_table_api  # noqa: E402


def test_paddle_table_api_health_reports_server_side_extract_limit() -> None:
    payload = paddle_table_api.health()

    assert payload["status"] == "healthy"
    assert "max_concurrent_extracts" in payload
    assert payload["active_extracts"] == 0
