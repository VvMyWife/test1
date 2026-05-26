from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_foundation_on_path() -> None:
    for foundation_root in _foundation_root_candidates():
        if not (foundation_root / "platform_foundation").exists():
            continue

        foundation_root_str = str(foundation_root)
        if foundation_root_str not in sys.path:
            sys.path.insert(0, foundation_root_str)
        return


def _foundation_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured_root = os.getenv("PLATFORM_FOUNDATION_ROOT")
    if configured_root and configured_root.strip():
        candidates.append(Path(configured_root).expanduser().resolve())

    candidates.append(Path(__file__).resolve().parents[3] / "foundation")
    return candidates
