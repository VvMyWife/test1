from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import shlex

from platform_foundation.inference import MinerUCliConfig

DEFAULT_WORKSPACE_ROOT = "/home/kaifang/mineru_workspace"
DEFAULT_DATA_ROOT = f"{DEFAULT_WORKSPACE_ROOT}/data"
DEFAULT_LOG_ROOT = f"{DEFAULT_WORKSPACE_ROOT}/logs"
DEFAULT_MINERU_API_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class PlatformApiSettings:
    workspace_root: str
    data_root: str
    log_root: str
    upload_temp_root: str
    mineru_cli: MinerUCliConfig


def load_settings_from_env(environ: Mapping[str, str] | None = None) -> PlatformApiSettings:
    env = os.environ if environ is None else environ
    workspace_root = _blank_to_none(env.get("PLATFORM_WORKSPACE_ROOT")) or DEFAULT_WORKSPACE_ROOT
    data_root = _blank_to_none(env.get("PLATFORM_DATA_ROOT")) or DEFAULT_DATA_ROOT
    log_root = _blank_to_none(env.get("PLATFORM_LOG_ROOT")) or DEFAULT_LOG_ROOT
    upload_temp_root = _blank_to_none(env.get("PLATFORM_UPLOAD_TEMP_ROOT")) or data_root
    return PlatformApiSettings(
        workspace_root=workspace_root,
        data_root=data_root,
        log_root=log_root,
        upload_temp_root=upload_temp_root,
        mineru_cli=MinerUCliConfig(
            command=_split_words(env.get("MINERU_COMMAND"), default=("mineru",)),
            output_root=_blank_to_none(env.get("MINERU_OUTPUT_ROOT")) or data_root,
            parse_method=_blank_to_none(env.get("MINERU_PARSE_METHOD")),
            backend=_blank_to_none(env.get("MINERU_BACKEND")),
            lang=_blank_to_none(env.get("MINERU_LANG")),
            api_url=_blank_to_none(env.get("MINERU_API_URL")) or DEFAULT_MINERU_API_URL,
            timeout_seconds=_coerce_float(env.get("MINERU_TIMEOUT_SECONDS")),
            extra_args=_split_words(env.get("MINERU_EXTRA_ARGS"), default=()),
        )
    )


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _split_words(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    text = _blank_to_none(value)
    if text is None:
        return default
    return tuple(shlex.split(text))


def _coerce_float(value: str | None) -> float | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None
