from __future__ import annotations

from app.config import load_settings_from_env


def test_load_settings_from_env_keeps_portable_defaults() -> None:
    settings = load_settings_from_env({})

    assert settings.workspace_root == "/home/kaifang/mineru_workspace"
    assert settings.data_root == "/home/kaifang/mineru_workspace/data"
    assert settings.log_root == "/home/kaifang/mineru_workspace/logs"
    assert settings.upload_temp_root == "/home/kaifang/mineru_workspace/data"
    assert settings.mineru_cli.command == ("mineru",)
    assert settings.mineru_cli.output_root == "/home/kaifang/mineru_workspace/data"
    assert settings.mineru_cli.api_url == "http://127.0.0.1:8000"
    assert settings.mineru_cli.extra_args == ()


def test_load_settings_from_env_builds_mineru_cli_config() -> None:
    settings = load_settings_from_env(
        {
            "PLATFORM_WORKSPACE_ROOT": "/srv/workspace",
            "PLATFORM_DATA_ROOT": "/srv/workspace/data",
            "PLATFORM_LOG_ROOT": "/srv/workspace/logs",
            "PLATFORM_UPLOAD_TEMP_ROOT": "/srv/workspace/data",
            "MINERU_COMMAND": "/opt/mineru/bin/mineru",
            "MINERU_OUTPUT_ROOT": "/var/lib/kf-platform/mineru-output",
            "MINERU_PARSE_METHOD": "auto",
            "MINERU_BACKEND": "pipeline",
            "MINERU_LANG": "ch",
            "MINERU_API_URL": "http://127.0.0.1:9000",
            "MINERU_TIMEOUT_SECONDS": "120",
            "MINERU_EXTRA_ARGS": "--device cuda",
        }
    )

    assert settings.workspace_root == "/srv/workspace"
    assert settings.data_root == "/srv/workspace/data"
    assert settings.log_root == "/srv/workspace/logs"
    assert settings.upload_temp_root == "/srv/workspace/data"
    assert settings.mineru_cli.command == ("/opt/mineru/bin/mineru",)
    assert settings.mineru_cli.output_root == "/var/lib/kf-platform/mineru-output"
    assert settings.mineru_cli.parse_method == "auto"
    assert settings.mineru_cli.backend == "pipeline"
    assert settings.mineru_cli.lang == "ch"
    assert settings.mineru_cli.api_url == "http://127.0.0.1:9000"
    assert settings.mineru_cli.timeout_seconds == 120.0
    assert settings.mineru_cli.extra_args == ("--device", "cuda")
