#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${PLATFORM_WORKSPACE_ROOT:-/home/liujiacheng/mineru_workspace}"
CONDA_SH="${CONDA_SH:-/home/liujiacheng/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mineru312}"

if [ -f "$CONDA_SH" ]; then
  # Non-interactive SSH sessions do not load .bashrc, so load conda explicitly.
  # shellcheck disable=SC1090
  . "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
fi

cd "$WORKSPACE_ROOT/backend/services/platform-api"

export PLATFORM_WORKSPACE_ROOT="$WORKSPACE_ROOT"
export PLATFORM_DATA_ROOT="${PLATFORM_DATA_ROOT:-$WORKSPACE_ROOT/data}"
export PLATFORM_LOG_ROOT="${PLATFORM_LOG_ROOT:-$WORKSPACE_ROOT/logs}"
export PLATFORM_UPLOAD_TEMP_ROOT="${PLATFORM_UPLOAD_TEMP_ROOT:-$WORKSPACE_ROOT/data}"
export PLATFORM_FOUNDATION_ROOT="${PLATFORM_FOUNDATION_ROOT:-$WORKSPACE_ROOT/backend/foundation}"

export MINERU_COMMAND="${MINERU_COMMAND:-mineru}"
export MINERU_OUTPUT_ROOT="${MINERU_OUTPUT_ROOT:-$WORKSPACE_ROOT/data}"
export MINERU_PARSE_METHOD="${MINERU_PARSE_METHOD:-auto}"
export MINERU_BACKEND="${MINERU_BACKEND:-pipeline}"
export MINERU_LANG="${MINERU_LANG:-ch}"
export MINERU_TIMEOUT_SECONDS="${MINERU_TIMEOUT_SECONDS:-300}"
export MINERU_EXTRA_ARGS="${MINERU_EXTRA_ARGS:-}"

mkdir -p "$PLATFORM_DATA_ROOT" "$PLATFORM_LOG_ROOT"

export PYTHONPATH="$WORKSPACE_ROOT/backend/services/platform-api${PYTHONPATH:+:$PYTHONPATH}"

exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
