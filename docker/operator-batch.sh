#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/mineru_workspace}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
MINERU_VENV="${MINERU_VENV:-/opt/venvs/mineru}"

export WORKSPACE="${WORKSPACE:-${WORKSPACE_ROOT}}"
export MINERU_API_URL="${MINERU_API_URL:-http://127.0.0.1:8000}"

TABLE_ENGINE="${TABLE_ENGINE:-${MINERU_TABLE_ENGINE:-ocr}}"
CONCURRENCY="${CONCURRENCY:-12}"
INPUT_DIR="${INPUT_DIR:-${WORKSPACE_ROOT}/input}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/output/${TABLE_ENGINE}}"
OVERWRITE="${OVERWRITE:-true}"

if [[ "${TABLE_ENGINE}" == "paddle" ]]; then
  export PADDLE_TABLE_API_URL="${PADDLE_TABLE_API_URL:-http://127.0.0.1:8200}"
fi

if [[ "$#" -gt 0 ]]; then
  exec "${MINERU_VENV}/bin/python" "${APP_ROOT}/scripts/run-daft-batch-operate.py" "$@"
fi

args=(
  "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --table-engine "${TABLE_ENGINE}"
  --concurrency "${CONCURRENCY}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  args+=(--overwrite)
fi

exec "${MINERU_VENV}/bin/python" "${APP_ROOT}/scripts/run-daft-batch-operate.py" "${args[@]}"
