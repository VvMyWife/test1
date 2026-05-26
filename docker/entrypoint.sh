#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/mineru_workspace}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
MINERU_VENV="${MINERU_VENV:-/opt/venvs/mineru}"
PADDLE_VENV="${PADDLE_VENV:-/opt/venvs/paddle}"

MINERU_API_HOST="${MINERU_API_HOST:-127.0.0.1}"
MINERU_API_PORT="${MINERU_API_PORT:-8000}"
PADDLE_TABLE_API_HOST="${PADDLE_TABLE_API_HOST:-127.0.0.1}"
PADDLE_TABLE_API_PORT="${PADDLE_TABLE_API_PORT:-8200}"
MINERU_API_URL="${MINERU_API_URL:-http://127.0.0.1:${MINERU_API_PORT}}"
PADDLE_TABLE_API_URL="${PADDLE_TABLE_API_URL:-http://127.0.0.1:${PADDLE_TABLE_API_PORT}}"

export MINERU_API_URL
export PADDLE_TABLE_API_URL
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-modelscope}"
export MINERU_API_MAX_CONCURRENT_REQUESTS="${MINERU_API_MAX_CONCURRENT_REQUESTS:-128}"
export PADDLE_TABLE_API_PRELOAD="${PADDLE_TABLE_API_PRELOAD:-true}"
export PADDLE_TABLE_API_PRELOAD_MODES="${PADDLE_TABLE_API_PRELOAD_MODES:-table_structure,ppstructurev3}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKSPACE_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${WORKSPACE_ROOT}/.cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-${WORKSPACE_ROOT}/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/pip}"

mkdir -p "${WORKSPACE_ROOT}/input" "${WORKSPACE_ROOT}/output" "${WORKSPACE_ROOT}/logs" "${WORKSPACE_ROOT}/run" "${XDG_CACHE_HOME}"

mineru_pid=""
paddle_pid=""

log() {
  printf '[mineru-docker] %s\n' "$*"
}

wait_http() {
  local name="$1"
  local url="$2"
  local timeout_seconds="${3:-300}"
  "${MINERU_VENV}/bin/python" - "$name" "$url" "$timeout_seconds" <<'PY'
import sys
import time
import urllib.request

name, url, timeout_seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.monotonic() + timeout_seconds
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            print(f"{name} ready: {url} status={response.status}", flush=True)
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(1)
raise SystemExit(f"{name} not ready: {last_error!r}")
PY
}

start_mineru_api() {
  if curl -fsS "${MINERU_API_URL}/health" >/dev/null 2>&1; then
    log "mineru-api already available at ${MINERU_API_URL}"
    return
  fi
  log "starting mineru-api at ${MINERU_API_URL}"
  "${MINERU_VENV}/bin/mineru-api" \
    --host "${MINERU_API_HOST}" \
    --port "${MINERU_API_PORT}" \
    --enable-vlm-preload "${MINERU_API_PRELOAD:-false}" \
    > "${WORKSPACE_ROOT}/logs/mineru-api-${MINERU_API_PORT}.log" 2>&1 &
  mineru_pid="$!"
  echo "${mineru_pid}" > "${WORKSPACE_ROOT}/run/mineru-api-${MINERU_API_PORT}.pid"
  wait_http "mineru-api" "${MINERU_API_URL}/health" "${MINERU_API_STARTUP_TIMEOUT_SECONDS:-300}"
}

start_paddle_api() {
  if curl -fsS "${PADDLE_TABLE_API_URL}/health" >/dev/null 2>&1; then
    log "paddle-table-api already available at ${PADDLE_TABLE_API_URL}"
    return
  fi
  log "starting paddle-table-api at ${PADDLE_TABLE_API_URL}"
  "${PADDLE_VENV}/bin/python" -m uvicorn \
    platform_foundation.ocr.paddle_table_api:app \
    --host "${PADDLE_TABLE_API_HOST}" \
    --port "${PADDLE_TABLE_API_PORT}" \
    > "${WORKSPACE_ROOT}/logs/paddle-table-api-${PADDLE_TABLE_API_PORT}.log" 2>&1 &
  paddle_pid="$!"
  echo "${paddle_pid}" > "${WORKSPACE_ROOT}/run/paddle-table-api-${PADDLE_TABLE_API_PORT}.pid"
  wait_http "paddle-table-api" "${PADDLE_TABLE_API_URL}/health" "${PADDLE_TABLE_API_STARTUP_TIMEOUT_SECONDS:-600}"
}

stop_children() {
  set +e
  if [[ -n "${paddle_pid}" ]] && kill -0 "${paddle_pid}" 2>/dev/null; then
    kill "${paddle_pid}"
  fi
  if [[ -n "${mineru_pid}" ]] && kill -0 "${mineru_pid}" 2>/dev/null; then
    kill "${mineru_pid}"
  fi
}
trap stop_children EXIT TERM INT

run_batch() {
  start_mineru_api
  if [[ "${ENABLE_PADDLE_API:-false}" == "true" ]]; then
    start_paddle_api
  fi
  shift || true
  if [[ "$#" -eq 0 ]]; then
    set -- "${WORKSPACE_ROOT}/input" \
      --output-dir "${WORKSPACE_ROOT}/output" \
      --table-engine "${TABLE_ENGINE:-ocr}" \
      --concurrency "${CONCURRENCY:-1}" \
      --overwrite
  fi
  cd "${WORKSPACE_ROOT}"
  exec "${MINERU_VENV}/bin/python" "${APP_ROOT}/scripts/run-daft-batch-operate.py" "$@"
}

case "${1:-server}" in
  server)
    start_mineru_api
    if [[ "${ENABLE_PADDLE_API:-false}" == "true" ]]; then
      start_paddle_api
    fi
    log "services are running. logs are under ${WORKSPACE_ROOT}/logs"
    wait -n
    ;;
  batch)
    run_batch "$@"
    ;;
  healthcheck)
    curl -fsS "${MINERU_API_URL}/health" >/dev/null
    if [[ "${ENABLE_PADDLE_API:-false}" == "true" ]]; then
      curl -fsS "${PADDLE_TABLE_API_URL}/health" >/dev/null
    fi
    ;;
  bash|sh|python|python3)
    exec "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
