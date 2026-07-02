#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mineru312}"
PADDLE_TABLE_API_HOST="${PADDLE_TABLE_API_HOST:-127.0.0.1}"
PADDLE_TABLE_API_PORT="${PADDLE_TABLE_API_PORT:-8200}"
PADDLE_TABLE_API_PRELOAD="${PADDLE_TABLE_API_PRELOAD:-true}"
PADDLE_TABLE_API_PRELOAD_MODES="${PADDLE_TABLE_API_PRELOAD_MODES:-layout,ocr,table}"

LOG_DIR="${WORKSPACE_ROOT}/logs"
RUN_DIR="${WORKSPACE_ROOT}/run"
PID_FILE="${RUN_DIR}/paddle-table-api-${PADDLE_TABLE_API_PORT}.pid"
LOG_FILE="${LOG_DIR}/paddle-table-api-${PADDLE_TABLE_API_PORT}.log"
BASE_URL="http://${PADDLE_TABLE_API_HOST}:${PADDLE_TABLE_API_PORT}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -f "${CONDA_SH}" && -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export PADDLE_TABLE_API_PRELOAD
export PADDLE_TABLE_API_PRELOAD_MODES

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "paddle-table-api already running: pid=$(cat "${PID_FILE}") url=${BASE_URL}"
else
  if command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ":${PADDLE_TABLE_API_PORT} "; then
    echo "port ${PADDLE_TABLE_API_PORT} is already listening; reuse ${BASE_URL}"
  else
    nohup python -m uvicorn \
      platform_foundation.ocr.paddle_table_api:app \
      --host "${PADDLE_TABLE_API_HOST}" \
      --port "${PADDLE_TABLE_API_PORT}" \
      > "${LOG_FILE}" 2>&1 &
    echo "$!" > "${PID_FILE}"
    echo "started paddle-table-api: pid=$(cat "${PID_FILE}") url=${BASE_URL} log=${LOG_FILE}"
  fi
fi

python - <<PY
import json
import time
import urllib.request

url = "${BASE_URL}/health"
start = time.perf_counter()
last_error = None
for _ in range(240):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            print(
                f"paddle-table-api ready: {url} status={response.status} "
                f"startup_seconds={time.perf_counter() - start:.3f} "
                f"cache={payload.get('cache')}"
            )
            break
    except Exception as exc:
        last_error = exc
        time.sleep(1)
else:
    raise SystemExit(f"paddle-table-api not ready: {last_error!r}")
PY

echo "export PADDLE_TABLE_API_URL=${BASE_URL}"
