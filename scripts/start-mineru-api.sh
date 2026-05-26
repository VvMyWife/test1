#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mineru312}"
MINERU_API_HOST="${MINERU_API_HOST:-127.0.0.1}"
MINERU_API_PORT="${MINERU_API_PORT:-8000}"
MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-modelscope}"
MINERU_API_PRELOAD="${MINERU_API_PRELOAD:-false}"
MINERU_API_MAX_CONCURRENT_REQUESTS="${MINERU_API_MAX_CONCURRENT_REQUESTS:-128}"

LOG_DIR="${WORKSPACE_ROOT}/logs"
RUN_DIR="${WORKSPACE_ROOT}/run"
PID_FILE="${RUN_DIR}/mineru-api-${MINERU_API_PORT}.pid"
LOG_FILE="${LOG_DIR}/mineru-api-${MINERU_API_PORT}.log"
BASE_URL="http://${MINERU_API_HOST}:${MINERU_API_PORT}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -f "${CONDA_SH}" && -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export MINERU_MODEL_SOURCE
export MINERU_API_MAX_CONCURRENT_REQUESTS

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "mineru-api already running: pid=$(cat "${PID_FILE}") url=${BASE_URL}"
else
  if command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ":${MINERU_API_PORT} "; then
    echo "port ${MINERU_API_PORT} is already listening; reuse ${BASE_URL}"
  else
    nohup mineru-api \
      --host "${MINERU_API_HOST}" \
      --port "${MINERU_API_PORT}" \
      --enable-vlm-preload "${MINERU_API_PRELOAD}" \
      > "${LOG_FILE}" 2>&1 &
    echo "$!" > "${PID_FILE}"
    echo "started mineru-api: pid=$(cat "${PID_FILE}") url=${BASE_URL} log=${LOG_FILE}"
  fi
fi

python - <<PY
import json
import time
import urllib.request

url = "${BASE_URL}/health"
start = time.perf_counter()
last_error = None
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            print(
                f"mineru-api ready: {url} status={response.status} "
                f"startup_seconds={time.perf_counter() - start:.3f} "
                f"max_concurrent_requests={payload.get('max_concurrent_requests')}"
            )
            break
    except Exception as exc:
        last_error = exc
        time.sleep(1)
else:
    raise SystemExit(f"mineru-api not ready: {last_error!r}")
PY

echo "export MINERU_API_URL=${BASE_URL}"
