#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PADDLE_TABLE_API_PORT="${PADDLE_TABLE_API_PORT:-8200}"
PID_FILE="${WORKSPACE_ROOT}/run/paddle-table-api-${PADDLE_TABLE_API_PORT}.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "paddle-table-api pid file not found: ${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}"
  echo "stopped paddle-table-api: pid=${PID}"
else
  echo "paddle-table-api process already stopped: pid=${PID}"
fi
rm -f "${PID_FILE}"
