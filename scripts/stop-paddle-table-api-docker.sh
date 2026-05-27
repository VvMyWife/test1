#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${PADDLE_TABLE_API_CONTAINER:-paddle-table-api}"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi

if "${docker_cmd[@]}" ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  "${docker_cmd[@]}" rm -f "${CONTAINER_NAME}" >/dev/null
  echo "stopped paddle-table-api docker: ${CONTAINER_NAME}"
else
  echo "paddle-table-api docker not found: ${CONTAINER_NAME}"
fi
