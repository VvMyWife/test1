#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_HOST_ROOT="${WORKSPACE_HOST_ROOT:-${REPO_ROOT}}"
IMAGE_NAME="${PADDLE_TABLE_API_IMAGE:-paddle-table-api:latest}"
CONTAINER_NAME="${PADDLE_TABLE_API_CONTAINER:-paddle-table-api}"
PADDLE_TABLE_API_HOST="${PADDLE_TABLE_API_HOST:-0.0.0.0}"
PADDLE_TABLE_API_PORT="${PADDLE_TABLE_API_PORT:-8200}"
PADDLE_TABLE_API_CHECK_HOST="${PADDLE_TABLE_API_CHECK_HOST:-127.0.0.1}"
PADDLE_TABLE_API_PUBLIC_HOST="${PADDLE_TABLE_API_PUBLIC_HOST:-${PADDLE_TABLE_API_CHECK_HOST}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CACHE_DIR="${PADDLE_TABLE_API_CACHE_DIR:-${WORKSPACE_HOST_ROOT}/.cache/paddle-table-api-home}"
BUILD_IMAGE="${PADDLE_TABLE_API_DOCKER_BUILD:-auto}"
BASE_IMAGE="${PADDLE_TABLE_API_BASE_IMAGE:-nvidia/cuda:11.8.0-runtime-ubuntu22.04}"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi

mkdir -p "${CACHE_DIR}"

if [[ "${BUILD_IMAGE}" == "true" ]] || ! "${docker_cmd[@]}" image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  "${docker_cmd[@]}" build \
    -t "${IMAGE_NAME}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -f "${REPO_ROOT}/docker/paddle-table-api.Dockerfile" \
    "${REPO_ROOT}"
fi

if "${docker_cmd[@]}" ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "paddle-table-api docker already running: ${CONTAINER_NAME}"
else
  if "${docker_cmd[@]}" ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    "${docker_cmd[@]}" rm -f "${CONTAINER_NAME}" >/dev/null
  fi
  "${docker_cmd[@]}" run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    -e PADDLE_TABLE_API_PRELOAD="${PADDLE_TABLE_API_PRELOAD:-true}" \
    -e PADDLE_TABLE_API_PRELOAD_MODES="${PADDLE_TABLE_API_PRELOAD_MODES:-table_structure,ppstructurev3}" \
    -p "${PADDLE_TABLE_API_HOST}:${PADDLE_TABLE_API_PORT}:8200" \
    -v "${WORKSPACE_HOST_ROOT}:${WORKSPACE_HOST_ROOT}" \
    -v "${CACHE_DIR}:/workspace" \
    "${IMAGE_NAME}" \
    serve >/dev/null
  echo "started paddle-table-api docker: ${CONTAINER_NAME} http://${PADDLE_TABLE_API_HOST}:${PADDLE_TABLE_API_PORT}"
fi

"${REPO_ROOT}/scripts/wait-http-ready.py" \
  "http://${PADDLE_TABLE_API_CHECK_HOST}:${PADDLE_TABLE_API_PORT}/health" \
  --name paddle-table-api \
  --timeout-seconds "${PADDLE_TABLE_API_STARTUP_TIMEOUT_SECONDS:-900}"

echo "export PADDLE_TABLE_API_URL=http://${PADDLE_TABLE_API_PUBLIC_HOST}:${PADDLE_TABLE_API_PORT}"
