#!/usr/bin/env bash
set -euo pipefail

PADDLE_VENV="${PADDLE_VENV:-/opt/venvs/paddle}"
PADDLE_TABLE_API_HOST="${PADDLE_TABLE_API_HOST:-0.0.0.0}"
PADDLE_TABLE_API_PORT="${PADDLE_TABLE_API_PORT:-8200}"

site_packages="$("${PADDLE_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

export LD_LIBRARY_PATH="${site_packages}/nvidia/cudnn/lib:${site_packages}/nvidia/cublas/lib:${site_packages}/nvidia/cuda_runtime/lib:${site_packages}/nvidia/cuda_nvrtc/lib:${site_packages}/nvidia/cusolver/lib:${site_packages}/nvidia/cusparse/lib:${site_packages}/nvidia/nccl/lib:${site_packages}/nvidia/cufft/lib:${site_packages}/nvidia/curand/lib:${site_packages}/nvidia/cuda_cupti/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PADDLE_TABLE_API_PRELOAD="${PADDLE_TABLE_API_PRELOAD:-true}"
export PADDLE_TABLE_API_PRELOAD_MODES="${PADDLE_TABLE_API_PRELOAD_MODES:-table_structure,ppstructurev3}"
export PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS="${PADDLE_TABLE_API_MAX_CONCURRENT_EXTRACTS:-}"
export HOME="${HOME:-/workspace}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/workspace/.cache/modelscope}"

mkdir -p "${HOME}" "${XDG_CACHE_HOME}" /workspace/logs /workspace/run

case "${1:-serve}" in
  serve)
    exec "${PADDLE_VENV}/bin/python" -m uvicorn \
      platform_foundation.ocr.paddle_table_api:app \
      --host "${PADDLE_TABLE_API_HOST}" \
      --port "${PADDLE_TABLE_API_PORT}"
    ;;
  healthcheck)
    exec "${PADDLE_VENV}/bin/python" - <<PY
import urllib.request

url = "http://127.0.0.1:${PADDLE_TABLE_API_PORT}/health"
with urllib.request.urlopen(url, timeout=5) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    ;;
  python|python3)
    shift
    exec "${PADDLE_VENV}/bin/python" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
