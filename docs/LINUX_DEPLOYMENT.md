# Linux deployment notes

This deployment plan follows the existing Linux workspace exactly:

```text
~/mineru_workspace/
├── constraints-mineru-pipeline.txt
├── constraints-mineru-torch.txt
├── data/
├── logs/
├── scripts/
└── summary/
```

No extra top-level deployment directory is required. Put this project's contents directly under
`~/mineru_workspace`.

After copying the project contents, the workspace will contain the project-owned directories/files such as
`backend/`, `docs/`, `README.md`, `AGENTS.md`, and `.python-version`. These are required by the source tree;
they are not new deployment names.

## Directory usage

- `~/mineru_workspace/backend`: backend source code copied from this project
- `~/mineru_workspace/docs`: project documentation copied from this project
- `~/mineru_workspace/data`: API temporary input files and MinerU output
- `~/mineru_workspace/logs`: service logs
- `~/mineru_workspace/scripts`: local startup/helper scripts
- `~/mineru_workspace/summary`: manual run summaries or exported result summaries
- `~/mineru_workspace/constraints-mineru-pipeline.txt`: existing MinerU pipeline constraints file
- `~/mineru_workspace/constraints-mineru-torch.txt`: existing MinerU torch constraints file

## Copy project contents

Run this from the machine that has the project directory:

```bash
rsync -a \
  --exclude ".git/" \
  --exclude ".pytest_cache/" \
  --exclude "__pycache__/" \
  ./platform-core-public-feature-foundation-base-operator/ \
  ~/mineru_workspace/
```

If you copy from Windows manually, copy the contents of
`platform-core-public-feature-foundation-base-operator` into `~/mineru_workspace`, not into a new nested
folder.

Do not use `rsync --delete` here. The existing `data`, `logs`, `scripts`, `summary`, and
`constraints-mineru-*.txt` entries belong to this workspace and should not be removed.

## Install dependencies

Run on the Linux server if you want to recreate dependencies with `uv`:

```bash
cd ~/mineru_workspace/backend
uv sync --frozen
```

For tests:

```bash
cd ~/mineru_workspace/backend
uv sync --frozen --group dev
```

Your current `mineru312` conda environment already contains Python, MinerU, FastAPI, and Uvicorn. The included
`scripts/start-platform-api.sh` therefore starts with `python -m uvicorn` and does not require `uv` to be
installed on the server.

## Environment variables

The code now defaults to this workspace:

```bash
/home/liujiacheng/mineru_workspace
```

The default data prefix is:

```bash
/home/liujiacheng/mineru_workspace/data
```

You can still override the defaults with environment variables:

```bash
export PLATFORM_WORKSPACE_ROOT="/home/liujiacheng/mineru_workspace"
export PLATFORM_DATA_ROOT="/home/liujiacheng/mineru_workspace/data"
export PLATFORM_LOG_ROOT="/home/liujiacheng/mineru_workspace/logs"
export PLATFORM_UPLOAD_TEMP_ROOT="/home/liujiacheng/mineru_workspace/data"
export PLATFORM_FOUNDATION_ROOT="/home/liujiacheng/mineru_workspace/backend/foundation"
export MINERU_COMMAND="mineru"
export MINERU_OUTPUT_ROOT="/home/liujiacheng/mineru_workspace/data"
export MINERU_PARSE_METHOD="auto"
export MINERU_BACKEND="pipeline"
export MINERU_LANG="ch"
export MINERU_TIMEOUT_SECONDS="300"
export MINERU_EXTRA_ARGS=""
```

If `mineru` is not on `PATH`, set `MINERU_COMMAND` to the real command path from your environment.

## Run the API manually

Because the service directory is named `platform-api`, run Uvicorn from that service directory:

```bash
cd ~/mineru_workspace/backend/services/platform-api

export PLATFORM_WORKSPACE_ROOT="/home/liujiacheng/mineru_workspace"
export PLATFORM_DATA_ROOT="/home/liujiacheng/mineru_workspace/data"
export PLATFORM_LOG_ROOT="/home/liujiacheng/mineru_workspace/logs"
export PLATFORM_UPLOAD_TEMP_ROOT="/home/liujiacheng/mineru_workspace/data"
export PLATFORM_FOUNDATION_ROOT="/home/liujiacheng/mineru_workspace/backend/foundation"
export MINERU_COMMAND="mineru"
export MINERU_OUTPUT_ROOT="/home/liujiacheng/mineru_workspace/data"
export MINERU_PARSE_METHOD="auto"
export MINERU_BACKEND="pipeline"
export MINERU_LANG="ch"
export MINERU_TIMEOUT_SECONDS="300"
export MINERU_EXTRA_ARGS=""

PYTHONPATH="/home/liujiacheng/mineru_workspace/backend/services/platform-api" \
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Start script

This project now includes this new file:

- `~/mineru_workspace/scripts/start-platform-api.sh`

It will be copied into your existing `scripts` directory when you copy the project contents.
The script explicitly loads `/home/liujiacheng/miniconda3/etc/profile.d/conda.sh` and activates `mineru312`,
because non-interactive SSH sessions do not automatically load your shell prompt environment.

Then run:

```bash
chmod +x ~/mineru_workspace/scripts/start-platform-api.sh
~/mineru_workspace/scripts/start-platform-api.sh
```

## Logs

Manual shell logs can be redirected to the existing `logs` directory:

```bash
/home/liujiacheng/mineru_workspace/scripts/start-platform-api.sh \
  > /home/liujiacheng/mineru_workspace/logs/platform-api.log \
  2> /home/liujiacheng/mineru_workspace/logs/platform-api.err.log
```

## Smoke test

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Expected response:

```json
{"success":true,"data":{"status":"ok"},"error":null}
```
