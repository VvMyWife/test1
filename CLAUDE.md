# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Archive AI Risk Review Platform** (MVP) - an AI-assisted archive document risk inspection system. It uses OCR + text classification + risk grading to automatically identify and classify batch image documents, supporting manual review and audit trails.

## Tech Stack

### Frontend
- React + TypeScript + Vite
- Ant Design component library
- React Query for data fetching
- Zustand for global state management

### Backend
- Python 3.12+ with FastAPI
- PostgreSQL (port 5433 via Docker)
- SQLAlchemy ORM + Alembic
- Celery for background tasks
- `uv` for dependency management

## Common Commands

### Frontend
```bash
cd frontend/kf-ai-reviewer
npm install
npm run dev
```

### Backend
```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
./run_worker.sh   # preferred: consumes default, pipeline, platform_ray_driver, platform_webhook
```

### Testing
```bash
cd backend
uv run pytest                              # all tests (excludes load tests)
uv run pytest tests/loadtest -m load       # load tests only
uv run pytest tests/test_pipeline.py -v     # single test file
uv run pytest tests/test_pipeline.py::test_function_name -v  # single test function
```

### Linting & Formatting
```bash
cd backend
uv run ruff check .       # lint
uv run ruff format .      # format (ruff)
# or
uv run black .            # format (black)
```

### Database (Docker)
```bash
docker compose -f backend/docker-compose.yml up -d
```
Uses port **5433** to avoid conflicts with local Postgres.

### Database Migrations
```bash
cd backend
uv run alembic upgrade head
```

## Architecture

### Three-Column Layout
The frontend uses a fixed 3-column design:
- **Left**: Archive database management, task management
- **Middle**: Document list and workspace
- **Right**: AI analysis results and human review

### Frontend Structure (`frontend/kf-ai-reviewer/src/`)
- `layout/` - Page layouts (MainLayout with 3-column design)
- `pages/` - Page components
- `features/` - Feature modules (database, document, ai-analysis, human-review)
- `services/` - API client layer
- `types/` - TypeScript type definitions
- `hooks/` - Custom React hooks
- `store/` - Zustand stores

### Backend Structure (`backend/app/`)
- `api/v1/` - API routes (databases, documents, ai, tasks, audit, overrides, jobs, **platform_jobs**, **rulesets**, **health**)
- `api/deps_platform.py` - Platform tenant header, optional API key, Redis rate limit dependency, admin secret for API key minting
- `models/` - SQLAlchemy ORM models (includes **platform_jobs**, **platform_api_keys**, rulesets)
- `schemas/` - Pydantic validation schemas
- `services/` - Business logic layer (includes **platform_job_service**, **platform_rate_limit_service**, **platform_webhook_service**, **platform_cache_service**)
- `db/` - Database session management
- `core/` - Configuration and utilities (`settings.platform_*` for quotas, webhooks, OCR cache)
- `execution/context.py` - **ExecutionContext** for platform/Celery driver boundaries
- `inference/` - Model access (OCR providers, LLM client, config_loader, pipeline); `rules_engine/` - CSV rule scan SSOT
- `operators/` - Replaceable operator units (`BaseOperator`, `contracts`, root modules like `extract_text_from_image`, `layout_extract_mineru`); **`operators/support/`** = bridges/DTOs to inference/rules_engine
- `pipeline/` - Runtime/orchestration: `build_pipeline_by_name`, thread-pool batch, Ray, lineage, `schemas` for image pipeline state; import `from app.pipeline import build_image_pipeline_v1`, … (do not re-export from `app.operators` to avoid import cycles)
- `tasks/` - Celery tasks (**celery_tasks**, **platform_tasks** Ray driver, **platform_webhook_tasks**)
- `celery_app.py` - Celery application configuration (task routes for **platform_ray_driver**, **platform_webhook** queues)

### Document Status Flow
```
uploaded → processing → ai_done → reviewing → confirmed → published
```
All status changes are controlled by the backend; frontend must not directly modify status fields.

### Operator & Pipeline Pattern

**Core principle**: `pipeline` → `operators` (unidirectional). Operators are stateless; pipeline handles orchestration.

- **`operators/`**: Stateless single-step units (OCR, cleaning, sensitive detection). Use `BaseOperator` contract + `OperatorContext` for execution info. **`operators/support/`** contains bridges to inference/rules_engine.
- **`pipeline/`**: Orchestration runtime. `build_image_pipeline_v1()` constructs the DAG; `LocalThreadPoolBackend` (default) or `RayBackend` (optional, `uv sync --extra ray`) executes it.
- **`ExecutionContext`**: Belongs to service layer (not foundation). Injected into pipeline via `ExecutionContextProvider` Protocol.
- Import from `app.pipeline` directly; do not re-export from `app.operators` (avoids import cycles).

## Key Design Documents

- `docs/design/01_system_goal_and_architecture.md` - System goals, architecture, state machine
- `docs/design/02_backend_module_design.md` - Backend module design
- `docs/design/03_frontend_module_design.md` - Frontend module design
- `docs/design/04_ai_pipeline_design.md` - AI pipeline design
- `docs/design/05_human_override_audit_state_machine.md` - Audit and state machine
- `docs/design/06_document_intelligence_platform.md` - Platform jobs (SSOT), rulesets, Celery+Ray, quotas/webhooks
- `docs/implementation/platform_capacity_runbook.md` - Platform capacity, Redis/Celery ops
- `docs/API_CONTRACT.md` - API contract (authoritative reference)
- `docs/DATABASE_SCHEMA.md` - Database schema (authoritative reference)

## Environment Variables

Create `backend/.env` based on `backend/.env.example`. Key variables:
- `DATABASE_URL` - PostgreSQL connection string
- `AI_OCR_PROVIDER` - OCR provider (paddle, mock, openai_compatible)
- `STORAGE_PATH` - Document storage path
- `REDIS_URL` - Celery broker + optional platform rate limit / quota (use `PLATFORM_REDIS_KEY_PREFIX` to isolate keys)
- `PLATFORM_REQUIRE_API_KEY`, `PLATFORM_ADMIN_SECRET`, `PLATFORM_*` quota/limit/backpressure - see `.env.example`

## Critical Rules (from AGENT_RULES.md)

### Architecture Constraints
- API prefix must remain `/api/v1/`
- Do not modify root directory structure
- Do not upgrade or add core dependencies without discussion
- All code must be written in designated directories (follow the structure above)

### Data Integrity
- All database writes must be transactional
- All status changes must go through Service layer
- Frontend cannot directly modify document status fields
- Never delete audit_logs records
- Use logical deletion (is_active=false), not physical deletion

### API Response Format
Always use this structure:
```json
{ "success": true, "data": {...}, "error": null }
{ "success": false, "data": null, "error": { "code": "...", "message": "..." } }
```

### Development Workflow
Implement new features in this order: models → schemas → services → api → frontend
