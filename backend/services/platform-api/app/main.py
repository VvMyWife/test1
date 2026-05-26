from __future__ import annotations

from .bootstrap import ensure_foundation_on_path

ensure_foundation_on_path()

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from .api.responses import error_response, success_response
from .api.v1.mineru import router as mineru_router
from .services.mineru_layout_service import PlatformApiError


def create_app() -> FastAPI:
    app = FastAPI(title="platform-api")

    @app.exception_handler(PlatformApiError)
    async def handle_platform_api_error(
        request: Request,
        exc: PlatformApiError,
    ):
        return error_response(code=exc.code, message=exc.message, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ):
        return error_response(
            code="VALIDATION_ERROR",
            message=str(exc),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(
        request: Request,
        exc: HTTPException,
    ):
        return error_response(
            code="HTTP_ERROR",
            message=str(exc.detail),
            status_code=exc.status_code,
        )

    @app.get("/api/v1/health")
    def health():
        return success_response({"status": "ok"})

    app.include_router(mineru_router)

    return app


app = create_app()
