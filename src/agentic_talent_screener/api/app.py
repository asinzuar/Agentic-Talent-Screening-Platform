import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_talent_screener import __version__
from agentic_talent_screener.api.config import Settings, get_settings
from agentic_talent_screener.api.routers import health, ranking, v1, workbench
from agentic_talent_screener.file_logging import log_to_file
from agentic_talent_screener.ranking.audit import AuditLedger
from agentic_talent_screener.workflow.storage import Store, WorkflowError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    server = logging.getLogger("uvicorn.error")
    access = logging.getLogger("uvicorn.access")
    # Respect Uvicorn's --no-access-log. It disables handlers and propagation.
    loggers = (server, access) if access.hasHandlers() else (server,)
    with log_to_file(app.state.settings.log_dir / "server.log", *loggers):
        server.info("Talent Screener API started")
        try:
            yield
        finally:
            server.info("Talent Screener API stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.workflow_store = Store(settings.workflow_db)

    @app.exception_handler(WorkflowError)
    async def workflow_error(request: Request, error: WorkflowError) -> JSONResponse:
        headers = {"Cache-Control": "no-store"}
        if error.status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(error.body, status_code=error.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        path = f"{settings.api_v1_prefix}/jobs"
        if request.url.path == path or request.url.path.startswith(f"{path}/"):
            return JSONResponse(
                {"code": "INVALID_REQUEST", "message": "Check the requested fields.", "retryable": False,
                 "field_errors": [{"field": ".".join(map(str, item["loc"][:-1] if item["type"] == "extra_forbidden"
                                                         else item["loc"])), "message": "Invalid or missing value."}
                                  for item in error.errors()]},
                status_code=422, headers={"Cache-Control": "no-store"},
            )
        return await request_validation_exception_handler(request, error)

    # In-memory, per-app ledger seeded with the sample profile's approval.
    ledger = AuditLedger()
    ledger.record_profile_approval(ranking.PROFILE)
    app.state.ledger = ledger

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Location", "Content-Disposition"],
        )

    app.include_router(workbench.router)
    app.include_router(health.router)
    app.include_router(v1.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
