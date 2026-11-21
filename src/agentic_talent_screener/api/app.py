from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_talent_screener import __version__
from agentic_talent_screener.api.config import Settings, get_settings
from agentic_talent_screener.api.routers import health, ranking, v1, workbench
from agentic_talent_screener.ranking.audit import AuditLedger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Set up shared resources (model clients, LangGraph graphs, Langfuse, storage) here
    # and store them on `app.state`; tear them down after the yield.
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings

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
        )

    app.include_router(workbench.router)
    app.include_router(health.router)
    app.include_router(v1.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
