from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from agentic_talent_screener import __version__
from agentic_talent_screener.api.dependencies import SettingsDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str


@router.get("/health")
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, environment=settings.environment)
