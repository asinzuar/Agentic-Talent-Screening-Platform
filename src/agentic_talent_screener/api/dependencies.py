from typing import Annotated

from fastapi import Depends, Request

from agentic_talent_screener.api.config import Settings
from agentic_talent_screener.ranking.audit import AuditLedger


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ledger(request: Request) -> AuditLedger:
    return request.app.state.ledger


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
LedgerDep = Annotated[AuditLedger, Depends(get_ledger)]
