from typing import Annotated

from fastapi import Depends, Request

from agentic_talent_screener.api.config import Settings


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
