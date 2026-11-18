from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings, read from `ATS_*` environment variables or a `.env` file."""

    model_config = SettingsConfigDict(env_prefix="ATS_", env_file=".env", extra="ignore")

    app_name: str = "Agentic Talent Screener"
    environment: str = "dev"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = []


@lru_cache
def get_settings() -> Settings:
    return Settings()
