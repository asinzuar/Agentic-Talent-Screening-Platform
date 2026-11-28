from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings, read from `ATS_*` environment variables or a `.env` file."""

    model_config = SettingsConfigDict(env_prefix="ATS_", env_file=".env", extra="ignore")

    app_name: str = "Agentic Talent Screener"
    environment: str = "dev"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = []
    workflow_db: Path = Path(".ats/workflow.sqlite3")
    log_dir: Path = Path(".ats/logs")
    workflow_tokens: dict[str, SecretStr] = {}

    @field_validator("workflow_tokens")
    @classmethod
    def check_workflow_tokens(cls, tokens: dict[str, SecretStr]) -> dict[str, SecretStr]:
        values = [token.get_secret_value() for token in tokens.values()]
        if any(not actor.strip() or actor == "system" for actor in tokens):
            raise ValueError("Configure distinct non-system workflow actor IDs.")
        if any(not value or not value.isascii() or any(char.isspace() for char in value) for value in values):
            raise ValueError("Configure non-blank ASCII bearer tokens without whitespace.")
        if len(values) != len(set(values)):
            raise ValueError("Workflow bearer tokens must be distinct.")
        return tokens


@lru_cache
def get_settings() -> Settings:
    return Settings()
