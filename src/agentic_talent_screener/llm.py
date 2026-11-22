"""Shared OpenRouter client for every model call in the project.

Settings are read from environment variables or a ``.env`` file:

- ``OPENROUTER_API_KEY``: required for any model call.
- ``ATS_LLM_MODEL``: OpenRouter model ID used when a caller doesn't pass one.
- ``ATS_LLM_MAX_RETRIES``: retries after a rate limit (429), server error (5xx)
  or network failure, with exponential backoff. Defaults to 4.
- ``ATS_OPENROUTER_BASE_URL``: API base URL (rarely needs changing).
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import requests
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_MODEL = "google/gemma-4-31b-it:free"
RETRY_BASE_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 30.0
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATS_", env_file=".env", env_ignore_empty=True, extra="ignore")

    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "ATS_OPENROUTER_API_KEY")
    )
    llm_model: str = DEFAULT_LLM_MODEL
    llm_max_retries: int = Field(default=4, ge=0, le=10)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


def default_llm_model() -> str:
    return LLMSettings().llm_model


def _retry_delay(attempt: int, response: requests.Response | None) -> float:
    """Honour a numeric Retry-After header, else exponential backoff with jitter."""
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), RETRY_MAX_DELAY_S)
        except ValueError:
            pass
    delay = min(RETRY_BASE_DELAY_S * 2**attempt, RETRY_MAX_DELAY_S)
    return delay * random.uniform(0.5, 1.0)


def _json_from_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("The model did not return valid JSON.") from error
    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")
    return result


def chat_json(
    messages: list[dict[str, str]],
    *,
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
    max_tokens: int,
    timeout: float,
    strict: bool = False,
) -> dict[str, Any]:
    """Send a chat request to OpenRouter and return the model's JSON object.

    The schema is passed as a structured-output ``response_format``. Not every
    provider enforces it, so callers must still validate the result.
    """
    settings = LLMSettings()
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to the environment or the project's .env file.")
    payload = {
        "model": model or settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": strict, "schema": schema},
        },
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    url = f"{settings.openrouter_base_url}/chat/completions"
    for attempt in range(settings.llm_max_retries + 1):
        last_attempt = attempt == settings.llm_max_retries
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as error:
            if last_attempt:
                raise ConnectionError(f"Cannot reach OpenRouter: {error}") from error
            time.sleep(_retry_delay(attempt, None))
            continue
        if response.status_code in _RETRYABLE_STATUS and not last_attempt:
            time.sleep(_retry_delay(attempt, response))
            continue
        break
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {response.text}") from error
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError("OpenRouter response did not contain chat content.") from error
    if not isinstance(content, str):
        raise RuntimeError("OpenRouter response did not contain chat content.")
    return _json_from_content(content)
