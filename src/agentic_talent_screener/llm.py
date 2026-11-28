"""Shared model clients: OpenRouter plus an explicit opt-in local L3 backend.

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
    l3_ollama_url: str = "http://127.0.0.1:11434"


def ollama_chat_json(messages: list[dict[str, str]], *, model: str,
                     schema: dict[str, Any], timeout: int = 120, num_predict: int = 64) -> dict[str, Any]:
    """Opt-in L3 only. No tools, credentials, provider fallbacks, or prompt logging.

    Keep this separate from ``chat_json`` so existing OpenRouter callers and their
    evaluation budget guards retain their original behavior.
    """
    url = LLMSettings().l3_ollama_url.rstrip("/")
    try:
        response = requests.post(f"{url}/api/chat", json={
            "model": model, "messages": messages, "stream": False, "format": schema,
            "options": {"temperature": 0, "seed": 42, "num_predict": num_predict, "num_ctx": 8192},
            "keep_alive": "10m",
        }, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Invalid response object.")
        if body.get("done_reason") == "length":
            raise ValueError("L3 Ollama response exhausted its completion budget.")
        content = body["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Missing text content.")
        return _json_from_content(content)
    except requests.RequestException:
        raise ConnectionError("L3 Ollama unavailable; check the server URL and imported model tag.") from None
    except (KeyError, TypeError, ValueError):
        raise ValueError("L3 Ollama returned an invalid or incomplete JSON response.") from None


def ollama_model_digest(model: str) -> str:
    """Resolve a local model tag before an experiment; never include response bodies in errors."""
    try:
        response = requests.get(f"{LLMSettings().l3_ollama_url.rstrip('/')}/api/tags", timeout=15)
        response.raise_for_status()
        for item in response.json()["models"]:
            if isinstance(item, dict) and item.get("name") in {model, f"{model}:latest"} and isinstance(item.get("digest"), str):
                return item["digest"]
    except (requests.RequestException, KeyError, TypeError, ValueError):
        raise ConnectionError("Cannot resolve L3 Ollama model digest.") from None
    raise ValueError("L3 model tag is not installed in Ollama.")


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


def _content_text(content: object) -> str | None:
    """Normalise OpenAI-compatible string and text-part content representations."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
            return None
        parts.append(item["text"])
    return "".join(parts)


def _safe_response_shape(body: object) -> str:
    """Describe an unexpected provider response without logging model text or PII."""
    if not isinstance(body, dict):
        return f"body_type={type(body).__name__}"
    details = [f"top_level_keys={sorted(str(key) for key in body)}"]
    choices = body.get("choices")
    if not isinstance(choices, list):
        details.append(f"choices_type={type(choices).__name__}")
    elif not choices:
        details.append("choices_count=0")
    else:
        choice = choices[0]
        details.append(f"first_choice_type={type(choice).__name__}")
        if isinstance(choice, dict):
            details.append(f"first_choice_keys={sorted(str(key) for key in choice)}")
            message = choice.get("message")
            details.append(f"message_type={type(message).__name__}")
            if isinstance(message, dict):
                details.append(f"message_keys={sorted(str(key) for key in message)}")
                details.append(f"content_type={type(message.get('content')).__name__}")
                if isinstance(choice.get("finish_reason"), str):
                    details.append(f"finish_reason={choice['finish_reason']}")
    error = body.get("error")
    if isinstance(error, dict):
        details.append(f"error_keys={sorted(str(key) for key in error)}")
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            details.append(f"error_metadata_keys={sorted(str(key) for key in metadata)}")
    return "; ".join(details)


def _safe_http_error(response: requests.Response) -> str:
    """Return provider diagnostics without copying an arbitrary response body to logs."""
    try:
        body = response.json()
    except (TypeError, ValueError):
        return f"OpenRouter request failed ({response.status_code}); response body was not JSON."
    return f"OpenRouter request failed ({response.status_code}); {_safe_response_shape(body)}"


def chat_json(
    messages: list[dict[str, str]],
    *,
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
    max_tokens: int,
    timeout: float,
    strict: bool = False,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a chat request to OpenRouter and return the model's JSON object.

    The schema is passed as a structured-output ``response_format``. Not every
    provider enforces it, so callers must still validate the result.
    """
    settings = LLMSettings()
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to the environment or the project's .env file.")
    payload: dict[str, Any] = {
        "model": model or settings.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": strict, "schema": schema},
        },
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
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
        raise RuntimeError(_safe_http_error(response)) from error
    try:
        body = response.json()
        content = _content_text(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"OpenRouter response did not contain chat content; {_safe_response_shape(locals().get('body'))}."
        ) from error
    if content is None:
        raise RuntimeError(f"OpenRouter response did not contain chat content; {_safe_response_shape(body)}.")
    return _json_from_content(content)
