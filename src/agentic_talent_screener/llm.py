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
import logging
import random
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import requests
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_MODEL = "google/gemma-4-31b-it:free"
RETRY_BASE_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 30.0
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "function_call", "content_filter", "error"})
LOGGER = logging.getLogger(__name__)
_MODEL_LOG_CONTEXT: ContextVar[str] = ContextVar("model_log_context", default="")


@contextmanager
def model_log_context(context: str) -> Iterator[None]:
    """Associate provider diagnostics with the worker's opaque task identifiers."""
    token = _MODEL_LOG_CONTEXT.set(context)
    try:
        yield
    finally:
        _MODEL_LOG_CONTEXT.reset(token)


def _log_identifier(value: str) -> str:
    # Only log conventional model/schema identifiers, never a URL, token, or multiline value.
    if not value.startswith("sk-") and "://" not in value and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}", value):
        return value
    return "<configured>"


def _request_context(provider: str, purpose: str, model: str) -> str:
    scope = _MODEL_LOG_CONTEXT.get()
    context = f"provider={provider} purpose={_log_identifier(purpose)} model={_log_identifier(model)}"
    return f"{context} {scope}" if scope else context


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
                     schema: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    """Opt-in L3 only. No tools, credentials, provider fallbacks, or prompt logging.

    Keep this separate from ``chat_json`` so existing OpenRouter callers and their
    evaluation budget guards retain their original behavior.
    """
    url = LLMSettings().l3_ollama_url.rstrip("/")
    context = _request_context("ollama", "l3_evidence", model)
    started = time.monotonic()
    response = None
    try:
        response = requests.post(f"{url}/api/chat", json={
            "model": model, "messages": messages, "stream": False, "format": schema,
            "options": {"temperature": 0, "seed": 42, "num_predict": 64, "num_ctx": 8192},
            "keep_alive": "10m",
        }, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as error:
        rejected = error.response if error.response is not None else response
        LOGGER.warning("Model request failed: %s reason=http_error http_status=%s elapsed=%.1fs",
                       context, getattr(rejected, "status_code", "unknown"),
                       time.monotonic() - started)
        raise ConnectionError("L3 Ollama unavailable; check the server URL and imported model tag.") from None
    except requests.RequestException as error:
        reason = "timeout" if isinstance(error, requests.Timeout) else "network_error"
        LOGGER.warning("Model request failed: %s reason=%s error=%s elapsed=%.1fs",
                       context, reason, type(error).__name__, time.monotonic() - started)
        raise ConnectionError("L3 Ollama unavailable; check the server URL and imported model tag.") from None
    reason = "invalid_json"
    try:
        body = response.json()
        reason = "invalid_response"
        if not isinstance(body, dict):
            raise ValueError("Invalid response object.")
        if body.get("done_reason") == "length":
            reason = "completion_limit"
            raise ValueError("L3 Ollama response exhausted its completion budget.")
        reason = "missing_content"
        content = body["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Missing text content.")
        reason = "invalid_json"
        return _json_from_content(content)
    except requests.RequestException as error:
        LOGGER.warning("Model request failed: %s reason=%s error=%s elapsed=%.1fs",
                       context, reason, type(error).__name__, time.monotonic() - started)
        raise ConnectionError("L3 Ollama unavailable; check the server URL and imported model tag.") from None
    except (KeyError, TypeError, ValueError):
        LOGGER.warning("Model request failed: %s reason=%s http_status=%s elapsed=%.1fs",
                       context, reason, getattr(response, "status_code", "unknown"), time.monotonic() - started)
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
    def known_keys(value: dict, *keys: str) -> list[str]:
        return sorted(key for key in keys if key in value)

    if not isinstance(body, dict):
        return f"body_type={type(body).__name__}"
    details = [f"top_level_keys={known_keys(body, 'id', 'provider', 'model', 'choices', 'usage', 'error')}"]
    choices = body.get("choices")
    if not isinstance(choices, list):
        details.append(f"choices_type={type(choices).__name__}")
    elif not choices:
        details.append("choices_count=0")
    else:
        choice = choices[0]
        details.append(f"first_choice_type={type(choice).__name__}")
        if isinstance(choice, dict):
            details.append(f"first_choice_keys={known_keys(choice, 'index', 'message', 'finish_reason', 'error')}")
            message = choice.get("message")
            details.append(f"message_type={type(message).__name__}")
            if isinstance(message, dict):
                details.append(f"message_keys={known_keys(message, 'role', 'content', 'reasoning', 'reasoning_details', 'refusal', 'tool_calls')}")
                details.append(f"content_type={type(message.get('content')).__name__}")
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                details.append(f"finish_reason={finish if finish in _FINISH_REASONS else 'unknown'}")
    usage = body.get("usage")
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        if type(completion_tokens) is int and 0 <= completion_tokens <= 10_000_000:
            details.append(f"completion_tokens={completion_tokens}")
    error = body.get("error")
    if isinstance(error, dict):
        details.append(f"error_keys={known_keys(error, 'code', 'message', 'metadata')}")
        code = error.get("code")
        if type(code) is int and 100 <= code <= 599:
            details.append(f"provider_error_code={code}")
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            details.append(f"error_metadata_keys={known_keys(metadata, 'provider_name', 'raw', 'request_id')}")
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
    reasoning: bool | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a chat request to OpenRouter and return the model's JSON object.

    The schema is passed as a structured-output ``response_format``. Not every
    provider enforces it, so callers must still validate the result.

    Pass a boolean to set ``reasoning.enabled``, a mapping for options such as
    a reasoning token budget, or ``None`` to use the provider default.
    """
    settings = LLMSettings()
    selected_model = model or settings.llm_model
    context = _request_context("openrouter", schema_name, selected_model)
    if settings.openrouter_api_key is None:
        LOGGER.warning("Model request failed: %s reason=missing_api_key", context)
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to the environment or the project's .env file.")
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": strict, "schema": schema},
        },
    }
    reasoning_options = {"enabled": reasoning} if isinstance(reasoning, bool) else reasoning
    if reasoning_options is not None:
        payload["reasoning"] = reasoning_options
    reasoning_enabled = reasoning_options.get("enabled") if reasoning_options else None
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    url = f"{settings.openrouter_base_url}/chat/completions"
    attempts = settings.llm_max_retries + 1
    for attempt in range(attempts):
        last_attempt = attempt + 1 == attempts
        request = f"{context} attempt={attempt + 1}/{attempts}"
        LOGGER.info("Model request started: %s timeout=%gs max_tokens=%d reasoning_enabled=%s",
                    request, timeout, max_tokens, reasoning_enabled if isinstance(reasoning_enabled, bool) else "default")
        started = time.monotonic()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as error:
            reason = "timeout" if isinstance(error, requests.Timeout) else "network_error"
            elapsed = time.monotonic() - started
            if last_attempt:
                LOGGER.warning("Model request failed: %s reason=%s error=%s elapsed=%.1fs",
                               request, reason, type(error).__name__, elapsed)
                raise ConnectionError(f"Cannot reach OpenRouter ({type(error).__name__}).") from error
            delay = _retry_delay(attempt, None)
            LOGGER.warning("Model request retry: %s reason=%s error=%s elapsed=%.1fs retry_in=%.1fs",
                           request, reason, type(error).__name__, elapsed, delay)
            time.sleep(delay)
            continue
        elapsed = time.monotonic() - started
        if response.status_code in _RETRYABLE_STATUS and not last_attempt:
            delay = _retry_delay(attempt, response)
            LOGGER.warning("Model request retry: %s reason=http_error http_status=%d elapsed=%.1fs retry_in=%.1fs",
                           request, response.status_code, elapsed, delay)
            time.sleep(delay)
            continue
        break
    response_context = f"{request} http_status={response.status_code} elapsed={elapsed:.1f}s"
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        diagnostic = _safe_http_error(response)
        LOGGER.warning("Model request failed: %s reason=http_error %s", response_context, diagnostic)
        raise RuntimeError(diagnostic) from error
    try:
        body = response.json()
    except (TypeError, ValueError) as error:
        LOGGER.warning("Model request failed: %s reason=invalid_response_json", response_context)
        raise RuntimeError("OpenRouter response was not JSON.") from error
    try:
        content = _content_text(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        content = None
    if content is None:
        shape = _safe_response_shape(body)
        LOGGER.warning("Model request failed: %s reason=missing_content %s", response_context, shape)
        raise RuntimeError(f"OpenRouter response did not contain chat content; {shape}.")
    try:
        result = _json_from_content(content)
    except ValueError:
        reason = "empty_content" if not content.strip() else "invalid_completion_json"
        LOGGER.warning("Model request failed: %s reason=%s %s", response_context, reason, _safe_response_shape(body))
        raise
    LOGGER.info("Model request completed: %s", response_context)
    return result
