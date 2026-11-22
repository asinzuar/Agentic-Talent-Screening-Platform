"""On-disk LLM response cache keyed by (model, schema, messages-hash).

Synthetic or public text only; never store resumes from real applicants.
Each entry is one JSON file named by its key.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .budget import BudgetGuard


class CacheMiss(RuntimeError):
    """Raised in offline mode when a response is not cached."""


def cache_key(
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
) -> str:
    blob = json.dumps(
        {"model": model, "schema_name": schema_name, "schema": schema, "messages": messages, "options": options or {}},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path(key))


class CachedChat:
    """Wraps a ``chat_json``-compatible callable with the cache and budget guard."""

    def __init__(
        self,
        cache: LLMCache,
        guard: BudgetGuard,
        *,
        offline: bool = False,
        call: Callable[..., dict[str, Any]] | None = None,
        default_model: Callable[[], str] | None = None,
        prices_per_mtok: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """``prices_per_mtok`` maps a paid model to (input, output) USD per million tokens.

        A paid model without a price is refused, so the cap cannot be bypassed.
        """
        self.prices = prices_per_mtok or {}
        self.cache, self.guard, self.offline = cache, guard, offline
        self._call, self._default_model = call, default_model

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        schema_name: str,
        model: str | None = None,
        **kw: Any,
    ) -> dict[str, Any]:
        if model is None:
            default_model = self._default_model
            if default_model is None:
                from agentic_talent_screener.llm import default_llm_model as default_model
            model = default_model()
        options = {k: kw[k] for k in ("reasoning", "strict") if k in kw}
        key = cache_key(model, schema_name, schema, messages, options)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.hits += 1
            return hit
        self.cache.misses += 1
        if self.offline:
            raise CacheMiss("offline mode: cache miss")
        estimate = self._estimate(model, messages, kw.get("max_tokens"))
        self.guard.check(model, estimate)
        call = self._call
        if call is None:
            from agentic_talent_screener.llm import chat_json as call
        result = call(messages, schema=schema, schema_name=schema_name, model=model, **kw)
        self.guard.record(self._actual(result, model, estimate))
        self.cache.put(key, result)
        return result

    def _estimate(self, model: str, messages: list[dict[str, str]], max_tokens: Any) -> float:
        """Upper-bound cost in USD; 0 for free models, 0 (refused by the guard) if no price is known."""
        if model.endswith(":free"):
            return 0.0
        price = self.prices.get(model)
        if price is None or not isinstance(max_tokens, int) or max_tokens <= 0:
            return 0.0
        in_tokens = sum(len(m.get("content", "")) for m in messages) / 3  # conservative chars-per-token
        return (in_tokens * price[0] + max_tokens * price[1]) / 1e6

    def _actual(self, result: dict[str, Any], model: str, estimate: float) -> float:
        """Cost from a response usage block when present, else the estimate (conservative)."""
        usage = result.get("usage") if isinstance(result, dict) else None
        price = self.prices.get(model)
        if isinstance(usage, dict) and price is not None:
            p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if isinstance(p, (int, float)) and isinstance(c, (int, float)):
                return (p * price[0] + c * price[1]) / 1e6
        return estimate
