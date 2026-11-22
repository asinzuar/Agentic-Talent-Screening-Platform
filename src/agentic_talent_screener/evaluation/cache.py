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


def cache_key(model: str, schema_name: str, schema: dict[str, Any], messages: list[dict[str, str]]) -> str:
    blob = json.dumps(
        {"model": model, "schema_name": schema_name, "schema": schema, "messages": messages},
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
    ) -> None:
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
        key = cache_key(model, schema_name, schema, messages)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.hits += 1
            return hit
        self.cache.misses += 1
        if self.offline:
            raise RuntimeError("offline mode: cache miss")
        self.guard.check(model)
        call = self._call
        if call is None:
            from agentic_talent_screener.llm import chat_json as call
        result = call(messages, schema=schema, schema_name=schema_name, model=model, **kw)
        self.cache.put(key, result)
        return result
