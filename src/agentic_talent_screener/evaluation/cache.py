"""On-disk LLM response cache keyed by (model, schema, messages-hash).

Entries live in the committed SQLite eval store (``store.py``). The DB is public: synthetic prompts only,
never resumes from real applicants, never keys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .budget import BudgetGuard
from .store import STORE_FILE, EvalStore


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
    """LLM response cache backed by the eval store's ``llm_cache`` table (E15b).

    ``target`` is an :class:`~.store.EvalStore` or a directory (the store file ``eval_store.sqlite3`` then lives in it).
    The store is public: only synthetic prompts' responses may be cached. ``synthetic`` flags the run and defaults to False (every write is refused until the caller passes True); a write
    is refused when it is False, or when the prompt or response contains the key prefix ``sk-or-``.
    """

    def __init__(self, target: str | Path | EvalStore, *, synthetic: bool = False) -> None:
        self.store = target if isinstance(target, EvalStore) else EvalStore(Path(target) / STORE_FILE)
        self._owns_store = not isinstance(target, EvalStore)
        self.synthetic = synthetic
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        return self.store.get_llm(key)

    def put(self, key: str, value: dict[str, Any], *, model: str, schema_name: str,
            messages: list[dict[str, str]]) -> None:
        self.store.put_llm(key, model, schema_name, value, synthetic=self.synthetic, prompt=messages)

    def close(self) -> None:
        """Compact and close the store when this cache opened it."""
        if self._owns_store:
            self.store.close()


class CachedChat:
    """Wraps a ``chat_json``-compatible callable with the cache and the free-only guard."""

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
        self.guard.check(model)  # free-only, before cache and network
        options = {k: kw[k] for k in ("reasoning", "strict") if k in kw}
        key = cache_key(model, schema_name, schema, messages, options)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.hits += 1
            return hit
        self.cache.misses += 1
        if self.offline:
            raise CacheMiss("offline mode: cache miss")
        call = self._call
        if call is None:
            from agentic_talent_screener.llm import chat_json as call
        result = call(messages, schema=schema, schema_name=schema_name, model=model, **kw)
        self.cache.put(key, result, model=model, schema_name=schema_name, messages=messages)
        return result
