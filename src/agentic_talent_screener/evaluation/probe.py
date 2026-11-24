"""Resume probe: which model calls of a plan are NOT in the committed store yet.

``probe_remaining`` replays the plan (the same pool, twins and systems as ``run``) against a COPY of the eval store in a
temporary directory. A cached call is answered from the store like in a real run; an uncached call is counted per
stage and answered by a minimal valid stub, so the replay goes on to the next document. Nothing is sent to any
provider, the committed store is never written, and no key or /key reading is needed.

The counts are exact for every document whose stages are cached or whose first miss comes before any stage that
depends on a stubbed answer. After a miss the stubbed document usually stops early (no claims), so its later stages
are added by the caller as an upper bound (``intake_missing``)."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .cache import CachedChat, LLMCache
from .metrics.robustness import run_paired
from .store import EvalStore, store_path

STAGES = {"l1b_security_findings": "l1", "l2_resume_extraction": "l2", "semantic_matches": "match",
          "naive_screener_score": "naive"}
STUBS: dict[str, dict[str, Any]] = {
    "l1b_security_findings": {"findings": []},
    "semantic_matches": {"matches": []},
    "naive_screener_score": {"score": 5.0},
    "l2_resume_extraction": {"identity_envelope": {}, "proposed_claims": [],
                             "extraction_quality": {"confidence": 0.5, "missing_fields": [],
                                                    "needs_human_review": False}},
}


class _ProbeGuard:
    """Nothing is ever sent, so nothing needs authorising. The router makes ``CachedChat`` take the same (routed) path
    as a real run, so the cache keys are the real ones."""

    meter = None

    def __init__(self, router: Any) -> None:
        self.router = router

    def check(self, model: str) -> None:
        return None

    def authorize_paid(self, *a: Any, **k: Any) -> None:
        return None


class _Recording:
    """Wraps a system: the first ``score`` is the clean pool, each later one a twin. Records the misses of each."""

    def __init__(self, inner: Any, counter: Counter[str]) -> None:
        self._inner, self._counter, self.name = inner, counter, inner.name
        self.deltas: list[Counter[str]] = []

    def score(self, candidates: Any, jd: Any) -> Any:
        before = Counter(self._counter)
        out = self._inner.score(candidates, jd)
        self.deltas.append(Counter(self._counter) - before)
        return out


def probe_remaining(settings: Any, store_root: str | Path, model: str, *, matcher: Any = None) -> dict[str, Any] | None:
    """Per LLM system: the uncached calls of the clean pool and of each twin, by stage. None if the probe cannot run."""
    from .runner import _is_llm, _prepare
    from .systems import FullPipeline, NaiveLLMScreener

    src = store_path(store_root)
    if not Path(src).is_file():
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            shutil.copy(src, store_path(tmp))
            store = EvalStore.for_root(tmp)
            try:
                plan = _prepare(settings, tmp, need_paired=True)
                counter: Counter[str] = Counter()

                def stub(messages: Any, *, schema: Any, schema_name: str, model: str | None = None, **kw: Any) -> Any:
                    counter[STAGES.get(schema_name, schema_name)] += 1
                    return dict(STUBS.get(schema_name, {}))

                from .budget import PAID_MODEL, ModelRouter, is_free_model

                router = (ModelRouter([], model, policy="paid-direct") if not is_free_model(model) and model == PAID_MODEL
                          else ModelRouter([model], None))
                chat = CachedChat(LLMCache(store, synthetic=True), _ProbeGuard(router), call=stub,  # type: ignore[arg-type]
                                  default_model=router.current)
                out: dict[str, Any] = {"documents": len(plan["pool"]) + len(plan["cases"]), "pool": len(plan["pool"]),
                                       "twins": len(plan["cases"]), "n_jds": len(plan["jds"]), "systems": {}}
                for name in settings.systems:
                    if not _is_llm(name):
                        continue
                    inner = (NaiveLLMScreener(chat) if name == "naive_llm"
                             else FullPipeline(chat, matcher=matcher, run_seed=settings.seed))
                    counter.clear()
                    rec = _Recording(inner, counter)
                    run_paired(rec, plan["pool"], plan["cases"], plan["jds"][0], pool_independent=True)  # type: ignore[arg-type]
                    n_paired = len(rec.deltas)
                    for jd in plan["jds"][1:]:
                        rec.score(plan["pool"], jd)
                    out["systems"][name] = {"clean": dict(rec.deltas[0]) if rec.deltas else {},
                                            "twins": [dict(d) for d in rec.deltas[1:n_paired]],
                                            "extra_jds": [dict(d) for d in rec.deltas[n_paired:]]}
                return out
            finally:
                store.close()
    except Exception:  # noqa: BLE001 - the probe is advisory: the caller falls back to the ratio estimate
        return None
