import json
import re
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation.budget import BudgetExceeded, BudgetGuard
from agentic_talent_screener.evaluation.cache import CachedChat, LLMCache, cache_key
from agentic_talent_screener.evaluation.cli import main
from agentic_talent_screener.evaluation.config import EvalConfig
from agentic_talent_screener.evaluation.manifest import build_manifest, sha256_file, write_manifest

SCHEMA = {"type": "object"}
MSGS = [{"role": "user", "content": "hi"}]


def test_config_defaults_and_validation():
    cfg = EvalConfig()
    assert cfg.budget_usd == 0 and cfg.seed == 0
    with pytest.raises(ValueError):
        EvalConfig(budget_usd=-1)
    with pytest.raises(ValueError):
        EvalConfig(bogus=1)


def test_budget_default_allows_only_free():
    g = BudgetGuard()
    g.check("google/gemma-4-31b-it:free")
    with pytest.raises(BudgetExceeded):
        g.check("google/gemma-4-31b-it")
    BudgetGuard(1.0).check("paid/model", 0.5)
    with pytest.raises(BudgetExceeded):
        BudgetGuard(1.0).check("paid/model", 2.0)
    with pytest.raises(ValueError):
        BudgetGuard(-1)


def test_cache_key_stable_and_distinct():
    assert cache_key("m", "s", SCHEMA, MSGS) == cache_key("m", "s", SCHEMA, MSGS)
    assert cache_key("m", "s", SCHEMA, MSGS) != cache_key("m2", "s", SCHEMA, MSGS)


def test_cached_chat_hits_and_offline(tmp_path):
    calls = []

    def fake(messages, **kw):
        calls.append(kw["model"])
        return {"ok": True}

    cache = LLMCache(tmp_path)
    chat = CachedChat(cache, BudgetGuard(), call=fake)
    assert chat(MSGS, schema=SCHEMA, schema_name="s", model="a:free") == {"ok": True}
    assert chat(MSGS, schema=SCHEMA, schema_name="s", model="a:free") == {"ok": True}
    assert len(calls) == 1 and cache.hits == 1 and cache.misses == 1
    off = CachedChat(LLMCache(tmp_path), BudgetGuard(), offline=True, call=fake)
    assert off(MSGS, schema=SCHEMA, schema_name="s", model="a:free") == {"ok": True}
    with pytest.raises(RuntimeError):
        off(MSGS, schema=SCHEMA, schema_name="other", model="a:free")


def test_cached_chat_default_model_and_paid_block(tmp_path):
    chat = CachedChat(LLMCache(tmp_path), BudgetGuard(), call=lambda *a, **k: {}, default_model=lambda: "paid/model")
    with pytest.raises(BudgetExceeded):
        chat(MSGS, schema=SCHEMA, schema_name="s")


def test_cache_corrupt_entry_is_miss(tmp_path):
    cache = LLMCache(tmp_path)
    (tmp_path / "k.json").write_text("{not json", encoding="utf-8")
    assert cache.get("k") is None


def test_manifest(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("abc")
    m = build_manifest(seed=3, models=["b", "a", "a"], datasets=[f])
    assert m["models"] == ["a", "b"] and m["datasets"]["d.txt"] == sha256_file(f)
    p = write_manifest(tmp_path / "run", m)
    assert json.loads(p.read_text())["seed"] == 3


def test_cli_run_writes_manifest(tmp_path):
    assert main(["run", "--runs-dir", str(tmp_path)]) == 0
    assert (next(tmp_path.iterdir()) / "manifest.json").exists()
    assert main(["run", "--budget-usd", "-1"]) == 2
    assert main(["report", "x"]) == 2


def test_api_does_not_import_evaluation():
    api = Path(__file__).resolve().parents[2] / "src" / "agentic_talent_screener" / "api"
    pat = re.compile(r"^\s*(from|import)\s+(agentic_talent_screener\.evaluation|\.+evaluation)\b", re.M)
    offenders = [str(p) for p in api.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders
