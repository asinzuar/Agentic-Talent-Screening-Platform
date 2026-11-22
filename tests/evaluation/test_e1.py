import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_talent_screener.evaluation.budget import BudgetExceeded, BudgetGuard
from agentic_talent_screener.evaluation.cache import CacheMiss, CachedChat, LLMCache, cache_key
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
    with pytest.raises(CacheMiss):
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
    assert main(["report", "x"]) == 2


PRICES = {"paid/model": (1.0, 1.0)}  # USD per million tokens


def _paid_chat(tmp_path, cap, results=None):
    guard = BudgetGuard(cap)
    calls = []

    def fake(messages, **kw):
        calls.append(1)
        return results or {"ok": True}

    return CachedChat(LLMCache(tmp_path), guard, call=fake, prices_per_mtok=PRICES), guard, calls


def test_positive_cap_binds(tmp_path):
    chat, guard, calls = _paid_chat(tmp_path, 0.0035)
    # each call estimates 1000 output tokens = $0.001 plus a tiny input cost
    made = 0
    with pytest.raises(BudgetExceeded):
        for i in range(10):
            chat([{"role": "user", "content": f"q{i}"}], schema=SCHEMA, schema_name="s", model="paid/model", max_tokens=1000)
            made += 1
    assert made == len(calls) == 3
    assert 0 < guard.spent_usd <= 0.0035


def test_paid_without_price_or_max_tokens_fails_closed(tmp_path):
    chat, _, calls = _paid_chat(tmp_path, 100.0)
    with pytest.raises(BudgetExceeded):
        chat(MSGS, schema=SCHEMA, schema_name="s", model="unpriced/model", max_tokens=100)
    with pytest.raises(BudgetExceeded):
        chat(MSGS, schema=SCHEMA, schema_name="s", model="paid/model")
    assert not calls


def test_actual_usage_is_recorded(tmp_path):
    usage = {"ok": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
    chat, guard, _ = _paid_chat(tmp_path, 1.0, usage)
    chat(MSGS, schema=SCHEMA, schema_name="s", model="paid/model", max_tokens=1000)
    assert guard.spent_usd == pytest.approx(20e-6)


def test_free_models_cost_nothing(tmp_path):
    chat, guard, _ = _paid_chat(tmp_path, 0.0)
    chat(MSGS, schema=SCHEMA, schema_name="s", model="x:free", max_tokens=1000)
    assert guard.spent_usd == 0


def test_cache_key_covers_strict_and_reasoning(tmp_path):
    n = []
    chat = CachedChat(LLMCache(tmp_path), BudgetGuard(), call=lambda *a, **k: n.append(1) or {})
    for kw in ({}, {"strict": True}, {"reasoning": {"effort": "low"}}):
        chat(MSGS, schema=SCHEMA, schema_name="s", model="a:free", **kw)
    assert len(n) == 3


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_guard_and_config_reject_non_finite(bad):
    with pytest.raises(ValueError):
        BudgetGuard(bad)
    with pytest.raises(ValueError):
        EvalConfig(budget_usd=bad)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "-1", "abc"])
def test_cli_rejects_bad_budget(bad, capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--budget-usd", bad])
    assert e.value.code == 2


def test_run_manifest_marks_skeleton(tmp_path):
    main(["run", "--runs-dir", str(tmp_path), "--suite", "robustness", "--offline"])
    m = json.loads((next(tmp_path.iterdir()) / "manifest.json").read_text())
    assert m["status"] == "skeleton" and m["suites_executed"] == [] and m["suites"] == ["robustness"]


BOUNDARY = re.compile(
    r"^\s*(from\s+(agentic_talent_screener|\.+)\s+import\s+[^\n]*\bevaluation\b"
    r"|from\s+(agentic_talent_screener\.|\.+)evaluation\b"
    r"|import\s+agentic_talent_screener\.evaluation\b)",
    re.M,
)


@pytest.mark.parametrize(
    "line",
    [
        "from agentic_talent_screener import evaluation",
        "from agentic_talent_screener import llm, evaluation",
        "from .. import evaluation",
        "from ..evaluation import x",
        "from agentic_talent_screener.evaluation.cache import x",
        "import agentic_talent_screener.evaluation",
    ],
)
def test_boundary_regex_flags_forms(line):
    assert BOUNDARY.search(line)


def test_api_does_not_import_evaluation():
    api = Path(__file__).resolve().parents[2] / "src" / "agentic_talent_screener" / "api"
    files = list(api.rglob("*.py"))
    assert files, "scanned no files; has api/ moved?"
    assert not [str(p) for p in files if BOUNDARY.search(p.read_text(encoding="utf-8"))]


def test_api_app_import_loads_no_evaluation_module():
    code = (
        "import sys, agentic_talent_screener.api.app;"
        "bad=[k for k in sys.modules if k.startswith('agentic_talent_screener.evaluation')];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
