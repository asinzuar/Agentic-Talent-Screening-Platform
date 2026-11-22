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


@pytest.mark.parametrize(
    "model,ok",
    [
        ("google/gemma-4-31b-it:free", True),
        ("meta-llama/llama-3.3-70b-instruct:free", True),
        ("openai/gpt-4o:nitro:free", False),
        ("openai/gpt-4o\x00:free", False),
        ("openai/gpt-4o\n:free", False),
        ("openai/gpt-4o:free\n", False),
        ("paid?:free", False),
        (":free", False),
        ("model:free", False),
        ("a/b/c:free", False),
        ("a/b:free ", False),
        ("a/b:FREE", False),
        ("a/b\u00e9:free", False),
        ("a/b:free:nitro", False),
        ("a/b", False),
    ],
)
def test_free_model_pattern_is_strict(model, ok):
    from agentic_talent_screener.evaluation.budget import is_free_model

    assert is_free_model(model) is ok
    if not ok:
        with pytest.raises(BudgetExceeded):
            BudgetGuard().check(model)


def test_default_llm_model_passes_guard():
    from agentic_talent_screener.llm import DEFAULT_LLM_MODEL

    BudgetGuard().check(DEFAULT_LLM_MODEL)


def test_guard_allows_only_free():
    g = BudgetGuard()
    g.check("google/gemma-4-31b-it:free")
    for bad in ("google/gemma-4-31b-it", "", "x:free-not", "paid/model"):
        with pytest.raises(BudgetExceeded):
            g.check(bad)
    with pytest.raises(ValueError):
        BudgetGuard(1.0)
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
    assert chat(MSGS, schema=SCHEMA, schema_name="s", model="v/a:free") == {"ok": True}
    assert chat(MSGS, schema=SCHEMA, schema_name="s", model="v/a:free") == {"ok": True}
    assert len(calls) == 1 and cache.hits == 1 and cache.misses == 1
    off = CachedChat(LLMCache(tmp_path), BudgetGuard(), offline=True, call=fake)
    assert off(MSGS, schema=SCHEMA, schema_name="s", model="v/a:free") == {"ok": True}
    with pytest.raises(CacheMiss):
        off(MSGS, schema=SCHEMA, schema_name="other", model="v/a:free")


def test_paid_model_refused_without_http(tmp_path):
    called = []
    chat = CachedChat(LLMCache(tmp_path), BudgetGuard(), call=lambda *a, **k: called.append(1) or {})
    with pytest.raises(BudgetExceeded):
        chat(MSGS, schema=SCHEMA, schema_name="s", model="paid/model")
    with pytest.raises(BudgetExceeded):
        CachedChat(LLMCache(tmp_path), BudgetGuard(), call=lambda *a, **k: called.append(1) or {}, default_model=lambda: "paid/model")(
            MSGS, schema=SCHEMA, schema_name="s"
        )
    assert not called


def test_paid_cache_entry_is_not_served(tmp_path):
    cache = LLMCache(tmp_path)
    cache.put(cache_key("paid/model", "s", SCHEMA, MSGS), {"x": 1})
    with pytest.raises(BudgetExceeded):
        CachedChat(cache, BudgetGuard(), offline=True)(MSGS, schema=SCHEMA, schema_name="s", model="paid/model")


def test_env_override_to_paid_model_refused(tmp_path, monkeypatch):
    import requests

    monkeypatch.setenv("ATS_LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def no_http(*a, **k):
        raise AssertionError("HTTP attempted")

    monkeypatch.setattr(requests, "post", no_http)
    monkeypatch.setattr(requests.Session, "request", no_http)
    with pytest.raises(BudgetExceeded):
        CachedChat(LLMCache(tmp_path), BudgetGuard())(MSGS, schema=SCHEMA, schema_name="s")


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


@pytest.fixture
def runs_tmp():
    """A throwaway dir under the (git-ignored) project runs/, since --runs-dir must stay under runs/."""
    import shutil
    import uuid

    from agentic_talent_screener.evaluation.cli import _project_root

    d = _project_root() / "runs" / f"test-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_cli_run_writes_manifest(runs_tmp):
    tmp_path = runs_tmp
    assert main(["run", "--runs-dir", str(tmp_path)]) == 0
    assert (next(tmp_path.iterdir()) / "manifest.json").exists()
    assert main(["report", "x"]) == 2


def test_cache_key_covers_strict_and_reasoning(tmp_path):
    n = []
    chat = CachedChat(LLMCache(tmp_path), BudgetGuard(), call=lambda *a, **k: n.append(1) or {})
    for kw in ({}, {"strict": True}, {"reasoning": {"effort": "low"}}):
        chat(MSGS, schema=SCHEMA, schema_name="s", model="v/a:free", **kw)
    assert len(n) == 3


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0, 0.01])
def test_guard_and_config_reject_non_finite(bad):
    with pytest.raises(ValueError):
        BudgetGuard(bad)
    with pytest.raises(ValueError):
        EvalConfig(budget_usd=bad)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "-1", "abc", "0.01", "1"])
def test_cli_rejects_bad_budget(bad, capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--budget-usd", bad])
    assert e.value.code == 2


def test_run_manifest_marks_skeleton(runs_tmp):
    tmp_path = runs_tmp
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
