"""E15b: the SQLite eval store, the LLMCache migration, runs in the DB. Offline."""

import json
import sqlite3
import subprocess
import warnings

import pytest

from agentic_talent_screener.evaluation import report
from agentic_talent_screener.evaluation.budget import BudgetGuard
from agentic_talent_screener.evaluation.cache import CachedChat, LLMCache, cache_key
from agentic_talent_screener.evaluation.store import (
    STORE_FILE,
    EvalStore,
    SizeBudgetWarning,
    StoreRefused,
    repo_root,
)

FREE = "v/a:free"
MSGS = [{"role": "user", "content": "synthetic prompt"}]
SCHEMA = {"type": "object"}


def test_round_trip_each_table(tmp_path):
    with EvalStore(tmp_path / STORE_FILE) as s:
        s.put_embedding("m", "ab" * 32, [0.5, -1.0, 2.25], synthetic=True)
        s.put_llm("k", FREE, "sch", {"a": [1, 2]}, synthetic=True, prompt=MSGS)
        s.put_run("r1", {"seed": 0}, {"metrics": {}}, "abc123", synthetic=True)
        s.record_spend("r1", FREE, 0.0, 0.01, 3, "dev")
        s.record_spend("r2", FREE, 0.5, 0.25, 2, "dev")
        s.record_spend("r2", FREE, 1.0, 1.0, 1, "other")
    with EvalStore(tmp_path / STORE_FILE) as s:
        assert s.get_embedding("m", "ab" * 32) == [0.5, -1.0, 2.25]
        assert s.get_embedding("m", "zz") is None
        assert s.get_llm("k") == {"a": [1, 2]} and s.get_llm("nope") is None
        run = s.get_run("r1")
        assert run["manifest"] == {"seed": 0} and run["git_sha"] == "abc123" and s.list_runs() == ["r1"]
        assert s.get_run("zz") is None
        tot = s.phase_total("dev")
        assert tot["usd_measured"] == 0.5 and tot["requests"] == 5 and abs(tot["usd_estimated"] - 0.26) < 1e-9
        assert s.phase_total("none") == {"usd_measured": 0, "usd_estimated": 0, "requests": 0}


def test_spend_validation(tmp_path):
    with EvalStore(tmp_path / STORE_FILE) as s:
        for bad in (-1, float("nan"), float("inf"), "x"):
            with pytest.raises(ValueError):
                s.record_spend("r", "m", bad, 0, 1, "p")
        with pytest.raises(ValueError):
            s.record_spend("r", "m", 0, 0, -1, "p")


def test_migration_from_older_version(tmp_path):
    from agentic_talent_screener.evaluation import store as st

    p = tmp_path / STORE_FILE
    EvalStore(p).close()
    seen = []

    def m2(c):
        seen.append(1)
        c.execute("CREATE TABLE extra (x)")

    with EvalStore(p, migrations=[*st.MIGRATIONS, m2]) as s:
        assert s.schema_version == 2 and seen == [1]
    with EvalStore(p, migrations=[*st.MIGRATIONS, m2]):  # idempotent
        assert seen == [1]
    with pytest.raises(ValueError, match="newer"):
        EvalStore(p)  # code knows only v1


def test_failed_migration_rolls_back(tmp_path):
    from agentic_talent_screener.evaluation import store as st

    def bad(c):
        c.execute("CREATE TABLE half (x)")
        raise RuntimeError("boom")

    p = tmp_path / STORE_FILE
    EvalStore(p).close()
    with pytest.raises(RuntimeError):
        EvalStore(p, migrations=[*st.MIGRATIONS, bad])
    with EvalStore(p) as s:
        assert s.schema_version == 1
        assert not s.conn.execute("SELECT 1 FROM sqlite_master WHERE name='half'").fetchall()


def test_cache_hit_after_restart(tmp_path):
    calls = []

    def fake(messages, **kw):
        calls.append(1)
        return {"ok": True}

    c1 = LLMCache(tmp_path)
    assert CachedChat(c1, BudgetGuard(), call=fake)(MSGS, schema=SCHEMA, schema_name="s", model=FREE) == {"ok": True}
    c1.close()
    c2 = LLMCache(tmp_path)
    off = CachedChat(c2, BudgetGuard(), offline=True, call=fake)
    assert off(MSGS, schema=SCHEMA, schema_name="s", model=FREE) == {"ok": True}
    assert len(calls) == 1 and c2.hits == 1
    row = c2.store.conn.execute("SELECT model, schema_name FROM llm_cache").fetchone()
    assert row == (FREE, "s")
    c2.close()


def test_refuses_non_synthetic_and_secrets(tmp_path):
    key = cache_key(FREE, "s", SCHEMA, MSGS)
    ns = LLMCache(tmp_path, synthetic=False)
    with pytest.raises(StoreRefused):
        ns.put(key, {"a": 1}, model=FREE, schema_name="s", messages=MSGS)
    c = LLMCache(tmp_path)
    with pytest.raises(StoreRefused):
        c.put(key, {"a": "sk-or-v1-abc"}, model=FREE, schema_name="s", messages=MSGS)
    with pytest.raises(StoreRefused):
        c.put(key, {"a": 1}, model=FREE, schema_name="s", messages=[{"role": "user", "content": "key sk-or-x"}])
    chat = CachedChat(c, BudgetGuard(), call=lambda *a, **k: {"leak": "sk-or-zzz"})
    with pytest.raises(StoreRefused):
        chat(MSGS, schema=SCHEMA, schema_name="s", model=FREE)
    assert c.get(key) is None
    with pytest.raises(StoreRefused):
        c.store.put_embedding("m", "h", [1.0])  # not flagged synthetic
    with pytest.raises(StoreRefused):
        c.store.put_run("r", {"k": "sk-or-1"}, {}, "x", synthetic=True)


def test_size_warning(tmp_path):
    with EvalStore(tmp_path / STORE_FILE) as s:
        s.put_llm("k", "m", "s", {"a": 1}, synthetic=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert s.check_size() is True
        with pytest.warns(SizeBudgetWarning):
            assert s.check_size(budget=10) is False


def test_no_wal_or_shm_left_after_close(tmp_path):
    p = tmp_path / STORE_FILE
    s = EvalStore(p)
    s.put_llm("k", "m", "s", {"a": 1}, synthetic=True)
    mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
    s.close()
    s.close()  # twice is fine
    assert mode == "delete"
    assert sorted(x.name for x in tmp_path.iterdir()) == [STORE_FILE]


def test_deterministic_file_for_same_writes(tmp_path):
    def build(d):
        d.mkdir()
        with EvalStore(d / STORE_FILE) as s:
            for k in ("b", "a", "c"):
                s.put_llm(k, "m", "s", {"z": 1, "a": 2}, synthetic=True, created_at="t")
        return (d / STORE_FILE).read_bytes()

    assert build(tmp_path / "x") == build(tmp_path / "y")


def test_run_in_db_and_compare_by_id(tmp_path):
    def res(v):
        return {"run_id": "x", "metrics": {"m": {"value": v, "verdict": "pass", "direction": "higher_better"}}}

    with EvalStore.for_root(tmp_path) as s:
        s.put_run("a1", {}, res(0.9), "g", synthetic=True)
        s.put_run("b1", {}, res(0.5), "g", synthetic=True)
    d = report.compare_runs(tmp_path / "runs" / "a1", tmp_path / "runs" / "b1", store_root=tmp_path)
    assert d["n_regressions"] == 1
    with pytest.raises(ValueError, match="not in the eval store"):
        report.compare_runs(tmp_path / "runs" / "a1", tmp_path / "runs" / "zz", store_root=tmp_path)
    (tmp_path / "run_file").mkdir()
    (tmp_path / "run_file" / "results.json").write_text(json.dumps(res(0.9)))
    d = report.compare_runs(tmp_path / "run_file", tmp_path / "runs" / "a1", store_root=tmp_path)
    assert d["n_regressions"] == 0


def test_compare_without_store_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="no eval store"):
        report.compare_runs(tmp_path / "a", tmp_path / "b", store_root=tmp_path)


def test_gitignore_keeps_db_ignores_journals():
    root = repo_root()

    def ignored(name):
        return subprocess.run(["git", "check-ignore", "-q", name], cwd=root).returncode == 0

    assert not ignored(STORE_FILE)
    assert ignored(STORE_FILE + "-wal") and ignored(STORE_FILE + "-shm")


def test_store_is_sqlite_file(tmp_path):
    EvalStore(tmp_path / STORE_FILE).close()
    con = sqlite3.connect(tmp_path / STORE_FILE)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"embeddings", "llm_cache", "runs", "spend_ledger"} <= names
