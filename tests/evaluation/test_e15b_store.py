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
        assert abs(s.phase_total("dev") - 0.51) < 1e-9 and isinstance(s.phase_total("dev"), float)
        tot = s.phase_summary("dev")
        assert tot["usd_measured"] == 0.5 and tot["requests"] == 5 and abs(tot["usd_estimated"] - 0.26) < 1e-9
        assert s.phase_total("none") == 0.0


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

    c1 = LLMCache(tmp_path, synthetic=True)
    assert CachedChat(c1, BudgetGuard(), call=fake)(MSGS, schema=SCHEMA, schema_name="s", model=FREE) == {"ok": True}
    c1.close()
    c2 = LLMCache(tmp_path, synthetic=True)
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
    c = LLMCache(tmp_path, synthetic=True)
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
        s.put_llm("k", "m", "s", {"a": 1}, synthetic=True, prompt=[])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert s.check_size() is True
        with pytest.warns(SizeBudgetWarning):
            assert s.check_size(budget=10) is False


def test_no_wal_or_shm_left_after_close(tmp_path):
    p = tmp_path / STORE_FILE
    s = EvalStore(p)
    s.put_llm("k", "m", "s", {"a": 1}, synthetic=True, prompt=[])
    mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
    s.close()
    s.close()  # twice is fine
    assert mode == "wal"  # WAL while open; converted back on close
    assert sorted(x.name for x in tmp_path.iterdir()) == [STORE_FILE]
    with open(p, "rb") as fh:
        assert fh.read(20)[18:20] == b""  # rollback-journal format, not WAL


def test_deterministic_file_for_same_writes(tmp_path):
    def build(d):
        d.mkdir()
        with EvalStore(d / STORE_FILE) as s:
            for k in ("b", "a", "c"):
                s.put_llm(k, "m", "s", {"z": 1, "a": 2}, synthetic=True, prompt=[], created_at="t")
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


def test_secret_check_is_inside_the_store_write(tmp_path):
    with EvalStore(tmp_path / STORE_FILE) as s:
        with pytest.raises(TypeError):
            s.put_llm("k", "m", "s", {"a": 1}, synthetic=True)  # prompt is required: no way to skip the scan
        for bad in ("sk-or-v1-abc", "sk-ant-abc", "ghp_" + "a" * 24, "AKIA" + "A" * 16):
            with pytest.raises(StoreRefused):
                s.put_llm("k", "m", "s", {"a": 1}, synthetic=True, prompt=[{"content": bad}])
            with pytest.raises(StoreRefused):
                s.put_llm("k", "m", "s", {"a": bad}, synthetic=True, prompt=[])
        assert s.get_llm("k") is None


def test_llmcache_put_requires_context_and_synthetic_flag(tmp_path):
    with pytest.raises(TypeError):
        LLMCache(tmp_path, synthetic=True).put("k", {"a": 1})  # model/schema_name/messages are required
    c = LLMCache(tmp_path)  # default: refuse
    assert c.synthetic is False
    with pytest.raises(StoreRefused):
        c.put("k", {"a": 1}, model="m", schema_name="s", messages=[])


def test_batched_writes_are_fast_and_atomic(tmp_path):
    import time

    with EvalStore(tmp_path / STORE_FILE) as s:
        t0 = time.perf_counter()
        for i in range(200):
            s.put_llm(f"k{i}", "m", "s", {"i": i, "pad": "x" * 2000}, synthetic=True, prompt=[])
        assert time.perf_counter() - t0 < 10  # was ~680 ms per row (fsync per row); generous bound
        with pytest.raises(RuntimeError):
            with s.batch():
                s.put_llm("b1", "m", "s", {}, synthetic=True, prompt=[])
                raise RuntimeError("abort")
        assert s.get_llm("b1") is None
        with s.batch():
            with s.batch():  # nests
                s.put_llm("b2", "m", "s", {}, synthetic=True, prompt=[])
        assert s.get_llm("b2") == {}


def test_spend_ledger_append_only_and_unique_ids(tmp_path):
    with EvalStore(tmp_path / STORE_FILE) as s:
        s.record_spend("r", "m", 0, 0.1, 1, "p", entry_id="e1")
        with pytest.raises(sqlite3.IntegrityError):
            s.record_spend("r", "m", 0, 0.1, 1, "p", entry_id="e1")
        with pytest.raises(sqlite3.DatabaseError):
            s.conn.execute("UPDATE spend_ledger SET usd_measured=0")
        with pytest.raises(sqlite3.DatabaseError):
            s.conn.execute("DELETE FROM spend_ledger")
        s.record_spend("r", "m", 0, 0.1, 1, "p")
        s.record_spend("r", "m", 0, 0.1, 1, "p")
        assert s.phase_summary("p")["requests"] == 3


def test_merge_unions_without_losing_ledger_rows(tmp_path):
    def fill(name, entries, key):
        with EvalStore(tmp_path / name) as s:
            for e in entries:
                s.record_spend("r", "m", 0.1, 0.2, 1, "ph", entry_id=e)
            s.put_llm(key, "m", "s", {"k": key}, synthetic=True, prompt=[])
            s.put_run("run-" + key, {}, {}, "g", synthetic=True)

    fill("a.sqlite3", ["e1", "e2"], "ka")
    fill("b.sqlite3", ["e2", "e3"], "kb")  # e2 is in both
    with EvalStore(tmp_path / "a.sqlite3") as s:
        added = s.merge_from(tmp_path / "b.sqlite3")
        assert added["spend_ledger"] == 1 and added["llm_cache"] == 1 and added["runs"] == 1
        assert abs(s.phase_total("ph") - 0.6) < 1e-9  # e1, e2, e3: none dropped, none doubled
        assert s.get_llm("kb") == {"k": "kb"} and s.list_runs() == ["run-ka", "run-kb"]
        assert s.merge_from(tmp_path / "b.sqlite3") == {"embeddings": 0, "llm_cache": 0, "runs": 0, "spend_ledger": 0}
        with pytest.raises(ValueError, match="no such store"):
            s.merge_from(tmp_path / "missing.sqlite3")
    other = sqlite3.connect(tmp_path / "b.sqlite3")
    other.execute("PRAGMA user_version=99")
    other.commit()
    other.close()
    with EvalStore(tmp_path / "a.sqlite3") as s:
        with pytest.raises(ValueError, match="schema v99"):
            s.merge_from(tmp_path / "b.sqlite3")


def test_cli_store_merge_and_stats(tmp_path, monkeypatch, capsys):
    from agentic_talent_screener.evaluation import cli

    monkeypatch.setattr(cli, "_project_root", lambda: tmp_path)
    with EvalStore(tmp_path / "o.sqlite3") as s:
        s.record_spend("r", "m", 0.1, 0.1, 1, "ph")
    assert cli.main(["store", "merge", str(tmp_path / "o.sqlite3")]) == 0
    assert "spend_ledger +1" in capsys.readouterr().out
    assert cli.main(["store", "stats"]) == 0
    assert "schema v1" in capsys.readouterr().out
    assert cli.main(["store", "merge"]) == 2


def test_docs_and_attributes_exist():
    from pathlib import Path

    root = repo_root()
    assert "eval_store.sqlite3 binary" in (root / ".gitattributes").read_text()
    doc = (Path(cli_dir := root / "src/agentic_talent_screener/evaluation") / "STORE.md").read_text(encoding="utf-8")
    assert cli_dir.is_dir()
    for needle in ("public", "synthetic", "store merge", "--cache-dir", "deprecated and ignored"):
        assert needle in doc
    from agentic_talent_screener.evaluation import cli

    help_text = cli.build_parser().format_help() + " ".join(
        a.help or "" for sp in cli.build_parser()._subparsers._group_actions[0].choices.values() for a in sp._actions)
    assert "deprecated and ignored" in help_text
