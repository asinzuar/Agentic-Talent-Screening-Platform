"""E15b: the SQLite eval store (embeddings, LLM cache, run summaries, spend ledger).

One file, ``eval_store.sqlite3``, at the repo root, committed to git so teammates can clone and re-run
``report`` / ``compare`` with zero API calls.

**The file is public. Nothing secret and nothing about a real person goes in.**
Only synthetic prompts and their responses, hashes, vectors, run summaries and spend totals.
Writers refuse anything that carries a key prefix (``sk-or-``) or comes from a non-synthetic run
(see :func:`assert_storable`). Big per-item raw files stay in git-ignored ``runs/``.

Design for small diffs: WAL while the handle is open (fast writes), then on close a TRUNCATE checkpoint, back to
rollback-journal mode (no ``-wal`` / ``-shm`` files left) and ``VACUUM``. Tables are ordered by primary key and
JSON is canonical. Size budget: ``SIZE_BUDGET_BYTES`` (20 MB; a warning, not an error).

Contract with E15a (``StoreLedger``): ``phase_total(phase) -> float`` is USD, the sum over rows of
``max(usd_measured, usd_estimated)``; ``phase_summary(phase)`` gives the breakdown. ``spend_ledger`` is append-only
(triggers refuse UPDATE and DELETE) and every row has a unique ``entry_id`` so :meth:`EvalStore.merge_from`
(``store merge <other.sqlite3>``) can union two stores without losing or doubling rows. Git cannot merge the
binary file; see ``STORE.md``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import threading
import uuid
import warnings
import weakref
from contextlib import contextmanager
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

STORE_FILE = "eval_store.sqlite3"
SIZE_BUDGET_BYTES = 20 * 1024 * 1024
SECRET_PREFIX = "sk-or-"
# Key shapes refused anywhere in a stored prompt, response or summary (defence in depth for a public file).
SECRET_PATTERNS = (re.compile(r"sk-or-"), re.compile(r"sk-ant-"), re.compile(r"ghp_[A-Za-z0-9]{20,}"),
                   re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), re.compile(r"AKIA[0-9A-Z]{16}"))


class StoreRefused(ValueError):
    """A write was refused: non-synthetic run, or the text carries a key prefix."""


class SizeBudgetWarning(UserWarning):
    """The store file is over its size budget."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def store_path(root: str | Path) -> Path:
    return Path(root).resolve() / STORE_FILE


def repo_root() -> Path:
    """The repo root, found the way the CLI finds it (nearest parent holding ``pyproject.toml``)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def assert_storable(*, synthetic: bool, texts: list[Any]) -> None:
    """Refuse a write that is not from a synthetic run or whose text contains the key prefix."""
    if not synthetic:
        raise StoreRefused("the eval store is public: only runs flagged synthetic may write to it")
    for t in texts:
        blob = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False, default=str)
        if any(p.search(blob) for p in SECRET_PATTERNS):
            raise StoreRefused("refusing to store text that looks like an API key (e.g. the prefix 'sk-or-')")


# --- migrations: each takes a connection and moves the schema from version N to N+1 -----------------------

def _m1(c: sqlite3.Connection) -> None:
    for stmt in (
        """CREATE TABLE embeddings (
        model_id TEXT NOT NULL, text_sha256 TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL,
        created_at TEXT NOT NULL, PRIMARY KEY (model_id, text_sha256)) WITHOUT ROWID""",
        """CREATE TABLE llm_cache (
        key TEXT PRIMARY KEY, model TEXT NOT NULL, schema_name TEXT NOT NULL, response TEXT NOT NULL,
        created_at TEXT NOT NULL) WITHOUT ROWID""",
        """CREATE TABLE runs (
        run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, manifest TEXT NOT NULL, results TEXT NOT NULL,
        git_sha TEXT NOT NULL) WITHOUT ROWID""",
        """CREATE TABLE spend_ledger (
        id INTEGER PRIMARY KEY, entry_id TEXT NOT NULL UNIQUE, run_id TEXT NOT NULL, model TEXT NOT NULL, usd_measured REAL NOT NULL,
        usd_estimated REAL NOT NULL, requests INTEGER NOT NULL, at TEXT NOT NULL, phase TEXT NOT NULL)""",
        "CREATE INDEX spend_phase ON spend_ledger (phase)",
        """CREATE TRIGGER spend_no_update BEFORE UPDATE ON spend_ledger
        BEGIN SELECT RAISE(ABORT, 'spend_ledger is append-only'); END""",
        """CREATE TRIGGER spend_no_delete BEFORE DELETE ON spend_ledger
        BEGIN SELECT RAISE(ABORT, 'spend_ledger is append-only'); END""",
    ):
        c.execute(stmt)


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_m1]
SCHEMA_VERSION = len(MIGRATIONS)


def pack_vector(vec: list[float] | tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EvalStore:
    """Thread-safe handle on the eval DB. Use as a context manager or call :meth:`close`."""

    def __init__(self, path: str | Path, *,
                 migrations: list[Callable[[sqlite3.Connection], None]] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0  # >0 inside batch(): writers join the open transaction
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self.path), timeout=30, check_same_thread=False, isolation_level=None)
        weakref.finalize(self, self._conn.close)  # a dropped handle never leaks the file lock
        self._conn.execute("PRAGMA journal_mode=WAL")  # fast writes; close() converts back (no -wal/-shm left)
        self._conn.execute("PRAGMA synchronous=NORMAL")  # cache rows; ledger rows commit with FULL
        self._migrate(migrations if migrations is not None else MIGRATIONS)

    @classmethod
    def for_root(cls, root: str | Path) -> EvalStore:
        return cls(store_path(root))

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("the store is closed")
        return self._conn

    def _migrate(self, migrations: list[Callable[[sqlite3.Connection], None]]) -> None:
        with self._lock:
            version = self.conn.execute("PRAGMA user_version").fetchone()[0]
            if version > len(migrations):
                raise ValueError(f"store schema v{version} is newer than this code (v{len(migrations)})")
            for n in range(version, len(migrations)):
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    migrations[n](self.conn)
                    self.conn.execute(f"PRAGMA user_version={n + 1}")  # DDL is transactional in SQLite
                    self.conn.execute("COMMIT")
                except BaseException:
                    self.conn.execute("ROLLBACK")
                    raise

    @property
    def schema_version(self) -> int:
        with self._lock:
            return self.conn.execute("PRAGMA user_version").fetchone()[0]

    @contextmanager
    def batch(self) -> Iterator[None]:
        """One transaction for many writes (one commit, one sync). Rolls back on error; nests."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._depth = 0
                self.conn.execute("ROLLBACK")
                raise
            self._depth = 0
            self.conn.execute("COMMIT")

    def _write(self, sql: str, args: tuple[Any, ...], *, durable: bool = False) -> None:
        with self._lock:
            if self._depth:
                self.conn.execute(sql, args)
                return
            if durable:
                self.conn.execute("PRAGMA synchronous=FULL")
            try:
                with self.batch():
                    self.conn.execute(sql, args)
            finally:
                if durable:
                    self.conn.execute("PRAGMA synchronous=NORMAL")

    def _one(self, sql: str, args: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        with self._lock:
            return self.conn.execute(sql, args).fetchone()

    # --- embeddings ---------------------------------------------------------------------------------------
    def put_embedding(self, model_id: str, text_sha256: str, vector: list[float] | tuple[float, ...], *,
                      synthetic: bool = False, created_at: str | None = None) -> None:
        assert_storable(synthetic=synthetic, texts=[model_id, text_sha256])
        if not vector:
            raise ValueError("empty vector")
        self._write("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                    (model_id, text_sha256, len(vector), pack_vector(vector), created_at or now_iso()))

    def get_embedding(self, model_id: str, text_sha256: str) -> list[float] | None:
        row = self._one("SELECT dim, vector FROM embeddings WHERE model_id=? AND text_sha256=?",
                        (model_id, text_sha256))
        if row is None or len(row[1]) != row[0] * 4:
            return None
        return unpack_vector(row[1])

    # --- LLM cache ----------------------------------------------------------------------------------------
    def put_llm(self, key: str, model: str, schema_name: str, response: Any, *, synthetic: bool = False,
                prompt: Any, created_at: str | None = None) -> None:
        """``prompt`` (the messages sent) is required: the secret scan covers prompt and response here, so no
        caller path can skip it."""
        assert_storable(synthetic=synthetic, texts=[response, prompt, model, schema_name])
        self._write("INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?)",
                    (key, model, schema_name, canonical_json(response), created_at or now_iso()))

    def get_llm(self, key: str) -> Any | None:
        row = self._one("SELECT response FROM llm_cache WHERE key=?", (key,))
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    # --- runs ---------------------------------------------------------------------------------------------
    def put_run(self, run_id: str, manifest: dict[str, Any], results: dict[str, Any], git_sha: str, *,
                synthetic: bool = False, created_at: str | None = None) -> None:
        assert_storable(synthetic=synthetic, texts=[manifest, results])
        self._write("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                    (run_id, created_at or now_iso(), canonical_json(manifest), canonical_json(results), git_sha))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT run_id, created_at, manifest, results, git_sha FROM runs WHERE run_id=?", (run_id,))
        if row is None:
            return None
        return {"run_id": row[0], "created_at": row[1], "manifest": json.loads(row[2]),
                "results": json.loads(row[3]), "git_sha": row[4]}

    def list_runs(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT run_id FROM runs ORDER BY run_id")]

    # --- spend ledger (E15a writes here) ------------------------------------------------------------------
    def record_spend(self, run_id: str, model: str, usd_measured: float, usd_estimated: float, requests: int,
                     phase: str, *, at: str | None = None, entry_id: str | None = None) -> None:
        for v in (usd_measured, usd_estimated):
            if not isinstance(v, (int, float)) or v != v or v < 0 or v == float("inf"):
                raise ValueError("spend amounts must be finite numbers >= 0")
        if not isinstance(requests, int) or requests < 0:
            raise ValueError("requests must be an integer >= 0")
        assert_storable(synthetic=True, texts=[run_id, model, phase])
        self._write("INSERT INTO spend_ledger (entry_id, run_id, model, usd_measured, usd_estimated, requests, at, "
                    "phase) VALUES (?,?,?,?,?,?,?,?)",
                    (entry_id or uuid.uuid4().hex, run_id, model, float(usd_measured), float(usd_estimated),
                     requests, at or now_iso(), phase), durable=True)

    def phase_total(self, phase: str) -> float:
        """USD for ``phase``: the sum over rows of ``max(usd_measured, usd_estimated)`` (E15a contract)."""
        row = self._one("SELECT COALESCE(SUM(MAX(usd_measured, usd_estimated)),0) FROM spend_ledger WHERE phase=?",
                        (phase,))
        assert row is not None
        return float(row[0])

    def phase_summary(self, phase: str) -> dict[str, float | int]:
        row = self._one("SELECT COALESCE(SUM(usd_measured),0), COALESCE(SUM(usd_estimated),0), "
                        "COALESCE(SUM(requests),0) FROM spend_ledger WHERE phase=?", (phase,))
        assert row is not None
        return {"usd_measured": row[0], "usd_estimated": row[1], "requests": row[2]}

    # --- merge (git cannot merge the binary file) ----------------------------------------------------------
    def merge_from(self, other: str | Path) -> dict[str, int]:
        """Union another store into this one: ``INSERT OR IGNORE`` per table, dedupe by key (``entry_id`` for the
        ledger). Refuses a store whose schema version differs. Returns rows added per table."""
        other_path = Path(other).resolve()
        if not other_path.is_file():
            raise ValueError(f"no such store: {other}")
        with self._lock:
            probe = sqlite3.connect(str(other_path))
            try:
                version = probe.execute("PRAGMA user_version").fetchone()[0]
            finally:
                probe.close()
            if version != self.schema_version:
                raise ValueError(f"cannot merge a store at schema v{version} into v{self.schema_version}; "
                                 "open it with matching code first")
            self.conn.execute("ATTACH DATABASE ? AS other", (str(other_path),))
            try:
                added: dict[str, int] = {}
                with self.batch():
                    for table, cols in (("embeddings", "model_id, text_sha256, dim, vector, created_at"),
                                        ("llm_cache", "key, model, schema_name, response, created_at"),
                                        ("runs", "run_id, created_at, manifest, results, git_sha"),
                                        ("spend_ledger", "entry_id, run_id, model, usd_measured, usd_estimated, "
                                                         "requests, at, phase")):
                        before = self.conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
                        self.conn.execute(f"INSERT OR IGNORE INTO main.{table} ({cols}) "
                                          f"SELECT {cols} FROM other.{table} ORDER BY {cols.split(',')[0]}")
                        added[table] = self.conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0] - before
            finally:
                self.conn.execute("DETACH DATABASE other")
        return added

    # --- housekeeping -------------------------------------------------------------------------------------
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def check_size(self, budget: int = SIZE_BUDGET_BYTES) -> bool:
        """Warn (not fail) when the file is over budget. Returns True when within budget."""
        size = self.size_bytes()
        if size > budget:
            warnings.warn(f"{self.path.name} is {size / 1048576:.1f} MB, over the {budget / 1048576:.0f} MB budget; "
                          "prune the cache before committing", SizeBudgetWarning, stacklevel=2)
            return False
        return True

    def close(self) -> None:
        """Checkpoint the WAL, return to rollback-journal mode, VACUUM (compact, small diffs), check size, close.
        A lock held by another process only skips the compaction (with a warning). Safe to call twice."""
        with self._lock:
            if self._conn is None:
                return
            try:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    self._conn.execute("PRAGMA journal_mode=DELETE")
                    self._conn.execute("VACUUM")
                except sqlite3.OperationalError as exc:
                    warnings.warn(f"{self.path.name}: could not compact on close ({exc})", stacklevel=2)
                self.check_size()
            finally:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> EvalStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
