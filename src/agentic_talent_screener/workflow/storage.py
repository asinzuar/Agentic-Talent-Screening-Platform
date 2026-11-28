import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from agentic_talent_screener.ranking.audit import GENESIS_HASH, AuditEvent

SYSTEM_ACTOR = "system"


def new_id() -> str:
    return uuid4().hex


def utc_now(after: timedelta = timedelta()) -> str:
    return (datetime.now(UTC) + after).isoformat(timespec="microseconds")


def encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


class WorkflowError(Exception):
    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.body = {"code": code, "message": message, "retryable": retryable}


def require(row: sqlite3.Row | None) -> sqlite3.Row:
    if row is None:
        raise WorkflowError(404, "NOT_FOUND", "Resource not found.")
    return row


def expect_revision(actual: str, expected: str) -> None:
    if actual != expected:
        raise WorkflowError(409, "STALE_REVISION", "Reload and confirm the current revision.")


class Store:
    def __init__(self, path: Path):
        self.path = path
        self._initialized = False
        self._lock = Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA synchronous = FULL")
        try:
            yield db
        finally:
            db.close()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Create the file with restricted permissions before SQLite opens it.
            self.path.touch(mode=0o600, exist_ok=True)
            with self._connect() as db:
                db.execute("PRAGMA journal_mode = WAL")
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
                    db.executescript(
                        f"BEGIN IMMEDIATE;\n{schema}\n"
                        "INSERT OR IGNORE INTO actors VALUES ('system', 'internal', 'system', NULL);\n"
                        "INSERT OR IGNORE INTO workflow_secrets VALUES ('score_plan', randomblob(32));\n"
                        "PRAGMA user_version = 1;\nCOMMIT;"
                    )
                elif version != 1:
                    raise RuntimeError(f"Unsupported workflow schema version: {version}")
            self._initialized = True

    @contextmanager
    def transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        self.initialize()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise


def audit(
    db: sqlite3.Connection, job_id: str, actor: str, event_type: str,
    references: dict, details: dict | None = None,
) -> None:
    last = db.execute(
        "SELECT event_no, event_hash FROM audit_events WHERE job_id = ? ORDER BY event_no DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    number = last["event_no"] + 1 if last else 1
    prev = last["event_hash"] if last else GENESIS_HASH
    event_id, timestamp = new_id(), utc_now()
    details = details or {}
    hashed_details = {"job_id": job_id, "event_no": number, "references": references, "details": details}
    event_hash = AuditEvent.compute_hash(
        event_id=event_id, timestamp=datetime.fromisoformat(timestamp), event_type=event_type,
        actor=actor, details=hashed_details, prev_hash=prev,
    )
    db.execute(
        "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, number, event_id, event_type, actor, timestamp, encode(references), encode(details), prev, event_hash),
    )


def events_for(db: sqlite3.Connection, job_id: str, reference: str, value: str) -> list[dict]:
    rows = db.execute(
        """SELECT * FROM audit_events WHERE job_id = ? AND json_extract(references_json, ?) = ?
        ORDER BY event_no""",
        (job_id, f"$.{reference}", value),
    )
    result = []
    for row in rows:
        event = dict(row)
        event["references"] = json.loads(event.pop("references_json"))
        event["details"] = json.loads(event.pop("details_json"))
        result.append(event)
    return result
