import argparse
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from threading import Event, Thread
from time import monotonic, sleep

from agentic_talent_screener.api.config import get_settings
from agentic_talent_screener.file_logging import exception_site, log_to_file
from agentic_talent_screener.job_analysis import JobAnalysisError
from agentic_talent_screener.llm import LOGGER as MODEL_LOGGER, model_log_context
from agentic_talent_screener.workflow import intake, requirements, scoring
from agentic_talent_screener.workflow.storage import SYSTEM_ACTOR, Store, WorkflowError, audit, encode, new_id, utc_now

WORKFLOW_LOGGER = logging.getLogger("agentic_talent_screener.workflow")
LOGGER = WORKFLOW_LOGGER.getChild("worker")

QUEUES = {
    "requirement_drafts": (("job_id", "draft_id"), requirements.process_draft),
    "intake_attempts": (("job_id", "candidate_id", "attempt_id"), intake.process_intake),
    "scoring_runs": (("job_id", "run_id"), scoring.process_run),
}
LEASE = timedelta(seconds=60)
PROGRESS_INTERVAL = 5.0


class LeaseLost(Exception):
    """Another worker claimed this operation or it already finished."""


@dataclass(frozen=True)
class Task:
    store: Store
    table: str
    keys: tuple
    token: str
    recovered: bool = False

    @property
    def log_context(self) -> str:
        identifiers = " ".join(f"{column}={value}" for column, value in zip(QUEUES[self.table][0], self.keys))
        return f"queue={self.table} {identifiers}"

    @property
    def predicate(self) -> str:
        columns = QUEUES[self.table][0]
        return " AND ".join(f"{column} = ?" for column in columns) + " AND state = 'running' AND lease_token = ?"

    def current(self, db: sqlite3.Connection) -> sqlite3.Row:
        row = db.execute(
            f"SELECT * FROM {self.table} WHERE {self.predicate} AND lease_until > ?",
            (*self.keys, self.token, utc_now()),
        ).fetchone()
        if row is None:
            raise LeaseLost
        return row

    def _set(self, db: sqlite3.Connection, fields: dict) -> None:
        assignments = ", ".join(f"{field} = ?" for field in fields)
        changed = db.execute(
            f"UPDATE {self.table} SET {assignments} WHERE {self.predicate} AND lease_until > ?",
            (*fields.values(), *self.keys, self.token, utc_now()),
        ).rowcount
        if changed != 1:
            raise LeaseLost

    def update(self, **fields) -> None:
        with self.store.transaction(write=True) as db:
            row = self.current(db)
            stage = fields.get("stage")
            stage_changed = stage is not None and row["stage"] != stage
            if stage_changed:
                audit(db, self.keys[0], SYSTEM_ACTOR, "processing_stage",
                      dict(zip(QUEUES[self.table][0], self.keys)), {"stage": stage})
            self._set(db, fields)
        if stage_changed:
            LOGGER.info("Workflow task stage: %s stage=%s", self.log_context, stage)

    def finish(self, db: sqlite3.Connection, **fields) -> None:
        self._set(db, {**fields, "lease_token": None, "lease_until": None})


def claim(store: Store) -> tuple[Task, Callable[[Task], None]] | None:
    with store.transaction(write=True) as db:
        now = utc_now()
        choices = []
        for table, (columns, process) in QUEUES.items():
            row = db.execute(
                f"""SELECT * FROM {table} WHERE state = 'queued' OR (state = 'running' AND lease_until <= ?)
                ORDER BY created_at LIMIT 1""", (now,),
            ).fetchone()
            if row is not None:
                choices.append((row["created_at"], table, columns, row, process))
        if not choices:
            return None
        _, table, columns, row, process = min(choices, key=lambda choice: choice[:2])
        token = new_id()
        values = {"state": "running", "lease_token": token, "lease_until": utc_now(LEASE)}
        if table == "scoring_runs":
            values["started_at"] = row["started_at"] or now
        elif table == "requirement_drafts":
            values.update(revision_token=new_id(), stage="extract_jd")
        else:
            values["stage"] = "document_safety"
        assignments = ", ".join(f"{column} = ?" for column in values)
        predicate = " AND ".join(f"{column} = ?" for column in columns)
        keys = tuple(row[column] for column in columns)
        db.execute(f"UPDATE {table} SET {assignments} WHERE {predicate}", (*values.values(), *keys))
        recovered = row["state"] == "running"
        audit(db, row["job_id"], SYSTEM_ACTOR, "task_claimed", dict(zip(columns, keys)),
              {"queue": table, "recovered": recovered})
        return Task(store, table, keys, token, recovered), process


def renew(task: Task, stop: Event) -> None:
    while not stop.wait(LEASE.total_seconds() / 4):
        try:
            task.update(lease_until=utc_now(LEASE))
        except LeaseLost:
            return
        except sqlite3.OperationalError as error:
            LOGGER.warning("Could not renew a workflow lease; will retry: %s sqlite_error=%s",
                           task.log_context, getattr(error, "sqlite_errorname", "unknown"))


def report_progress(task: Task, stop: Event, started_at: float) -> None:
    while not stop.wait(PROGRESS_INTERVAL):
        try:
            with task.store.transaction() as db:
                row = task.current(db)
                if task.table == "scoring_runs":
                    run = scoring.run_view(db, *task.keys)
                    scored, reused, failed, skipped = (
                        run[key] for key in ("scored_count", "reused_count", "error_count", "skipped_count")
                    )
                    progress = (
                        f"processed={scored + reused + failed + skipped}/{run['ready_count']} "
                        f"scored={scored} reused={reused} failed={failed} skipped={skipped}"
                    )
                else:
                    progress = f"stage={row['stage']}"
        except LeaseLost:
            return
        except sqlite3.OperationalError:
            LOGGER.warning("Could not read workflow task progress: %s", task.log_context)
            continue
        if not stop.is_set():
            LOGGER.info("Workflow task progress: %s %s elapsed=%.0fs", task.log_context, progress, monotonic() - started_at)


def fail(task: Task, failure: Exception, elapsed: float) -> None:
    safe = failure if isinstance(failure, WorkflowError) else WorkflowError(
        503, "PROCESSING_FAILED", "Processing could not finish. Retry the operation.", retryable=True,
    )
    with task.store.transaction(write=True) as db:
        row = task.current(db)
        stage = "score_pool" if task.table == "scoring_runs" else row["stage"]
        fields = {"error_json": encode(safe.body)}
        if task.table == "requirement_drafts":
            fields.update(state="failed", stage=None, revision_token=new_id())
        elif task.table == "intake_attempts":
            fields.update(state="finished", outcome="manual_review", hold_code=safe.body["code"],
                          retryable=int(safe.body["retryable"]), finished_at=utc_now())
            db.execute("UPDATE candidates SET revision_token = ? WHERE (job_id, candidate_id, current_attempt_id) = (?, ?, ?)",
                       (new_id(), *task.keys))
        else:
            fields.update(state="failed", finished_at=utc_now())
        task.finish(db, **fields)
        audit(db, row["job_id"], SYSTEM_ACTOR, "task_failed",
              dict(zip(QUEUES[task.table][0], task.keys)), {"queue": task.table, "code": safe.body["code"]})
    reason = failure.diagnostic if isinstance(failure, JobAnalysisError) else safe.body["code"]
    LOGGER.warning("Workflow task failed: %s stage=%s reason=%s retryable=%s error=%s at=%s elapsed=%.1fs",
                   task.log_context, stage, reason, safe.body["retryable"], type(failure).__name__,
                   exception_site(failure), elapsed)


def run_once(store: Store) -> bool:
    picked = claim(store)
    if not picked:
        return False
    task, process = picked
    stop = Event()
    started_at = monotonic()
    heartbeat = Thread(target=renew, args=(task, stop), daemon=True)
    reporter = Thread(target=report_progress, args=(task, stop, started_at), daemon=True)
    LOGGER.info("Workflow task started: %s recovered=%s", task.log_context, task.recovered)
    heartbeat.start()
    reporter.start()
    try:
        with model_log_context(task.log_context):
            process(task)
    except LeaseLost:
        LOGGER.info("Workflow task lease was lost: %s", task.log_context)
        return True
    except Exception as error:
        try:
            fail(task, error, monotonic() - started_at)
        except LeaseLost:
            LOGGER.warning("Could not record workflow failure after losing the lease: %s error=%s at=%s",
                           task.log_context, type(error).__name__, exception_site(error))
        except sqlite3.Error as storage_error:
            LOGGER.warning("Could not record workflow failure: %s error=%s at=%s sqlite_error=%s",
                           task.log_context, type(error).__name__, exception_site(error),
                           getattr(storage_error, "sqlite_errorname", "unknown"))
            raise
        return True
    finally:
        stop.set()
        heartbeat.join()
        reporter.join()
    LOGGER.info("Workflow task finished: %s elapsed=%.0fs", task.log_context, monotonic() - started_at)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Process durable job analysis, intake, and scoring tasks.")
    parser.add_argument("--once", action="store_true", help="Process one available task, then exit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    store = Store(settings.workflow_db)
    with log_to_file(settings.log_dir / "worker.log", WORKFLOW_LOGGER, MODEL_LOGGER):
        LOGGER.info("Workflow worker started")
        try:
            if args.once:
                run_once(store)
                return
            while True:
                try:
                    worked = run_once(store)
                except sqlite3.OperationalError as error:
                    LOGGER.warning("Workflow database is temporarily unavailable: sqlite_error=%s",
                                   getattr(error, "sqlite_errorname", "unknown"))
                    worked = False
                if not worked:
                    sleep(0.5)
        except KeyboardInterrupt:
            return
        finally:
            LOGGER.info("Workflow worker stopped")


if __name__ == "__main__":
    main()
