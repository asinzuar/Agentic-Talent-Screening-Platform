"""Tamper-evident audit ledger (SPEC §6.3 audit trail, §10 override reasons, L5).

Each event stores the SHA-256 of its canonical JSON plus the previous event's hash, so editing,
removing or reordering any past event breaks ``verify_chain()``. This is tamper-*evident*, not
tamper-proof: whoever controls the storage can rewrite the whole chain, so anchor the head hash
elsewhere if that matters.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import FORBIDDEN_IDENTITY_FIELDS, PoolRanking, RequirementProfile

GENESIS_HASH = "0" * 64
_CANDIDATE_ID = re.compile(r"^C-\d{4}$")

EventType = Literal["profile_approved", "pool_scored", "hr_override", "benchmark_run"]


def _find_forbidden(obj: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_IDENTITY_FIELDS:
                bad.append(f"{path}{k}")
            bad.extend(_find_forbidden(v, f"{path}{k}."))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(_find_forbidden(v, f"{path}{i}."))
    return bad


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    timestamp: datetime
    event_type: EventType
    actor: str = Field(min_length=1)
    details: dict[str, Any]
    prev_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("details")
    @classmethod
    def _no_identity_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        bad = _find_forbidden(v)
        if bad:
            raise ValueError(f"identity/PII keys are not allowed in audit details: {bad}")
        return v

    @staticmethod
    def compute_hash(
        *,
        event_id: str,
        timestamp: datetime,
        event_type: str,
        actor: str,
        details: Mapping[str, Any],
        prev_hash: str,
    ) -> str:
        payload = {
            "event_id": event_id,
            "timestamp": timestamp.astimezone(UTC).isoformat(),
            "event_type": event_type,
            "actor": actor,
            "details": details,
            "prev_hash": prev_hash,
        }
        return hashlib.sha256((_canonical(payload) + prev_hash).encode("utf-8")).hexdigest()

    def recompute_hash(self) -> str:
        return self.compute_hash(
            event_id=self.event_id,
            timestamp=self.timestamp,
            event_type=self.event_type,
            actor=self.actor,
            details=self.details,
            prev_hash=self.prev_hash,
        )


class AuditLedger:
    """Append-only, hash-chained in-memory ledger (thread-safe)."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    # ---- reading -------------------------------------------------------------------
    @property
    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def head_hash(self) -> str:
        with self._lock:
            return self._events[-1].event_hash if self._events else GENESIS_HASH

    def __len__(self) -> int:
        return len(self._events)

    # ---- appending -----------------------------------------------------------------
    def _append(self, event_type: EventType, actor: str, details: dict[str, Any]) -> AuditEvent:
        actor = (actor or "").strip()
        if not actor:
            raise ValueError("actor must not be blank")
        bad = _find_forbidden(details)
        if bad:
            raise ValueError(f"identity/PII keys are not allowed in audit details: {bad}")
        with self._lock:
            prev = self._events[-1].event_hash if self._events else GENESIS_HASH
            event_id, ts = str(uuid4()), datetime.now(UTC)
            digest = AuditEvent.compute_hash(
                event_id=event_id, timestamp=ts, event_type=event_type,
                actor=actor, details=details, prev_hash=prev,
            )
            event = AuditEvent(
                event_id=event_id, timestamp=ts, event_type=event_type, actor=actor,
                details=details, prev_hash=prev, event_hash=digest,
            )
            self._events.append(event)
            return event

    def record_profile_approval(self, profile: RequirementProfile) -> AuditEvent:
        if not profile.approved or not profile.approved_by or not profile.approved_at:
            raise ValueError("only an approved requirement profile can be recorded as approved")
        return self._append(
            "profile_approved",
            profile.approved_by,
            {
                "profile_id": profile.profile_id,
                "profile_version": profile.version,
                "approved_at": profile.approved_at.astimezone(UTC).isoformat(),
                "active_requirements": len(profile.active_requirements),
            },
        )

    def record_pool_scoring(
        self,
        pool: PoolRanking,
        actor: str = "system",
        parameters: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        details: dict[str, Any] = {
            "profile_id": pool.profile_id,
            "profile_version": pool.profile_version,
            "config_version": pool.config_version,
            "catalogue_version": pool.catalogue_version,
            "n_scored": pool.n_scored,
            "n_errors": pool.n_errors,
            "n_flagged": pool.n_flagged,
            "ranking": [
                {"candidate_id": r.candidate_id, "rank": r.rank, "fit_score": r.fit_score}
                for r in pool.results
            ],
        }
        if parameters:
            details["parameters"] = json.loads(json.dumps(parameters))
        return self._append("pool_scored", actor, details)

    def record_benchmark(
        self,
        benchmark: Mapping[str, Any],
        actor: str = "system",
        parameters: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        """Log a compact summary of an evaluation run (metrics, lift, adversarial outcome)."""
        k = benchmark["k"]
        details: dict[str, Any] = {
            "k": k,
            "profile_id": benchmark["profile"]["profile_id"],
            "profile_version": benchmark["profile"]["version"],
            "config_version": benchmark["config"]["config_version"],
            "metrics": {
                name: {f"ndcg_at_{k}": s["metrics"][f"ndcg_at_{k}"],
                       "spearman_rho": s["metrics"]["spearman_rho"],
                       "kendall_tau": s["metrics"]["kendall_tau"],
                       "top1_hit": s["metrics"]["top1_hit"]}
                for name, s in benchmark["systems"].items()
            },
            "lift": benchmark["lift"],
            "cheaters_blocked": benchmark["adversarial"]["cheaters_blocked"],
            "ground_truth": benchmark["ground_truth"],
        }
        if parameters:
            details["parameters"] = json.loads(json.dumps(parameters))
        return self._append("benchmark_run", actor, json.loads(json.dumps(details)))

    def record_hr_override(
        self, candidate_id: str, actor: str, action: str, reason: str
    ) -> AuditEvent:
        if not (reason or "").strip():
            raise ValueError("a non-blank reason is required for HR actions (SPEC §10)")
        if not (action or "").strip():
            raise ValueError("action must not be blank")
        if not _CANDIDATE_ID.match(candidate_id or ""):
            raise ValueError("candidate_id must be a blind ID like C-0932")
        return self._append(
            "hr_override",
            actor,
            {"candidate_id": candidate_id, "action": action.strip(), "reason": reason.strip()},
        )

    # ---- integrity -----------------------------------------------------------------
    def verify_chain(self) -> bool:
        prev = GENESIS_HASH
        for e in self.events:
            if e.prev_hash != prev or e.recompute_hash() != e.event_hash:
                return False
            prev = e.event_hash
        return True

    # ---- (de)serialisation ---------------------------------------------------------
    def to_jsonl(self) -> str:
        return "\n".join(_canonical(e.model_dump(mode="json")) for e in self.events)

    @classmethod
    def from_jsonl(cls, text: str | Iterable[str]) -> AuditLedger:
        lines = text.splitlines() if isinstance(text, str) else list(text)
        ledger = cls()
        for line in lines:
            if line.strip():
                ledger._events.append(AuditEvent.model_validate_json(line))
        return ledger
