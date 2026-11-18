"""FastAPI backend for the HR What-If workbench.

Run:  uv run uvicorn agentic_talent_screener.ranking.api:app --reload
State (the audit ledger) is in memory only; persistence is an open question (SPEC §12 Q12).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .audit import AuditEvent, AuditLedger
from .config import ComponentWeights, ScoringConfig
from .eval_harness import run_benchmark, sample_ground_truth
from .sample_data import sample_catalogue, sample_pool, sample_requirement_profile
from .schemas import PoolRanking
from .scorer import score_pool

WORKBENCH_HTML = Path(__file__).with_name("workbench.html")

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: ComponentWeights = ComponentWeights()  # validated: each 0-1, sum to 1.0
    exclude_flagged: bool = True
    actor: NonBlank = "system"


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: ComponentWeights = ComponentWeights()
    ground_truth: (
        dict[
            Annotated[str, StringConstraints(pattern=r"^C-\d{4}$")],
            Annotated[float, Field(ge=0.0, le=5.0)],
        ]
        | None
    ) = None
    actor: NonBlank = "system"


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: Annotated[str, StringConstraints(pattern=r"^C-\d{4}$")]
    actor: NonBlank
    action: NonBlank
    reason: NonBlank = Field(description="Required, non-blank (SPEC §10)")


def create_app() -> FastAPI:
    """Build an app with its own in-memory ledger seeded with the sample profile approval."""
    app = FastAPI(title="Agentic Talent Screener: HR Workbench")
    profile = sample_requirement_profile()
    catalogue = sample_catalogue()
    known_ids = {c.candidate_id for c in sample_pool()}

    ledger = AuditLedger()
    ledger.record_profile_approval(profile)
    app.state.ledger = ledger

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WORKBENCH_HTML, media_type="text/html")

    @app.get("/api/sample")
    def sample() -> dict[str, Any]:
        cfg = ScoringConfig()
        ranking = score_pool(sample_pool(), profile, catalogue, cfg)
        return {
            "config": cfg.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "ranking": ranking.model_dump(mode="json"),
        }

    @app.post("/api/rank", response_model=PoolRanking)
    def rank(req: RankRequest) -> PoolRanking:
        cfg = ScoringConfig(
            weights=req.weights,
            exclude_flag_risks=("medium", "high") if req.exclude_flagged else (),
        )
        try:
            result = score_pool(sample_pool(), profile, catalogue, cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ledger.record_pool_scoring(
            result,
            actor=req.actor,
            parameters={
                "weights": req.weights.model_dump(),
                "exclude_flagged": req.exclude_flagged,
            },
        )
        return result

    @app.get("/api/benchmark")
    def benchmark_default() -> dict[str, Any]:
        """Read-only default benchmark (not logged), so read-only users can view results."""
        return run_benchmark(sample_pool(), profile, catalogue, ScoringConfig(), sample_ground_truth())

    @app.post("/api/benchmark")
    def benchmark(req: BenchmarkRequest) -> dict[str, Any]:
        cfg = ScoringConfig(weights=req.weights)
        truth = req.ground_truth if req.ground_truth is not None else sample_ground_truth()
        result = run_benchmark(sample_pool(), profile, catalogue, cfg, truth)
        ledger.record_benchmark(
            result,
            actor=req.actor,
            parameters={
                "weights": req.weights.model_dump(),
                "custom_ground_truth": req.ground_truth is not None,
            },
        )
        return result

    @app.post("/api/override")
    def override(req: OverrideRequest) -> dict[str, Any]:
        if req.candidate_id not in known_ids:
            raise HTTPException(status_code=404, detail="unknown candidate_id")
        try:
            event: AuditEvent = ledger.record_hr_override(
                req.candidate_id, req.actor, req.action, req.reason
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return event.model_dump(mode="json")

    @app.get("/api/audit")
    def audit() -> dict[str, Any]:
        return {
            "chain_valid": ledger.verify_chain(),
            "events": [e.model_dump(mode="json") for e in ledger.events],
        }

    @app.get("/api/audit/export")
    def audit_export() -> Response:
        return Response(
            content=ledger.to_jsonl() + "\n",
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="audit_ledger.jsonl"'},
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
