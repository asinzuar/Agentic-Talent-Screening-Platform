from fastapi import APIRouter

from agentic_talent_screener.api.routers import jobs, ranking

router = APIRouter()
router.include_router(ranking.router)
router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
