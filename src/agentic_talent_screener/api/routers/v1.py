from fastapi import APIRouter

from agentic_talent_screener.api.routers import ranking

# Versioned API. Add feature routers here, e.g.:
#   from agentic_talent_screener.api.routers import jobs
#   router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
router = APIRouter()
router.include_router(ranking.router)
