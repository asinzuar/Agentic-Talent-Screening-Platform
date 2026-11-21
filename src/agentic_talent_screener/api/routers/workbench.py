from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

WORKBENCH_HTML = Path(__file__).parent.parent / "static" / "workbench.html"

router = APIRouter(include_in_schema=False)


@router.get("/")
def index() -> FileResponse:
    return FileResponse(WORKBENCH_HTML, media_type="text/html")
