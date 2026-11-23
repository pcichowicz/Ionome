"""
FastAPI application: wires the routers, security middleware and static files together.

Run with `uv run python -m api` (see api/__main__.py). The JSON endpoints and the HTML pages share this one app.
Every route requires authentication when API_TOKEN is set (see api/security.py).
"""
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from api.endpoints import pages, results, runs, samples, studies
from api.security import require_auth, security_middleware
from api.services.runs import recover_interrupted_runs
from core.db import create_db_and_tables
from core.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: make sure the database tables exist and clean up after a previous crash. Nothing to do on shutdown."""
    await create_db_and_tables()
    # a restart kills runs started by the previous process; without this they stay RUNNING forever
    recover_interrupted_runs()
    yield


_dev = settings.env == "development"
app = FastAPI(
    title="LCMS Pipeline API",
    lifespan=lifespan,
    dependencies=[Depends(require_auth)],
    # the auto-generated docs are not covered by dependencies: only serve them in development
    docs_url="/docs" if _dev else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _dev else None,
)
app.middleware("http")(security_middleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")

# Include order matters: FastAPI matches routes in the order they are added. pages.router holds the
# "/studies/{study_id}" HTML catch-all, which would swallow the more specific /studies/... JSON routes,
# so it must come last (and inside pages.py "/studies/new" must stay before "/studies/{study_id}").
app.include_router(studies.router)
app.include_router(samples.router)
app.include_router(runs.router)
app.include_router(results.router)
app.include_router(pages.router)
