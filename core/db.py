"""
Database access: lazily-built SQLAlchemy engines and session factories for the SQLite file in LCMS_DB_PATH.

Two flavours exist on purpose:
  * async (`async_session_maker`, `get_async_session`) for the FastAPI request handlers;
  * sync (`SyncSessionLocal`, `get_sync_engine`) for pipeline code and the worker processes that execute runs.

Nothing here touches the disk or the settings at import time. The engines are built on first use, so
`import core.db` (and everything that imports it) works without any environment and creates no files.
"""
from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from core.settings import settings
from core.declarative_base import Base


@lru_cache(maxsize=1)
def get_engine():
    """The shared async engine (created once). The 30 s timeout lets writers wait for SQLite's file lock."""
    # SQLite creates the file itself but never the parent directory.
    settings.lcms_db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(settings.db_url, connect_args={"timeout": 30})


@lru_cache(maxsize=1)
def get_sync_engine():
    """The shared synchronous engine for the same database file (created once)."""
    settings.lcms_db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{settings.lcms_db_path}", connect_args={"timeout": 30})


@lru_cache(maxsize=1)
def _async_factory():
    # expire_on_commit=False: objects stay readable after commit, which the request handlers rely on
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@lru_cache(maxsize=1)
def _sync_factory():
    return sessionmaker(bind=get_sync_engine())


def reset_engines() -> None:
    """For tests: drop the cached engines (e.g. after pointing LCMS_DB_PATH somewhere else)."""
    for fn in (get_engine, get_sync_engine, _async_factory, _sync_factory):
        fn.cache_clear()


class _Lazy:
    """Callable/attribute proxy that resolves its target on first use, so module-level names exist before settings do."""

    def __init__(self, getter):
        self._getter = getter

    def __call__(self, *a, **kw):
        return self._getter()(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._getter(), name)


# Same names as before: `SyncSessionLocal()` and `async_session_maker()` still return sessions.
SyncSessionLocal = _Lazy(_sync_factory)
async_session_maker = _Lazy(_async_factory)


async def create_db_and_tables():
    """Create any missing tables from the models (idempotent; does not migrate existing tables). Run at API start-up."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one async session per request, closed afterwards."""
    async with async_session_maker() as session:
        yield session
