"""
Application settings (paths, database, API, workers), read from environment variables and the repo-root `.env`.

Two design points:
  * Settings are built lazily, on first attribute access, so importing any module never needs (or reads) an
    environment. A missing/invalid value raises `SettingsError` with a readable message instead of a pydantic
    traceback.
  * Relative paths in `.env` are resolved against the *current working directory*, so run every command from
    the repository root (or use absolute paths).

Typical use: `from core.settings import settings; settings.lcms_results_dir`.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    """
    Every field maps to one environment variable (its `alias`). Fields marked `...` are required and have no
    default, so a fresh checkout fails loudly until `.env` is created from `.env.example`.
    """
    model_config = SettingsConfigDict(
        env_file = PROJECT_ROOT / ".env",
        env_file_encoding = "utf-8",
        populate_by_name = True,
        )

    env: str = "development"
    pipeline_name: str = Field(..., alias="PIPELINE_NAME")

    # "local" (default) or "s3". The S3 backend in core.storage is a placeholder and not functional yet.
    storage_backend: str = Field("local", alias="STORAGE_BACKEND")

    # Where the data lives. Raw files and sample sheet: <raw>/<study_id>/ ; outputs: <results>/<study_id>/ ;
    # registered study configs: <yaml_dir>/<study_id>.yaml
    lcms_raw_data_dir: Path = Field(..., alias="LCMS_RAW_DATA_DIR")
    lcms_results_dir: Path = Field(..., alias="LCMS_RESULTS_DIR")
    lcms_reference_library_dir: Path = Field(..., alias="LCMS_REFERENCE_LIBRARY_DIR")
    yaml_dir: Path = Field(..., alias="YAML_DIR")
    # Kept so existing .env files still load; the pipeline now always uses core.paths.RESULT_SUBDIRS.
    output_dirs: list[str] = Field(["features", "logs", "plots", "library", "qc_reports", "cache"], alias="OUTPUT_DIRS")

    aws_s3_bucket: str | None = Field(None, alias="AWS_S3_BUCKET")
    aws_region: str | None = Field(None, alias="AWS_REGION")

    # SQLite database holding studies, samples and run history.
    lcms_db_name:str = Field(..., alias="LCMS_DB_NAME")
    lcms_db_path: Path = Field(..., alias="LCMS_DB_PATH")
    lcms_db_echo: bool = Field(False, alias="LCMS_DB_ECHO")   # log every SQL statement (debugging only)

    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(..., alias="API_RELOAD")          # uvicorn auto-reload; development only
    # If set, every page and API call needs it as the password of HTTP Basic auth (any user name).
    # Required when API_HOST is not a loopback address.
    api_token: str | None = Field(None, alias="API_TOKEN")
    # Host names the server answers to (comma separated); blocks DNS-rebinding requests to a local server.
    api_allowed_hosts: str = Field("localhost,127.0.0.1", alias="API_ALLOWED_HOSTS")

    # Size of the process pool that executes pipeline runs started from the web app.
    lcms_max_workers: int = Field(4, alias="LCMS_MAX_WORKERS")
    lcms_cache_enabled:bool = Field(True, alias="LCMS_CACHE_ENABLED")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    api_log_info: str = "info"
    log_format: str = "json"

    @property
    def allowed_hosts(self) -> list[str]:
        """`API_ALLOWED_HOSTS` split into a clean list (blank entries dropped)."""
        return [h.strip() for h in self.api_allowed_hosts.split(",") if h.strip()]

    @property
    def db_url(self):
        """Async SQLAlchemy URL for the SQLite file at `lcms_db_path` (the web app's connection)."""
        return f"sqlite+aiosqlite:///{self.lcms_db_path}"

    def model_post_init(self, __context) -> None:
        """Cross-field validation: rules that depend on more than one variable."""
        if self.storage_backend == "s3" and not self.aws_s3_bucket:
            raise ValueError("AWS_S3_BUCKET is required when STORAGE_BACKEND=s3")
        # The API can write files and start runs, so a non-loopback bind without a token would expose both.
        if self.api_host not in ("127.0.0.1", "localhost", "::1") and not self.api_token:
            raise ValueError(
                f"API_HOST={self.api_host} is reachable from other machines and the API can write files and "
                "start pipeline runs: set API_TOKEN, or bind to 127.0.0.1"
            )

class SettingsError(RuntimeError):
    """The settings are missing or invalid. The message says what and how to fix it (no pydantic traceback)."""


def _explain(exc: ValidationError) -> str:
    """Turn a pydantic ValidationError into a short, actionable message naming each bad variable and the `.env` path."""
    env_file = PROJECT_ROOT / ".env"
    problems = []
    for err in exc.errors():
        name = ".".join(str(x) for x in err["loc"]) or "settings"
        problems.append(f"  - {name}: " + ("not set" if err["type"] == "missing" else err["msg"]))
    where = f"{env_file} ({'found' if env_file.is_file() else 'NOT FOUND'})"
    hint = ("Create it from the template and fill in the paths:\n    cp .env.example .env\n"
            "(or export the variables in your shell). See GETTING_STARTED.md, step 4a."
            if not env_file.is_file() else "Fix or add the values listed above in that file.")
    return f"Ionome cannot start: {len(problems)} setting(s) missing or invalid.\n" + "\n".join(problems) + \
        f"\nSettings are read from the environment and from {where}.\n{hint}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    The process-wide Settings object, built on first use (not on import) and cached.

    Raises:
        SettingsError: a required variable is missing or a value is invalid.
    """
    load_dotenv()  # core.paths / core.storage still read os.environ
    try:
        return Settings()
    except ValidationError as exc:
        raise SettingsError(_explain(exc)) from None


def reset_settings() -> None:
    """For tests: forget the cached settings so the next access re-reads the environment."""
    get_settings.cache_clear()


class _LazySettings:
    """
    `from core.settings import settings` keeps working; the real object is created at first attribute access.
    Assigning an attribute (monkeypatch.setattr(settings, "env", "production") in tests) overrides it on this proxy
    only, and `del` / monkeypatch undo removes the override again; the underlying Settings object is never mutated.
    """

    def __getattr__(self, name):           # only called when the proxy itself has no such attribute
        return getattr(get_settings(), name)

    def __repr__(self):
        return repr(get_settings())


settings = _LazySettings()
