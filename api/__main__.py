"""Entry point: `uv run python -m api`. Host, port and auto-reload come from the settings (.env)."""
import sys

import uvicorn

from core.settings import SettingsError, settings

if __name__ == "__main__":
    try:
        host, port, reload = settings.api_host, settings.api_port, settings.api_reload
    except SettingsError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)
