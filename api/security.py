"""
Access control for the API and the pages it serves.

- require_auth:   HTTP Basic with API_TOKEN as the password, when a token is configured
- SecurityMiddleware: blocks cross-site state-changing requests and adds browser hardening headers
"""
from __future__ import annotations

import secrets
from urllib.parse import urlparse

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from core.settings import settings

_basic = HTTPBasic(auto_error=False)

# no inline scripts or styles anywhere in the templates, so nothing needs 'unsafe-inline'
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


async def require_auth(request: Request) -> None:
    """Applied to the whole app. No-op without API_TOKEN; otherwise any user name with the token as password is accepted."""
    if not settings.api_token:
        return
    creds: HTTPBasicCredentials | None = await _basic(request)
    if creds is None or not secrets.compare_digest(creds.password.encode(), settings.api_token.encode()):
        raise HTTPException(status_code=401, detail="authentication required",
                            headers={"WWW-Authenticate": 'Basic realm="ionome"'})


async def security_middleware(request: Request, call_next):
    """Refuse cross-site writes, then add the security headers to every response."""
    if request.method in _UNSAFE_METHODS:
        # A page on another site can make the browser send a POST to this server (and attach saved credentials).
        # Browsers label such requests: Origin / Sec-Fetch-Site. Same-origin pages and non-browser clients pass.
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host", ""):
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
    response = await call_next(request)
    if request.url.path not in ("/docs", "/openapi.json"):      # Swagger UI needs inline scripts and a CDN
        response.headers.setdefault("Content-Security-Policy", CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    if request.url.path.startswith("/static/"):
        # development convenience kept from the previous middleware: edits to JS/CSS show up on reload
        response.headers["Cache-Control"] = "no-store"
    return response
