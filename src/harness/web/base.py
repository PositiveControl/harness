"""Character-FastAPI factory.

`build_character_app(character, adapter)` is the single entry point.
It assembles a FastAPI app with the base router mounted plus the
character-specific router (if one exists at
`harness.web.characters.<character_name>`).

Discovery is convention-based: a Python module at
`src/harness/web/characters/<name>.py` exports
`build_router(character, adapter) -> APIRouter`. If that module isn't
present, the app still works — it just doesn't expose the extension
endpoints. This keeps the harness primitive runnable against any
character (you get /healthz / /character / /chat / /capabilities for
free) while letting characters add structured endpoints as they
mature.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from harness.character import Character
from harness.turn import TurnService

from .base_router import build_base_router

_DEFAULT_RATE_LIMIT = "120/minute"
"""IP-keyed default. Generous for an invite-list tailnet audience;
character extensions or deployments can tighten via the `rate_limit`
arg to `build_character_app`."""


def build_character_app(
    character: Character,
    adapter: object,
    *,
    rate_limit: str = _DEFAULT_RATE_LIMIT,
    cors_allow_origins: list[str] | None = None,
    title_override: str | None = None,
    turn_service: TurnService | None = None,
) -> FastAPI:
    """Build a FastAPI app exposing `character` over HTTP.

    `adapter` is the model adapter — the base router structurally
    requires `.id` and `.complete(messages, *, max_tokens,
    temperature) -> str`. Character extensions may further require
    `complete_grammar()` (for grammar-constrained endpoints) or other
    structural methods; those checks happen inside the extension's
    `build_router`.

    `cors_allow_origins` defaults to `["*"]` — fine for the tailnet
    SPA dev story (no third-party origin can reach the port; the
    threat is bypassed by the ingress posture). Lock down per
    deployment if you ever expose a public domain.
    """
    limiter = Limiter(key_func=get_remote_address, default_limits=[rate_limit])

    title = title_override or f"harness — {character.name}"
    app = FastAPI(
        title=title,
        version="0.0.1",
        description=character.premise.strip() or f"Character endpoint for {character.name}",
    )
    app.state.limiter = limiter
    app.state.character = character
    app.state.adapter = adapter
    app.state.boot_unix = time.time()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins if cors_allow_origins is not None else ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    app.include_router(
        build_base_router(
            character,
            adapter,
            started_at_unix=app.state.boot_unix,
            turn_service=turn_service,
        )
    )

    extension_router = _discover_character_router(character, adapter)
    if extension_router is not None:
        app.include_router(extension_router)

    return app


def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    """slowapi's exception handler — wraps RateLimitExceeded into a JSON
    429 with the configured limit string in the body so SPAs can
    surface the actual cap rather than guessing."""
    # exc carries the `.detail` from slowapi; type narrowing isn't worth
    # a runtime check for a one-line wrapper.
    detail = getattr(exc, "detail", "rate limit exceeded")
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded", "detail": str(detail)},
    )


def _discover_character_router(character: Character, adapter: object) -> Any:
    """Look up `harness.web.characters.<character_name>` and call its
    `build_router(character, adapter) -> APIRouter`. Returns None when
    the module doesn't exist (every character works with just the
    base router) or when the module exists but doesn't export the
    expected callable (warn-loud-fail-soft so a misconfigured
    extension doesn't crash the whole server)."""
    module_name = f"harness.web.characters.{character.name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None
    builder = getattr(module, "build_router", None)
    if not callable(builder):
        import warnings

        warnings.warn(
            f"harness.web extension module {module_name!r} is missing "
            f"`build_router(character, adapter) -> APIRouter`; serving "
            f"base endpoints only.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return builder(character, adapter)


__all__ = ["build_character_app"]
