"""Generalizable character-FastAPI factory (harness-3jz1.9).

Turns any harness character into a web-served FastAPI app. The TFR
explainer (harness-3jz1.4) is the first concrete extension; future
character-as-web-tool work (CFI cite tool, debrief copilot, returns
handler, etc.) plugs in the same way by dropping a module under
`harness.web.characters.<name>` that exports a `build_router(character,
adapter) -> APIRouter` callable.

Public surface:
    build_character_app(character, adapter) -> FastAPI

Base endpoints every served character exposes:
    GET  /healthz       — liveness {status, character, model, uptime_s}
    GET  /character     — full character sheet as JSON
    POST /chat          — single-turn {text} -> {reply}; PersonaAdapter-
                          wrapped when the character ships one. Today
                          this is the no-tool primitive; the full tool-
                          loop / contract-prelude grounding stack used
                          by `harness chat` is a follow-up extraction.
    GET  /capabilities  — list of every endpoint the app exposes

Generalizable helpers also live here:
    tool_registry.build_request_registry — request-scoped tool
                          bindings (used by character extensions that
                          need per-request tool behavior, e.g. the TFR
                          explainer binding `read_parsed_notam` to the
                          request's parsed NOTAM struct).
    grammar.complete_json — schema-constrained JSON output via the
                          adapter's `complete_grammar()`. The harness
                          model adapter boundary is preserved — this
                          helper calls the adapter through its public
                          protocol; outlines / MLX-internals stay
                          behind the adapter.
    logging.RequestDebugLog — per-request capture under
                          HARNESS_WEB_DEBUG_DIR for replay and fixture
                          growth.

Ingress posture is Tailscale-only: the harness web command defaults to
binding `0.0.0.0` and relies on the host firewall + Tailscale ACLs to
constrain who can reach the port. No public domain in v1.
"""

from .base import build_character_app

__all__ = ["build_character_app"]
