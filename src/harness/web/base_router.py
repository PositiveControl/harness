"""Base router with the endpoints every served character exposes.

`build_base_router(character, adapter, *, started_at_unix)` returns a
`fastapi.APIRouter` carrying:

    GET  /healthz      — liveness probe for launchd / systemd / curl
    GET  /character    — full character sheet as JSON
    POST /chat         — single-turn conversational reply
    GET  /capabilities — list of mounted endpoints (introspected from
                         the FastAPI app at request time)

`/chat` is the no-tool conversational primitive — it runs the
character's voice rewriter (PersonaAdapter for Airton-family
characters) but NOT the orchestrator's tool loop or the
`assemble_context` contract prelude. The full grounding stack used by
`harness chat` lives inside the CLI today; lifting it into a shared
service layer that this endpoint can call is tracked as a follow-up
under harness-3jz1.9.x.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from harness.character import Character
from harness.model.adapter import ChatMessage
from harness.persona import PersonaAdapter

from .logging import RequestDebugLog


class ChatRequest(BaseModel):
    """POST /chat body."""

    text: str = Field(..., min_length=1, description="The user message.")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """POST /chat response body."""

    reply: str
    character: str
    model: str


def build_base_router(
    character: Character,
    adapter: object,
    *,
    started_at_unix: float | None = None,
) -> APIRouter:
    """Construct the base router. `adapter` is structurally typed —
    the router only requires `.id` and `.complete(messages, *,
    max_tokens, temperature) -> str`."""
    router = APIRouter()
    boot_ts = started_at_unix if started_at_unix is not None else time.time()

    @router.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "character": character.name,
            "model": getattr(adapter, "id", type(adapter).__name__),
            "uptime_s": round(time.time() - boot_ts, 3),
        }

    @router.get("/character")
    def character_sheet() -> dict[str, Any]:
        return _serialize_character(character)

    @router.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        # Wrap the adapter in PersonaAdapter when the character ships
        # one. Characters with voice_rewriter != "persona" (caveman or
        # none) skip the wrap; their /chat is the bare-adapter reply.
        # No tool loop, no forced contract prelude — see module
        # docstring for the limitation.
        runtime_adapter: object = adapter
        if character.voice_rewriter == "persona":
            runtime_adapter = PersonaAdapter(adapter, character)  # type: ignore[arg-type]

        messages = [
            ChatMessage(role="system", content=character.system_prompt()),
            ChatMessage(role="user", content=body.text),
        ]
        with RequestDebugLog.start(character_name=character.name, endpoint="/chat") as log:
            log.raw_input = {"text": body.text}
            reply = runtime_adapter.complete(  # type: ignore[attr-defined]
                messages, max_tokens=body.max_tokens, temperature=body.temperature
            )
            log.raw_model_completion = reply
        return ChatResponse(
            reply=reply,
            character=character.name,
            model=getattr(adapter, "id", type(adapter).__name__),
        )

    @router.get("/capabilities")
    def capabilities(request: Request) -> dict[str, Any]:
        """Enumerate the mounted endpoints. Drives capability-aware
        SPAs so a new character extension shows up in the UI without
        front-end changes."""
        endpoints: list[dict[str, Any]] = []
        for route in request.app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if path is None or methods is None:
                continue
            # Skip FastAPI's own internals (/openapi.json, /docs, /redoc).
            if path in {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}:
                continue
            endpoints.append(
                {
                    "path": path,
                    "methods": sorted(m for m in methods if m != "HEAD"),
                    "summary": getattr(route, "summary", None) or _route_name(route),
                }
            )
        endpoints.sort(key=lambda e: e["path"])
        return {
            "character": character.name,
            "endpoints": endpoints,
        }

    return router


def _route_name(route: object) -> str:
    endpoint = getattr(route, "endpoint", None)
    return getattr(endpoint, "__name__", "") if endpoint else ""


def _serialize_character(character: Character) -> dict[str, Any]:
    """Render the character sheet as a JSON-friendly dict. Mirrors
    `harness describe` output but as structured data rather than the
    rich console rendering."""
    return {
        "name": character.name,
        "premise": character.premise,
        "self_awareness": character.self_awareness,
        "values": [{"id": v.id, "rule": v.rule} for v in character.values],
        "taboos": list(character.taboos),
        "directives": list(character.directives),
        "deep_domains": list(character.deep_domains),
        "shallow_domains": list(character.shallow_domains),
        "on_being_wrong": character.on_being_wrong,
        "voice_rewriter": character.voice_rewriter,
        "default_contract_role": character.default_contract_role,
        "require_assemble_context": character.require_assemble_context,
        "require_search_memory": character.require_search_memory,
        "lead_with_citation": character.lead_with_citation,
        "scope_redirect_template": character.scope_redirect_template,
        "catchers": list(character.catchers),
    }


__all__ = ["ChatRequest", "ChatResponse", "build_base_router"]


# Suppress unused-import lint: JSONResponse is re-exported for character
# extension modules that import it via `from harness.web.base_router
# import JSONResponse` rather than reaching into fastapi directly.
_ = JSONResponse
