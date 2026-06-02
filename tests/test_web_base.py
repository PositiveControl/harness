"""Tests for the generalizable character-FastAPI factory (harness-3jz1.9).

Character-agnostic — exercises the base endpoints against a real
loaded character (airton_c_tfr) plus the echo adapter so the test
suite stays fast and offline. The TFR character extension integration
tests live under harness-3jz1.4.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from harness.character import Character, load_character
from harness.config import settings
from harness.model.adapter import ChatMessage
from harness.model.echo import EchoAdapter
from harness.web import build_character_app


@pytest.fixture(name="character")
def fixture_character() -> Character:
    """Use airton_c_tfr — it's our concrete reference character. Other
    characters work the same way (the base layer is character-agnostic);
    we just need one valid character on disk."""
    return load_character(settings.root / "character" / "airton_c_tfr")


@pytest.fixture(name="client")
def fixture_client(character: Character) -> TestClient:
    app = build_character_app(character, EchoAdapter())
    return TestClient(app)


def test_healthz_returns_status_and_character(client: TestClient, character: Character) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["character"] == character.name
    assert body["model"] == "echo"
    assert isinstance(body["uptime_s"], float)
    assert body["uptime_s"] >= 0.0


def test_character_endpoint_returns_full_sheet(client: TestClient, character: Character) -> None:
    resp = client.get("/character")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == character.name
    assert body["premise"] == character.premise
    assert isinstance(body["values"], list)
    assert all({"id", "rule"} == set(v) for v in body["values"])
    assert isinstance(body["taboos"], list)
    assert "deep_domains" in body
    assert "scope_redirect_template" in body  # set on airton_c_tfr
    # airton_c_tfr inherits airton_c's contract pattern.
    assert body["require_assemble_context"] is True


def test_chat_round_trips_through_echo_adapter(client: TestClient) -> None:
    resp = client.post("/chat", json={"text": "ping"})
    assert resp.status_code == 200
    body = resp.json()
    # EchoAdapter returns its input; PersonaAdapter wraps when the
    # character ships voice_rewriter == "persona" (airton_c_tfr does).
    # We don't pin the exact reply shape (rewriter passes can mutate
    # it); the contract is just that we get *a* string reply back.
    assert isinstance(body["reply"], str)
    assert len(body["reply"]) > 0
    assert body["character"] == "airton_c_tfr"
    assert body["model"] == "echo"


def test_chat_rejects_empty_text(client: TestClient) -> None:
    resp = client.post("/chat", json={"text": ""})
    assert resp.status_code == 422  # pydantic validation


def test_chat_uses_grounded_turn_service_when_wired(character: Character, tmp_path: object) -> None:
    """harness-fl313: when build_character_app is handed a TurnService,
    /chat runs the grounded path (transcript persistence + audit) rather
    than the bare system+complete primitive. Persisting the turn is the
    observable proof the service ran."""
    from pathlib import Path

    from harness.cli import _RetrievalState
    from harness.store.transcript import Transcript
    from harness.turn import TurnContext, TurnService

    assert isinstance(tmp_path, Path)
    transcript = Transcript(tmp_path / "web.sqlite")
    service = TurnService(
        TurnContext(
            character=character,
            adapter=EchoAdapter(),
            transcript=transcript,
            load_history=lambda: (None, []),
            speaker="web-user",
            session="web-s1",
            channel="web",
            retrieval_state=_RetrievalState(),
        )
    )
    app = build_character_app(character, EchoAdapter(), turn_service=service)
    client = TestClient(app)

    resp = client.post("/chat", json={"text": "hello over http"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["reply"], str)
    assert len(body["reply"]) > 0

    # The grounded path persisted the turn — the bare path would not have.
    rows = transcript.tail("web-s1", limit=10)
    roles = [(r.role, r.speaker) for r in rows]
    assert ("user", "web-user") in roles
    assert ("assistant", character.name) in roles
    assert any("hello over http" in r.content for r in rows)


def test_capabilities_lists_base_endpoints(client: TestClient) -> None:
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["character"] == "airton_c_tfr"
    paths = {e["path"]: e for e in body["endpoints"]}
    assert "/healthz" in paths
    assert "/character" in paths
    assert "/chat" in paths
    assert "/capabilities" in paths
    assert paths["/chat"]["methods"] == ["POST"]
    assert paths["/healthz"]["methods"] == ["GET"]


def test_capabilities_excludes_fastapi_internals(client: TestClient) -> None:
    resp = client.get("/capabilities")
    paths = {e["path"] for e in resp.json()["endpoints"]}
    # FastAPI's own paths must not pollute the capabilities surface.
    assert "/openapi.json" not in paths
    assert "/docs" not in paths
    assert "/redoc" not in paths


def test_extension_router_mounts_when_present(
    character: Character, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drop a temporary character router into harness.web.characters and
    verify build_character_app discovers + mounts it."""
    import sys
    import types

    module_name = f"harness.web.characters.{character.name}"
    module = types.ModuleType(module_name)

    def build_router(_char: Character, _adapter: object) -> APIRouter:
        r = APIRouter()

        @r.get("/__ext_marker")
        def marker() -> dict[str, str]:
            return {"hit": "yes"}

        return r

    module.build_router = build_router  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)

    app = build_character_app(character, EchoAdapter())
    client = TestClient(app)
    resp = client.get("/__ext_marker")
    assert resp.status_code == 200
    assert resp.json() == {"hit": "yes"}

    # Capabilities surface should also enumerate the new endpoint.
    cap = client.get("/capabilities").json()
    assert any(e["path"] == "/__ext_marker" for e in cap["endpoints"])


def test_extension_with_no_build_router_warns_and_serves_base(
    character: Character, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed extension module (missing build_router) should warn
    rather than break the whole server."""
    import sys
    import types
    import warnings

    module_name = f"harness.web.characters.{character.name}"
    bad_module = types.ModuleType(module_name)
    # No build_router attribute.
    monkeypatch.setitem(sys.modules, module_name, bad_module)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = build_character_app(character, EchoAdapter())
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    client = TestClient(app)
    # Base endpoints still work.
    assert client.get("/healthz").status_code == 200


def test_chat_uses_persona_adapter_when_character_has_one(
    client: TestClient, character: Character
) -> None:
    """airton_c_tfr inherits voice_rewriter='persona' from the airton_c
    lineage. /chat should wrap the echo adapter in PersonaAdapter before
    completing — verified indirectly by checking the reply isn't a bare
    echo of the user message (the rewriter post-pass mutates it)."""
    assert character.voice_rewriter == "persona"
    resp = client.post("/chat", json={"text": "test input string xyz"})
    body = resp.json()
    # We don't assert the exact shape (rewriter behavior depends on
    # the character config), but we DO assert /chat reached the
    # PersonaAdapter path — the echo of the prompt plus the system
    # prompt produces a longer reply than the raw text.
    assert len(body["reply"]) > len("test input string xyz")


def _build_chat_messages(text: str) -> Iterable[ChatMessage]:
    """Convenience to mirror how the base router constructs messages —
    kept here so future tests can reuse the shape without inspecting
    base_router internals."""
    return [
        ChatMessage(role="system", content=""),
        ChatMessage(role="user", content=text),
    ]
