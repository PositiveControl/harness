"""Tests for src/harness/persona/caveman_rewriter.py — harness-inj.3.

Covers:
- intensity validation on construction and per-call
- register_map loading (valid, invalid, missing)
- pick_intensity resolution: surface hit, miss, default fallback,
  ctor fallback
- complete() rewrites at the configured intensity (substance pass +
  rewrite pass observed as two base-adapter calls)
- auto-clarity bypass: `normal` intensity returns the draft unchanged
  (one base-adapter call total)
- stream() emits draft then rewrite, with a normal-intensity surface
  skipping the rewrite stream
- complete_with_tools passes through when rewrite_on_tools=False;
  rewrites the string result when True
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from harness.model.adapter import ChatMessage
from harness.persona.caveman_rewriter import (
    ALLOWED_INTENSITIES,
    CavemanRewriter,
    build_caveman_messages,
    load_register_map,
)


class FakeAdapter:
    """Records the message lists passed to `complete` / `stream` /
    `complete_with_tools` and returns canned responses in order. Lets
    tests assert which pass each call belongs to (substance vs
    rewrite) by inspecting the system prompt."""

    id = "fake"
    context_window = 8192

    def __init__(self, complete_outputs: list[str] | None = None) -> None:
        self.complete_calls: list[list[ChatMessage]] = []
        self.stream_calls: list[list[ChatMessage]] = []
        self.tool_calls: list[tuple[list[ChatMessage], object]] = []
        self._complete_outputs = list(complete_outputs or [])
        self._tool_result: object | None = None

    def queue_complete(self, *outputs: str) -> None:
        self._complete_outputs.extend(outputs)

    def set_tool_result(self, result: object) -> None:
        self._tool_result = result

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        materialized = list(messages)
        self.complete_calls.append(materialized)
        if self._complete_outputs:
            return self._complete_outputs.pop(0)
        return "DEFAULT"

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        materialized = list(messages)
        self.stream_calls.append(materialized)
        chunk = self._complete_outputs.pop(0) if self._complete_outputs else "DEFAULT"
        # Emit two deltas so tests can tell streaming from one-shot.
        mid = max(1, len(chunk) // 2)
        yield chunk[:mid]
        yield chunk[mid:]

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: object = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> object:
        self.tool_calls.append((list(messages), tools))
        return self._tool_result


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def test_allowed_intensities_includes_normal() -> None:
    # `normal` is the auto-clarity bypass sentinel; it must be valid so
    # per-surface maps can request "no rewrite here."
    assert "normal" in ALLOWED_INTENSITIES


def test_build_caveman_messages_validates_intensity() -> None:
    with pytest.raises(ValueError, match="intensity"):
        build_caveman_messages("draft", intensity="maximum")


def test_build_caveman_messages_shape() -> None:
    msgs = build_caveman_messages("some draft text", intensity="lite")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert "caveman-lite" in msgs[0].content.lower()
    assert msgs[1].role == "user"
    assert "some draft text" in msgs[1].content


def test_load_register_map_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_register_map(tmp_path / "nope.yaml") == {}


def test_load_register_map_valid(tmp_path: Path) -> None:
    path = tmp_path / "reg.yaml"
    path.write_text("default: lite\nsurfaces:\n  plan_shall: lite\n  short_ack: full\n")
    loaded = load_register_map(path)
    assert loaded["default"] == "lite"
    surfaces = loaded.get("surfaces")
    assert isinstance(surfaces, dict)
    assert surfaces["plan_shall"] == "lite"
    assert surfaces["short_ack"] == "full"


def test_load_register_map_rejects_invalid_default(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("default: loudest\nsurfaces: {}\n")
    with pytest.raises(ValueError, match="default"):
        load_register_map(path)


def test_load_register_map_rejects_invalid_surface_value(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("default: lite\nsurfaces:\n  plan_shall: gigagradient\n")
    with pytest.raises(ValueError, match="plan_shall"):
        load_register_map(path)


def test_rewriter_rejects_invalid_intensity() -> None:
    base = FakeAdapter()
    with pytest.raises(ValueError, match="intensity"):
        CavemanRewriter(base, intensity="beast")


def test_pick_intensity_from_surface_map() -> None:
    base = FakeAdapter()
    reg = {"default": "full", "surfaces": {"plan_shall": "lite", "short_ack": "full"}}
    rewriter = CavemanRewriter(base, intensity="ultra", register_map=reg)
    assert rewriter.pick_intensity("plan_shall") == "lite"
    assert rewriter.pick_intensity("short_ack") == "full"


def test_pick_intensity_unknown_surface_falls_back_to_default() -> None:
    base = FakeAdapter()
    reg = {"default": "full", "surfaces": {"plan_shall": "lite"}}
    rewriter = CavemanRewriter(base, intensity="ultra", register_map=reg)
    # Surface not in map → use register_map default, not ctor intensity.
    assert rewriter.pick_intensity("not_defined") == "full"


def test_pick_intensity_no_default_falls_back_to_ctor() -> None:
    base = FakeAdapter()
    reg: dict[str, object] = {"surfaces": {"plan_shall": "lite"}}
    rewriter = CavemanRewriter(base, intensity="ultra", register_map=reg)
    # No default in map → ctor intensity is the terminal fallback.
    assert rewriter.pick_intensity("nope") == "ultra"
    assert rewriter.pick_intensity(None) == "ultra"


def test_complete_runs_two_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    base = FakeAdapter(complete_outputs=["DRAFT_SUBSTANCE", "REWRITTEN_LITE"])
    rewriter = CavemanRewriter(base, intensity="lite")

    result = rewriter.complete([_msg("user", "morning")])

    assert result == "REWRITTEN_LITE"
    assert len(base.complete_calls) == 2
    # First call: the caller's messages (substance pass).
    assert base.complete_calls[0][0].content == "morning"
    # Second call: the caveman system prompt + the draft.
    sys_prompt = base.complete_calls[1][0].content
    assert "caveman-lite" in sys_prompt.lower()
    user_prompt = base.complete_calls[1][1].content
    assert "DRAFT_SUBSTANCE" in user_prompt


def test_complete_normal_surface_bypasses_rewrite() -> None:
    base = FakeAdapter(complete_outputs=["DESTRUCTIVE_CONFIRM_PROSE"])
    reg = {"default": "lite", "surfaces": {"destructive_confirm": "normal"}}
    rewriter = CavemanRewriter(base, intensity="lite", register_map=reg)

    result = rewriter.complete(
        [_msg("user", "delete all projects")],
        surface="destructive_confirm",
    )

    assert result == "DESTRUCTIVE_CONFIRM_PROSE"
    # One call only — substance pass — no rewrite dispatched.
    assert len(base.complete_calls) == 1


def test_complete_respects_per_surface_intensity() -> None:
    base = FakeAdapter(complete_outputs=["DRAFT", "REWRITE_FULL"])
    reg = {"default": "lite", "surfaces": {"short_ack": "full"}}
    rewriter = CavemanRewriter(base, intensity="lite", register_map=reg)

    rewriter.complete([_msg("user", "got it")], surface="short_ack")

    assert len(base.complete_calls) == 2
    system_prompt = base.complete_calls[1][0].content.lower()
    assert "caveman-full" in system_prompt


def test_stream_emits_draft_then_rewrite() -> None:
    base = FakeAdapter(complete_outputs=["DRAFT_A", "REWRITE_B"])
    rewriter = CavemanRewriter(base, intensity="lite")

    chunks = list(rewriter.stream([_msg("user", "morning")]))

    joined = "".join(chunks)
    assert "DRAFT_A" in joined
    assert "REWRITE_B" in joined
    assert "caveman pass" in joined  # handoff marker
    assert len(base.stream_calls) == 2


def test_stream_normal_surface_skips_rewrite_stream() -> None:
    base = FakeAdapter(complete_outputs=["CONFIRM_PROSE"])
    reg = {"default": "lite", "surfaces": {"destructive_confirm": "normal"}}
    rewriter = CavemanRewriter(base, intensity="lite", register_map=reg)

    chunks = list(rewriter.stream([_msg("user", "drop it")], surface="destructive_confirm"))

    joined = "".join(chunks)
    assert "CONFIRM_PROSE" in joined
    assert "caveman pass" not in joined
    assert len(base.stream_calls) == 1


def test_complete_with_tools_passes_through_by_default() -> None:
    base = FakeAdapter()
    base.set_tool_result("tool-produced-string")
    rewriter = CavemanRewriter(base, intensity="lite")

    result = rewriter.complete_with_tools(
        [_msg("user", "list files")],
        tools=[{"name": "list_dir"}],
    )

    assert result == "tool-produced-string"
    # No rewrite pass — base.complete was never called.
    assert base.complete_calls == []


def test_complete_with_tools_forwards_tools_as_keyword_to_base() -> None:
    """Regression: CavemanRewriter used to pass `tools` positionally
    to the base adapter. Every real base (MLX / Ollama / Echo)
    declares tools keyword-only, so every tool turn raised TypeError.
    This test pins the contract: base is invoked with tools=<list>,
    never positionally."""

    class KeywordOnlyToolsBase:
        id = "kwonly"
        context_window = 8192
        last_tools: object = None

        def complete(
            self,
            messages: Iterable[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            return ""

        def complete_with_tools(
            self,
            messages: Iterable[ChatMessage],
            *,
            tools: object = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            KeywordOnlyToolsBase.last_tools = tools
            return "ok"

    base = KeywordOnlyToolsBase()
    rewriter = CavemanRewriter(base, intensity="lite")
    payload = [{"name": "plan"}]

    rewriter.complete_with_tools([_msg("user", "plan")], tools=payload)

    assert KeywordOnlyToolsBase.last_tools == payload


def test_complete_with_tools_rewrites_when_opted_in() -> None:
    base = FakeAdapter(complete_outputs=["REWRITTEN_RESULT"])
    base.set_tool_result("raw-tool-output")
    rewriter = CavemanRewriter(base, intensity="lite", rewrite_on_tools=True)

    result = rewriter.complete_with_tools(
        [_msg("user", "list files")],
        tools=[{"name": "list_dir"}],
    )

    assert result == "REWRITTEN_RESULT"
    # One rewrite pass on the tool-produced string.
    assert len(base.complete_calls) == 1
    user_prompt = base.complete_calls[0][1].content
    assert "raw-tool-output" in user_prompt


def test_load_delegates_to_base_when_available() -> None:
    class LoadableAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.loaded = False

        def load(self) -> None:
            self.loaded = True

    base = LoadableAdapter()
    rewriter = CavemanRewriter(base, intensity="lite")
    rewriter.load()
    assert base.loaded is True


def test_register_map_on_disk_matches_ship_file(tmp_path: Path) -> None:
    """Exercise the loader against ab's own register_map.yaml so
    regressions in the shipped file (typo, disallowed intensity,
    wrong shape) fail here first."""
    repo_root = Path(__file__).resolve().parents[1]
    reg_path = repo_root / "character" / "airton_b" / "register_map.yaml"
    loaded = load_register_map(reg_path)
    assert loaded.get("default") in ALLOWED_INTENSITIES
    surfaces = loaded.get("surfaces")
    assert isinstance(surfaces, dict)
    # Sanity: the spec's auto-clarity surfaces resolve to `normal`.
    for surface in ("destructive_confirm", "error_retraction", "clarifying_response"):
        assert surfaces.get(surface) == "normal"
