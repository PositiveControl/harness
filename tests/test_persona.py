from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.character import load_character
from harness.evals.voice import run_voice_eval
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.echo import EchoAdapter
from harness.persona import PersonaAdapter, build_rewriter_messages

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


@dataclass
class _RecordingAdapter:
    """Captures every call to complete() so tests can assert on the
    messages and ordering that the persona adapter issued."""

    id: str = "recording"
    context_window: int = 8192
    calls: list[list[ChatMessage]] = field(default_factory=list)
    reply: str = "ok"

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(list(messages))
        return self.reply


def test_build_rewriter_messages_shape() -> None:
    character = load_character(AIRTON)
    msgs = build_rewriter_messages(character, draft="some generic reply")

    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert "voice editor for airton" in msgs[0].content.lower()
    assert "some generic reply" in msgs[1].content
    # The rewriter instructions include key anti-patterns to avoid
    # (the string wraps across a newline, so match a shorter substring)
    assert "Cut generic-assistant filler" in msgs[0].content
    assert "Return only the rewritten reply" in msgs[0].content


def test_build_rewriter_messages_respects_exclude_example_ids() -> None:
    character = load_character(AIRTON)
    excluded = frozenset({"self_reference"})
    msgs = build_rewriter_messages(character, draft="draft", exclude_example_ids=excluded)

    system = msgs[0].content
    excluded_sample = next(s for s in character.voice_samples if s.id == "self_reference")
    assert excluded_sample.gold.strip() not in system


def test_persona_adapter_issues_two_calls() -> None:
    character = load_character(AIRTON)
    base = _RecordingAdapter(reply="pass1-output")
    adapter: ModelAdapter = PersonaAdapter(base, character)

    out = adapter.complete(
        [ChatMessage(role="user", content="hello")],
        temperature=0.5,
    )

    # The recorder returns `reply` for every call, so pass 2's output
    # equals `reply`. What we care about here is that two calls happened
    # and the second was shaped like a rewrite request.
    assert out == "pass1-output"
    assert len(base.calls) == 2
    # First call is the user's message; second is the rewriter's system + draft.
    first_call = base.calls[0]
    assert any(m.role == "user" and m.content == "hello" for m in first_call)
    second_call = base.calls[1]
    assert any(m.role == "system" and "voice editor" in m.content.lower() for m in second_call)
    assert any(m.role == "user" and "pass1-output" in m.content for m in second_call)


def test_persona_adapter_id_derived_from_base() -> None:
    character = load_character(AIRTON)
    base = _RecordingAdapter()
    adapter = PersonaAdapter(base, character)
    assert adapter.id == "persona[recording]"
    assert adapter.context_window == base.context_window


def test_persona_adapter_load_delegates_to_base() -> None:
    character = load_character(AIRTON)

    @dataclass
    class _LoadableAdapter:
        id: str = "loadable"
        context_window: int = 8192
        loaded: bool = False

        def load(self) -> None:
            self.loaded = True

        def complete(
            self,
            messages: Iterable[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            return ""

    base = _LoadableAdapter()
    adapter = PersonaAdapter(base, character)
    adapter.load()
    assert base.loaded is True


def test_voice_eval_persona_populates_draft_field() -> None:
    character = load_character(AIRTON)
    results = run_voice_eval(
        character,
        EchoAdapter(),
        sample_ids=["self_reference"],
        persona=True,
    )

    assert len(results) == 1
    r = results[0]
    assert r.draft is not None
    assert r.draft.startswith("[echo] ")  # pass 1 is echo of user prompt
    # pass 2 feeds the draft back to echo as the user message; echo returns
    # [echo] <last user content>, which includes "Draft reply to rewrite..."
    assert r.actual.startswith("[echo] ")
    assert r.actual != r.draft


def test_voice_eval_without_persona_leaves_draft_none() -> None:
    character = load_character(AIRTON)
    results = run_voice_eval(
        character,
        EchoAdapter(),
        sample_ids=["self_reference"],
        persona=False,
    )

    assert len(results) == 1
    assert results[0].draft is None


def test_build_rewriter_messages_concrete_focus() -> None:
    character = load_character(AIRTON)
    msgs = build_rewriter_messages(character, draft="some reply", focus="concrete")

    assert len(msgs) == 2
    system = msgs[0].content
    # The concrete instructions include specific substitutions
    assert "MORE CONCRETE" in system
    assert "ensure X is done" in system


def test_build_rewriter_messages_unknown_focus_raises() -> None:
    character = load_character(AIRTON)
    import pytest

    with pytest.raises(ValueError, match="Unknown rewriter focus"):
        build_rewriter_messages(character, draft="x", focus="nonsense")


def test_persona_adapter_chain_rewrites_runs_three_calls() -> None:
    character = load_character(AIRTON)
    base = _RecordingAdapter(reply="output")
    adapter = PersonaAdapter(base, character, chain_rewrites=True)

    adapter.complete([ChatMessage(role="user", content="hi")])

    # draft + style rewrite + concrete rewrite = 3
    assert len(base.calls) == 3


def test_persona_adapter_no_chain_runs_two_calls() -> None:
    character = load_character(AIRTON)
    base = _RecordingAdapter(reply="output")
    adapter = PersonaAdapter(base, character, chain_rewrites=False)

    adapter.complete([ChatMessage(role="user", content="hi")])

    assert len(base.calls) == 2


@dataclass
class _StreamingRecordingAdapter:
    """Base adapter stub for persona-stream tests. Each .stream() call
    returns a canned list of deltas; the test asserts each call's draft
    reaches the caller and that the inter-pass separator fires exactly
    once per rewrite pass."""

    id: str = "recording"
    context_window: int = 8192
    per_call_deltas: list[list[str]] = field(default_factory=list)
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return "".join(self.stream(messages, max_tokens=max_tokens, temperature=temperature))

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterable[str]:
        self.calls.append(list(messages))
        deltas = self.per_call_deltas.pop(0) if self.per_call_deltas else ["ok"]
        yield from deltas


def test_persona_adapter_stream_yields_draft_then_rewrite() -> None:
    character = load_character(AIRTON)
    base = _StreamingRecordingAdapter(
        per_call_deltas=[["draft ", "text"], ["styled ", "text"]]
    )
    adapter = PersonaAdapter(base, character, chain_rewrites=False)

    chunks = list(adapter.stream([ChatMessage(role="user", content="hi")]))
    joined = "".join(chunks)
    # draft text, separator, rewrite text — all visible in the stream
    assert "draft text" in joined
    assert "styled text" in joined
    # Separator marks the pass handoff
    assert joined.count("voice pass") == 1
    # Two base calls — one draft, one style rewrite
    assert len(base.calls) == 2


def test_persona_adapter_stream_chain_runs_three_calls() -> None:
    character = load_character(AIRTON)
    base = _StreamingRecordingAdapter(
        per_call_deltas=[["draft"], ["styled"], ["concrete"]]
    )
    adapter = PersonaAdapter(base, character, chain_rewrites=True)

    joined = "".join(adapter.stream([ChatMessage(role="user", content="hi")]))
    assert "draft" in joined
    assert "styled" in joined
    assert "concrete" in joined
    assert joined.count("voice pass") == 1
    assert joined.count("concrete pass") == 1
    assert len(base.calls) == 3


def test_persona_adapter_stream_falls_back_without_base_stream() -> None:
    """If the base adapter has no stream method, persona yields a single
    chunk from complete() so the API stays uniform."""
    character = load_character(AIRTON)
    base = _RecordingAdapter(reply="final")
    adapter = PersonaAdapter(base, character, chain_rewrites=False)

    chunks = list(adapter.stream([ChatMessage(role="user", content="hi")]))
    # Fallback emits the fully-composed reply as one chunk
    assert "".join(chunks) == "final"
