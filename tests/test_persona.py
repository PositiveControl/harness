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
