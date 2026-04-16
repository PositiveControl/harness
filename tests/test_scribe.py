from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harness.character import load_character
from harness.model.adapter import ChatMessage
from harness.scribe import parse_scribe_output, run_scribe
from harness.scribe.extractor import extract_candidates, format_window
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.store.transcript import Transcript

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


@dataclass
class _FakeEmbedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            out.append(v / norm if norm > 0 else v)
        return np.stack(out)


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued replies in order, one per complete() call."""

    replies: list[str]
    id: str = "scripted"
    context_window: int = 8192
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(list(messages))
        return self.replies.pop(0) if self.replies else ""


# ---------- parse_scribe_output ----------


def test_parse_scribe_valid_json() -> None:
    raw = """
    {
      "episodic": [
        {"title": "Debug session", "body": "We chased a bug.", "principle": null, "tags": ["debug"]}
      ],
      "semantic": [
        {"subject": "mark", "predicate": "prefers", "object": "raw sqlite", "confidence": 0.9}
      ]
    }
    """
    result = parse_scribe_output(raw)
    assert result.parse_error is None
    assert len(result.episodic) == 1
    assert result.episodic[0].title == "Debug session"
    assert result.episodic[0].tags == ("debug",)
    assert len(result.semantic) == 1
    assert result.semantic[0].confidence == 0.9


def test_parse_scribe_strips_markdown_fences() -> None:
    raw = '```json\n{"episodic": [], "semantic": []}\n```'
    result = parse_scribe_output(raw)
    assert result.parse_error is None
    assert result.episodic == ()
    assert result.semantic == ()


def test_parse_scribe_drops_malformed_items_not_whole_result() -> None:
    raw = """
    {
      "episodic": [
        {"title": "ok", "body": "also ok"},
        {"title": "", "body": "missing title"},
        {"body": "no title at all"}
      ],
      "semantic": [
        {"subject": "a", "predicate": "b", "object": "c", "confidence": 0.8},
        {"subject": "", "predicate": "empty", "object": "subject", "confidence": 0.9}
      ]
    }
    """
    result = parse_scribe_output(raw)
    assert len(result.episodic) == 1  # only the one valid item
    assert len(result.semantic) == 1


def test_parse_scribe_records_parse_error_on_garbage() -> None:
    result = parse_scribe_output("not even a little bit JSON")
    assert result.parse_error is not None
    assert result.episodic == ()
    assert result.semantic == ()


def test_parse_scribe_confidence_clamped_to_unit_range() -> None:
    raw = (
        '{"episodic": [], "semantic": ['
        '{"subject": "a", "predicate": "b", "object": "c", "confidence": 2.5}]}'
    )
    result = parse_scribe_output(raw)
    assert result.semantic[0].confidence == 1.0


# ---------- format_window ----------


def test_format_window_renders_speaker_and_content(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "h.sqlite")
    try:
        m1 = t.append(session="s", channel="c", speaker="mark", role="user", content="hi")
        m2 = t.append(
            session="s", channel="c", speaker="airton", role="assistant", content="morning"
        )
        window = format_window([m1, m2])
        assert "mark: hi" in window
        assert "airton: morning" in window
    finally:
        t.close()


# ---------- extract_candidates ----------


def test_extract_candidates_with_empty_window_returns_empty() -> None:
    character = load_character(AIRTON)
    adapter = _ScriptedAdapter(replies=[])
    result = extract_candidates(adapter, character, turns=[])
    assert result.episodic == ()
    assert result.semantic == ()


def test_extract_candidates_parses_model_output(tmp_path: Path) -> None:
    character = load_character(AIRTON)
    t = Transcript(tmp_path / "h.sqlite")
    try:
        m = t.append(session="s", channel="c", speaker="mark", role="user", content="morning")
        reply = (
            '{"episodic": [], "semantic": ['
            '{"subject": "mark", "predicate": "said", "object": "morning", "confidence": 1.0}'
            "]}"
        )
        adapter = _ScriptedAdapter(replies=[reply])
        result = extract_candidates(adapter, character, turns=[m])
        assert len(result.semantic) == 1
        assert result.semantic[0].subject == "mark"
    finally:
        t.close()


# ---------- run_scribe end-to-end ----------


def test_run_scribe_persists_candidates_and_advances_watermark(tmp_path: Path) -> None:
    character = load_character(AIRTON)
    t = Transcript(tmp_path / "h.sqlite")
    ep = EpisodicStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    sem = SemanticStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    try:
        t.append(session="s1", channel="c", speaker="mark", role="user", content="hi")
        t.append(session="s1", channel="c", speaker="airton", role="assistant", content="morning")

        reply = (
            '{"episodic": ['
            '{"title": "Morning greeting", "body": "Mark said hi.", '
            '"principle": null, "tags": ["greeting"]}'
            '], "semantic": ['
            '{"subject": "mark", "predicate": "said", "object": "hi", "confidence": 1.0}'
            "]}"
        )
        scripted = _ScriptedAdapter(replies=[reply])
        summary = run_scribe(scripted, character, t, ep, sem, session_id="s1", window_size=20)
        assert summary.turns_processed == 2
        assert summary.episodic_written == 1
        assert summary.semantic_written == 1

        # Second run is a no-op — nothing new past the watermark
        second = run_scribe(scripted, character, t, ep, sem, session_id="s1")
        assert second.turns_processed == 0
        assert second.episodic_written == 0
    finally:
        t.close()
        ep.close()
        sem.close()


def test_run_scribe_records_parse_errors_and_continues(tmp_path: Path) -> None:
    character = load_character(AIRTON)
    t = Transcript(tmp_path / "h.sqlite")
    ep = EpisodicStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    sem = SemanticStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    try:
        t.append(session="s", channel="c", speaker="mark", role="user", content="one")
        scripted = _ScriptedAdapter(replies=["not JSON"])
        summary = run_scribe(scripted, character, t, ep, sem, session_id="s")
        assert summary.parse_errors  # at least one
        assert summary.episodic_written == 0
    finally:
        t.close()
        ep.close()
        sem.close()
