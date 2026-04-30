"""Tests for the phraseology lint pipeline (harness-q35t).

Covers:
- Pre-model cite-or-silent gate (empty retrieval ⇒ out_of_scope, no
  model call).
- Happy-path verdicts (ok / wrong / incomplete).
- Post-model cite-grounding gate (model picks a section outside the
  top-K candidate set ⇒ downgrade to out_of_scope).
- Unparseable JSON ⇒ out_of_scope with diagnostic mismatch.
- JSON normalization (unicode minus, leading §).
- Tool wrapper produces a ToolResult with JSON output + grounded set.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from harness.character import load_character
from harness.model.adapter import ChatMessage
from harness.store.episodic import EpisodicStore
from harness.tools.phraseology_lint import (
    PhraseologyLintTool,
    PhraseologyVerdict,
    _parse_verdict_json,
    lint_utterance,
)

# Reuse airton_c1's FAA grammar — that's the character whose corpus
# the seeded chunks model.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GRAMMAR = load_character(_REPO_ROOT / "character" / "airton_c1").citation_grammar
assert _GRAMMAR is not None


@dataclass
class _LexicalEmbedder:
    """Token-bag deterministic embedder (matches test_cite_grounding)."""

    dimension: int = 16
    id: str = "lex-test"

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            tokens = text.lower().split()
            v = np.zeros(self.dimension, dtype=np.float32)
            for tok in tokens:
                bucket = sum(ord(c) for c in tok) % self.dimension
                v[bucket] += 1.0
            norm = float(np.linalg.norm(v))
            out.append(v / norm if norm > 0 else v)
        return np.stack(out)


@dataclass
class _ScriptedAdapter:
    """Returns canned strings for `complete()`. Records every prompt
    so cite-or-silent tests can assert the model was NOT called."""

    replies: list[str] = field(default_factory=list)
    calls: list[list[ChatMessage]] = field(default_factory=list)
    id: str = "scripted"
    context_window: int = 8192

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("scripted adapter ran out of replies")
        return self.replies.pop(0)


def _seed_phraseology_corpus(store: EpisodicStore) -> None:
    """Seed three JO 7110.65 phraseology sections shaped like the
    airton_c1 corpus chunks."""
    store.ingest(
        external_id="seed-3-9-10",
        title="TAKEOFF CLEARANCE",
        body=(
            "When issuing a clearance for takeoff, first state the runway "
            "number followed by the takeoff clearance. PHRASEOLOGY: "
            "RUNWAY (number), CLEARED FOR TAKEOFF."
        ),
        principle="JO_7110.65 §3-9-10 (Departure Procedures and Separation — TAKEOFF CLEARANCE)",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="seed-3-10-5",
        title="LANDING CLEARANCE",
        body=(
            "When issuing a clearance to land, first state the runway "
            "number followed by the landing clearance. PHRASEOLOGY: "
            "RUNWAY (number) CLEARED TO LAND."
        ),
        principle="JO_7110.65 §3-10-5 (Arrival Procedures and Separation — LANDING CLEARANCE)",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="seed-10-2-6",
        title="HIJACKED AIRCRAFT",
        body=(
            "When a pilot notifies ATC of a hijacking situation, assign "
            "code 7500. PHRASEOLOGY: (Identification) SQUAWK SEVEN FIVE ZERO ZERO."
        ),
        principle="JO_7110.65 §10-2-6 (Emergency Assistance — HIJACKED AIRCRAFT)",
        tier="seed",
        source="yaml",
    )


# ---------- _parse_verdict_json ----------


def test_parse_strips_section_prefix_and_normalizes_minus() -> None:
    raw = json.dumps(
        {
            "verdict": "ok",
            "expected_section": "§3−9−10",  # noqa: RUF001
            "expected_phraseology": "RUNWAY (number), CLEARED FOR TAKEOFF.",
            "mismatch": None,
            "citation_quote": None,
        }
    )
    verdict = _parse_verdict_json(raw)
    assert verdict is not None
    assert verdict.expected_section == "3-9-10"
    assert verdict.verdict == "ok"


def test_parse_returns_none_on_invalid_verdict() -> None:
    raw = json.dumps({"verdict": "fine", "expected_section": "3-9-10"})
    assert _parse_verdict_json(raw) is None


def test_parse_returns_none_on_garbage() -> None:
    assert _parse_verdict_json("not JSON at all") is None
    assert _parse_verdict_json("") is None


def test_parse_extracts_embedded_json() -> None:
    """Tolerant decode — pulls JSON out of stray prose."""
    raw = (
        "Sure, here you go: "
        '{"verdict": "ok", "expected_section": "3-9-10", '
        '"expected_phraseology": "X", "mismatch": null, "citation_quote": null} '
        "Hope that helps."
    )
    verdict = _parse_verdict_json(raw)
    assert verdict is not None
    assert verdict.verdict == "ok"
    assert verdict.expected_section == "3-9-10"


# ---------- lint_utterance ----------


def test_empty_store_short_circuits_to_out_of_scope(tmp_path: Path) -> None:
    """Pre-model gate: no candidate sections ⇒ refuse without calling
    the model."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    adapter = _ScriptedAdapter(replies=[])  # no replies — would fail if called
    try:
        verdict = lint_utterance(
            "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
        )
    finally:
        store.close()
    assert verdict.verdict == "out_of_scope"
    assert verdict.expected_section is None
    assert verdict.mismatch is None
    assert adapter.calls == [], "model was invoked despite empty retrieval"


def test_happy_path_ok_verdict(tmp_path: Path) -> None:
    """Happy path: model returns a verdict for a section that's in the
    candidate set ⇒ verdict survives the cite-grounding gate."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    reply = json.dumps(
        {
            "verdict": "ok",
            "expected_section": "3-9-10",
            "expected_phraseology": "RUNWAY (number), CLEARED FOR TAKEOFF.",
            "mismatch": None,
            "citation_quote": "RUNWAY (number), CLEARED FOR TAKEOFF.",
        }
    )
    adapter = _ScriptedAdapter(replies=[reply])
    try:
        verdict = lint_utterance(
            "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
        )
    finally:
        store.close()
    assert verdict.verdict == "ok"
    assert verdict.expected_section == "3-9-10"
    assert verdict.expected_phraseology == "RUNWAY (number), CLEARED FOR TAKEOFF."
    assert len(adapter.calls) == 1


def test_wrong_verdict_with_grounded_section(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    reply = json.dumps(
        {
            "verdict": "wrong",
            "expected_section": "3-9-10",
            "expected_phraseology": "RUNWAY (number), CLEARED FOR TAKEOFF.",
            "mismatch": "wrong preposition: 'TO TAKEOFF' should be 'FOR TAKEOFF'",
            "citation_quote": "RUNWAY (number), CLEARED FOR TAKEOFF.",
        }
    )
    adapter = _ScriptedAdapter(replies=[reply])
    try:
        verdict = lint_utterance(
            "RUNWAY TWO SEVEN, CLEARED TO TAKEOFF.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
        )
    finally:
        store.close()
    assert verdict.verdict == "wrong"
    assert verdict.expected_section == "3-9-10"
    assert verdict.mismatch is not None
    assert "TO TAKEOFF" in verdict.mismatch


def test_cite_grounding_gate_downgrades_ungrounded(tmp_path: Path) -> None:
    """Post-model gate: model picks §99-99-99 (not in any candidate
    chunk's principle). Verdict downgraded to out_of_scope with a
    diagnostic mismatch."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    reply = json.dumps(
        {
            "verdict": "ok",
            "expected_section": "99-99-99",
            "expected_phraseology": "FAKE TEMPLATE",
            "mismatch": None,
            "citation_quote": "fabricated",
        }
    )
    adapter = _ScriptedAdapter(replies=[reply])
    try:
        verdict = lint_utterance(
            "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
        )
    finally:
        store.close()
    assert verdict.verdict == "out_of_scope"
    assert verdict.expected_section is None
    assert verdict.mismatch is not None
    assert "99-99-99" in verdict.mismatch


def test_unparseable_model_output_returns_oos(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    adapter = _ScriptedAdapter(replies=["not even a hint of JSON here"])
    try:
        verdict = lint_utterance(
            "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
        )
    finally:
        store.close()
    assert verdict.verdict == "out_of_scope"
    assert verdict.expected_section is None
    assert verdict.mismatch == "model output unparseable"


def test_oos_verdict_nulls_citation_fields(tmp_path: Path) -> None:
    """Model returned out_of_scope but parroted a section. The pipeline
    must null the citation fields so eval comparison is unambiguous."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    reply = json.dumps(
        {
            "verdict": "out_of_scope",
            "expected_section": "3-9-10",
            "expected_phraseology": "RUNWAY (number), CLEARED FOR TAKEOFF.",
            "mismatch": "pilot transmission",
            "citation_quote": "stray",
        }
    )
    adapter = _ScriptedAdapter(replies=[reply])
    try:
        verdict = lint_utterance(
            "REQUEST IFR CLEARANCE TO BOSTON.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
        )
    finally:
        store.close()
    assert verdict.verdict == "out_of_scope"
    assert verdict.expected_section is None
    assert verdict.expected_phraseology is None
    assert verdict.citation_quote is None
    assert verdict.mismatch == "pilot transmission"


def test_scenario_hint_gets_into_prompt(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    reply = json.dumps(
        {
            "verdict": "ok",
            "expected_section": "3-9-10",
            "expected_phraseology": "RUNWAY (number), CLEARED FOR TAKEOFF.",
            "mismatch": None,
            "citation_quote": None,
        }
    )
    adapter = _ScriptedAdapter(replies=[reply])
    try:
        lint_utterance(
            "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF.",
            adapter=adapter,
            episodic_store=store,
            grammar=_GRAMMAR,
            scenario_hint="departure",
        )
    finally:
        store.close()
    assert len(adapter.calls) == 1
    user_msg = next(m for m in adapter.calls[0] if m.role == "user")
    assert "SCENARIO HINT: departure" in user_msg.content


# ---------- PhraseologyLintTool ----------


def test_tool_wraps_lint_pipeline(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    reply = json.dumps(
        {
            "verdict": "wrong",
            "expected_section": "3-10-5",
            "expected_phraseology": "RUNWAY (number) CLEARED TO LAND.",
            "mismatch": "wrong preposition",
            "citation_quote": "RUNWAY (number) CLEARED TO LAND.",
        }
    )
    adapter = _ScriptedAdapter(replies=[reply])
    tool = PhraseologyLintTool(adapter=adapter, store=store, grammar=_GRAMMAR)
    try:
        result = tool.call(utterance="RUNWAY ONE EIGHT, CLEARED FOR LAND.")
    finally:
        store.close()
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["verdict"] == "wrong"
    assert payload["expected_section"] == "3-10-5"
    assert "3-10-5" in result.citations_grounded
    assert len(result.hits) == 1
    assert result.hits[0].title == "§3-10-5"


def test_tool_spec_shape() -> None:
    """ToolSpec required fields and parameter schema are well-formed."""
    # Construct the tool with throwaway deps; we only inspect the spec.
    adapter = _ScriptedAdapter(replies=[])

    class _StubStore:
        pass

    # Spec inspection only — no retrieval; type-check ignored on the stub.
    tool = PhraseologyLintTool(
        adapter=adapter,
        store=_StubStore(),  # type: ignore[arg-type]
    )
    spec = tool.spec
    assert spec.name == "phraseology_lint"
    assert spec.tier == "read"
    assert "utterance" in spec.parameters["properties"]
    assert spec.parameters["required"] == ["utterance"]
    # scenario_hint is optional
    assert "scenario_hint" in spec.parameters["properties"]


# ---------- Verdict dataclass round-trip ----------


def test_verdict_dataclass_immutable() -> None:
    v = PhraseologyVerdict(
        verdict="ok",
        expected_section="3-9-10",
        expected_phraseology="X",
        mismatch=None,
        citation_quote=None,
    )
    # frozen dataclass — any mutation raises FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.verdict = "wrong"  # type: ignore[misc]
