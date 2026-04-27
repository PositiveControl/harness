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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from harness.model.adapter import ChatMessage
from harness.store.episodic import EpisodicRecord, EpisodicStore
from harness.tools.phraseology_lint import (
    _INJECTED_HIT_SCORE,
    PhraseologyLintTool,
    PhraseologyVerdict,
    _inject_missing_anchored_sections,
    _matched_anchor_sections,
    _parse_verdict_json,
    _verb_anchor_rerank,
    default_verb_anchors_path,
    lint_utterance,
    load_verb_anchors,
)


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
    tool = PhraseologyLintTool(adapter=adapter, store=store)
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


# ---------- verb-anchor re-rank (harness-ptya) ----------


def test_load_verb_anchors_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing yaml ⇒ empty mapping, not an error. Lint pipeline still
    works without the map (just no re-rank)."""
    assert load_verb_anchors(tmp_path / "nope.yaml") == {}


def test_load_verb_anchors_uppercases_and_strips(tmp_path: Path) -> None:
    """Anchors normalized to uppercase + trimmed. Section keys lose any
    leading § or whitespace so they match the rec.principle anchors."""
    yaml_text = (
        '"10-2-6":\n  - squawk\n  - SQUAWK SEVEN FIVE ZERO ZERO\n"§ 5-7-2 ":\n  - reduce speed\n'
    )
    path = tmp_path / "verb_anchors.yaml"
    path.write_text(yaml_text)
    out = load_verb_anchors(path)
    assert out["10-2-6"] == ("SQUAWK", "SQUAWK SEVEN FIVE ZERO ZERO")
    assert out["5-7-2"] == ("REDUCE SPEED",)


def test_load_verb_anchors_skips_malformed_entries(tmp_path: Path) -> None:
    """Non-string sections, non-list verbs, empty entries — drop, don't
    crash. The map is best-effort; bad entries shouldn't poison good ones."""
    yaml_text = (
        '"10-2-6":\n'
        "  - SQUAWK\n"
        "  - 1234\n"  # not a string — skipped
        "  - ''\n"  # empty — skipped
        "5: not_a_list\n"  # non-string section, non-list verbs
        '"":\n'  # empty section
        "  - SOMETHING\n"
    )
    path = tmp_path / "verb_anchors.yaml"
    path.write_text(yaml_text)
    out = load_verb_anchors(path)
    assert out == {"10-2-6": ("SQUAWK",)}


def test_load_verb_anchors_malformed_yaml_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(": :: not yaml :: :")
    assert load_verb_anchors(path) == {}


def test_default_verb_anchors_path_convention(tmp_path: Path) -> None:
    """Mirrors `default_synonyms_path` — `<character>/corpus/verb_anchors.yaml`."""
    p = default_verb_anchors_path(tmp_path / "character" / "airton_c1")
    assert p == tmp_path / "character" / "airton_c1" / "corpus" / "verb_anchors.yaml"


def test_matched_anchor_sections_word_boundary() -> None:
    """SQUAWK in the utterance fires §10-2-6; bare 'CONTACTING' (no
    word-boundary on CONTACT) does NOT fire §2-1-17."""
    anchors = {
        "10-2-6": ("SQUAWK",),
        "2-1-17": ("CONTACT",),
    }
    assert _matched_anchor_sections(
        "AMERICAN ONE TWENTY THREE, SQUAWK SEVEN FIVE ZERO ZERO.",
        anchors,
    ) == {"10-2-6"}
    # 'CONTACT' is a real verb; check the boundary check rejects substrings
    assert _matched_anchor_sections("WE ARE TRANSCATHETER-BOUND.", anchors) == set()
    assert _matched_anchor_sections("CONTACT DEPARTURE ONE TWO THREE.", anchors) == {"2-1-17"}


def test_matched_anchor_sections_multi_word_phrase() -> None:
    """Multi-word anchors match as literal phrases (re.escape handles
    spaces). 'LINE UP' alone doesn't fire 'LINE UP AND WAIT'."""
    anchors = {"3-9-4": ("LINE UP AND WAIT",)}
    assert _matched_anchor_sections("RUNWAY 27, LINE UP AND WAIT.", anchors) == {"3-9-4"}
    assert _matched_anchor_sections("RUNWAY 27, LINE UP.", anchors) == set()


def test_matched_anchor_sections_empty_map_returns_empty() -> None:
    """No anchors loaded ⇒ never fire (saves a regex loop)."""
    assert _matched_anchor_sections("anything", {}) == set()


def _rec(principle: str, body: str = "", title: str = "") -> EpisodicRecord:
    """Build a real EpisodicRecord for rerank tests. Only `principle` is
    semantically load-bearing for the rerank — everything else is filler."""
    return EpisodicRecord(
        id=0,
        external_id=None,
        title=title,
        body=body,
        principle=principle,
        tags=(),
        tier="seed",
        source="test",
        session_id=None,
        user_id=None,
        created_at=datetime.now(UTC),
    )


def test_verb_anchor_rerank_promotes_matched_section() -> None:
    """Hits whose §-anchor is in `matched_sections` move to the front,
    preserving relative retrieval order. Unmatched stay behind in their
    original order."""
    rec_a = _rec("JO_7110.65 §2-4-17 (Numbers Usage — DIGIT GROUP)")
    rec_b = _rec("JO_7110.65 §10-2-6 (Emergency — HIJACKED AIRCRAFT)")
    rec_c = _rec("JO_7110.65 §3-9-10 (Departure — TAKEOFF CLEARANCE)")
    hits = [(rec_a, 0.9), (rec_b, 0.6), (rec_c, 0.4)]
    out = _verb_anchor_rerank(hits, {"10-2-6"})
    assert out == [(rec_b, 0.6), (rec_a, 0.9), (rec_c, 0.4)]


def test_verb_anchor_rerank_no_match_passes_through() -> None:
    """Empty `matched_sections` ⇒ identity. The pipeline only pays the
    cost when a verb actually fired."""
    rec_a = _rec("JO_7110.65 §2-4-17 (Numbers Usage)")
    rec_b = _rec("JO_7110.65 §3-9-10 (Departure)")
    hits = [(rec_a, 0.8), (rec_b, 0.5)]
    assert _verb_anchor_rerank(hits, set()) == hits


def test_verb_anchor_rerank_no_matched_hits_passes_through() -> None:
    """Verb fired but no hit in the slate carries the anchored §. We
    don't fabricate a virtual hit — the cite-grounding gate would just
    re-reject it; better to be a no-op."""
    rec_a = _rec("JO_7110.65 §2-4-17 (Numbers Usage)")
    rec_b = _rec("JO_7110.65 §3-9-10 (Departure)")
    hits = [(rec_a, 0.8), (rec_b, 0.5)]
    out = _verb_anchor_rerank(hits, {"10-2-6"})
    # Stable: matched-§ partition is empty, so unmatched come back in order.
    assert out == hits


def test_verb_anchor_rerank_multiple_matched_sections_preserves_order() -> None:
    """When two anchored §s both have hits, both go to the front in
    their retrieval order — neither anchor 'wins' over the other."""
    rec_a = _rec("JO_7110.65 §2-4-17 (Numbers Usage)")
    rec_b = _rec("JO_7110.65 §10-2-6 (Hijack)")
    rec_c = _rec("JO_7110.65 §5-7-2 (Speed Adjustment)")
    rec_d = _rec("JO_7110.65 §3-9-10 (Departure)")
    hits = [(rec_a, 0.9), (rec_b, 0.7), (rec_c, 0.5), (rec_d, 0.3)]
    out = _verb_anchor_rerank(hits, {"10-2-6", "5-7-2"})
    # rec_b (10-2-6) and rec_c (5-7-2) both promoted, in retrieval order
    assert out == [(rec_b, 0.7), (rec_c, 0.5), (rec_a, 0.9), (rec_d, 0.3)]


def test_inject_missing_anchored_sections_text_mode_fetch(tmp_path: Path) -> None:
    """Verb anchor fires for a § that hybrid retrieval didn't return.
    Injection text-mode-probes the store on the section number and
    prepends one matching row at the sentinel score."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    try:
        # Pretend hybrid retrieval surfaced only one filler row, NOT
        # §10-2-6 — same shape as the real cluster #3 failure where the
        # dense embedder dominates with §2-4-17 instead.
        filler = list(store.search("anything", k=1, mode="hybrid"))
        out = _inject_missing_anchored_sections(
            filler,
            matched_sections={"10-2-6"},
            episodic_store=store,
            user_id=None,
        )
    finally:
        store.close()
    # Original filler row is still there; §10-2-6 injected at the back
    # (the rerank phase will then move it to the front).
    sections = [_anchor_for(rec) for rec, _ in out]
    assert "10-2-6" in sections, f"expected §10-2-6 in injected slate, got {sections}"
    # Injected row carries the sentinel score so it doesn't disrupt
    # ordering of the real hits.
    injected = [(rec, score) for rec, score in out if _anchor_for(rec) == "10-2-6"]
    assert all(score == _INJECTED_HIT_SCORE for _, score in injected)


def test_inject_missing_anchored_sections_no_op_when_present(tmp_path: Path) -> None:
    """If the matched § already appears in the slate, injection skips
    it (no duplicate row)."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    try:
        # Real retrieval against the seeded corpus DOES include §10-2-6
        # for a SQUAWK query.
        hits = list(store.search("SQUAWK SEVEN FIVE ZERO ZERO", k=3, mode="hybrid"))
        before_count = sum(1 for rec, _ in hits if _anchor_for(rec) == "10-2-6")
        assert before_count >= 1, "test setup expected §10-2-6 in raw retrieval"
        out = _inject_missing_anchored_sections(
            hits,
            matched_sections={"10-2-6"},
            episodic_store=store,
            user_id=None,
        )
    finally:
        store.close()
    after_count = sum(1 for rec, _ in out if _anchor_for(rec) == "10-2-6")
    assert after_count == before_count, "injection added a duplicate row"


def test_inject_missing_anchored_sections_silently_drops_absent_section(
    tmp_path: Path,
) -> None:
    """Verb fired for a § that's not in the corpus at all — injection
    is a no-op (no fabricated rows). The cite-grounding gate would
    reject it anyway; we don't fabricate."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_phraseology_corpus(store)
    try:
        filler = list(store.search("anything", k=2, mode="hybrid"))
        before = list(filler)
        out = _inject_missing_anchored_sections(
            filler,
            matched_sections={"99-99-99"},  # not in seeded corpus
            episodic_store=store,
            user_id=None,
        )
    finally:
        store.close()
    assert out == before


def _anchor_for(rec: EpisodicRecord) -> str | None:
    """Tiny helper for the injection tests — extracts the §-anchor from
    a record's principle. Mirrors `_extract_anchor` without requiring
    test files to import the persona internals."""
    from harness.persona.cite_grounding import _extract_anchor

    return _extract_anchor(rec.principle or "")


def test_lint_utterance_uses_verb_anchor_to_promote_into_top_3(tmp_path: Path) -> None:
    """End-to-end: lexical embedder ranks §2-4-17 first for a numeric
    utterance, but the SQUAWK anchor promotes §10-2-6 into the prompt's
    top-_PROMPT_CANDIDATES so the model sees the right section."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    # Seed a numbers-heavy §2-4-17 row (will dominate dense + bm25 on the
    # SQUAWK SEVEN FIVE ZERO ZERO query because of overlapping number tokens),
    # plus the actual hijack section, plus filler.
    store.ingest(
        external_id="seed-2-4-17",
        title="DIGIT GROUP",
        body=(
            "Numbers in transmissions are spoken digit by digit. SEVEN FIVE "
            "ZERO ZERO maps to 7500. PHRASEOLOGY examples include SEVEN, "
            "FIVE, ZERO, ZERO across many altitude and speed contexts."
        ),
        principle="JO_7110.65 §2-4-17 (Communication — DIGIT GROUP)",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="seed-10-2-6",
        title="HIJACKED AIRCRAFT",
        body=(
            "When a pilot reports a hijack, assign code 7500. PHRASEOLOGY: "
            "(Identification) SQUAWK SEVEN FIVE ZERO ZERO."
        ),
        principle="JO_7110.65 §10-2-6 (Emergency Assistance — HIJACKED AIRCRAFT)",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="seed-3-9-10",
        title="TAKEOFF CLEARANCE",
        body="RUNWAY (number), CLEARED FOR TAKEOFF.",
        principle="JO_7110.65 §3-9-10 (Departure Procedures — TAKEOFF CLEARANCE)",
        tier="seed",
        source="yaml",
    )

    adapter_with_anchor = _ScriptedAdapter(
        replies=[
            json.dumps(
                {
                    "verdict": "ok",
                    "expected_section": "10-2-6",
                    "expected_phraseology": "(Identification) SQUAWK SEVEN FIVE ZERO ZERO.",
                    "mismatch": None,
                    "citation_quote": "SQUAWK SEVEN FIVE ZERO ZERO.",
                }
            )
        ]
    )

    verdict = lint_utterance(
        "AMERICAN ONE TWENTY THREE, SQUAWK SEVEN FIVE ZERO ZERO.",
        adapter=adapter_with_anchor,
        episodic_store=store,
        verb_anchors={"10-2-6": ("SQUAWK SEVEN FIVE ZERO ZERO",)},
    )
    assert verdict.verdict == "ok"
    assert verdict.expected_section == "10-2-6"

    # Verify the prompt actually carried §10-2-6 in the candidate list —
    # without the rerank, the lexical embedder would have put §2-4-17 at
    # rank 0 (SEVEN FIVE ZERO ZERO matches its body more than the hijack
    # section's preface).
    assert len(adapter_with_anchor.calls) == 1
    user_msg = adapter_with_anchor.calls[0][1].content
    # Earliest mention of §10-2-6 should precede §2-4-17 because the
    # rerank moved it up the slate.
    assert user_msg.index("§10-2-6") < user_msg.index("§2-4-17")
