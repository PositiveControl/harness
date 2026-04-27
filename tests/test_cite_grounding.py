"""Tests for the cite-grounding catcher (lane F-prime, harness-11ha)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.persona.cite_grounding import (
    CiteGroundingResult,
    _extract_anchor,
    check_cite_groundedness,
)
from harness.store.episodic import EpisodicStore


@dataclass
class _LexicalEmbedder:
    """Token-bag deterministic embedder for tests. Each text becomes a
    16-dim L2-normalised vector keyed by char-bucket counts. Two texts
    sharing tokens score higher than two sharing none — enough signal
    for retrieval ordering tests without pulling a real ST model."""

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


def _seed_atc_corpus(store: EpisodicStore) -> None:
    """Seed three §-form chunks shaped like JO 7110.65 corpus."""
    store.ingest(
        external_id="seed-3-10-3",
        title="SAME RUNWAY SEPARATION",
        body="The arriving aircraft must not cross the landing threshold "
        "until the preceding aircraft has landed and is clear of the runway.",
        principle="JO_7110.65 §3-10-3 (Air Traffic Control — SAME RUNWAY SEPARATION)",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="seed-3-10-4",
        title="TAXI AND GROUND MOVEMENT",
        body="Taxi instructions for ground movement of aircraft on airport "
        "movement areas. Ground movement procedures.",
        principle="JO_7110.65 §3-10-4 (Air Traffic Control — TAXI AND GROUND MOVEMENT)",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="seed-10-2-6",
        title="HIJACKED AIRCRAFT",
        body="When an aircraft squawks 7500, verify by saying 'verify squawking "
        "seven five zero zero'. Assume hijack if no acknowledgment.",
        principle="JO_7110.65 §10-2-6 (Emergency Assistance — HIJACKED AIRCRAFT)",
        tier="seed",
        source="yaml",
    )


# ---------- _extract_anchor ----------


def test_extract_anchor_strips_jo_prefix() -> None:
    assert _extract_anchor("JO 7110.65 §10-2-5") == "10-2-5"


def test_extract_anchor_strips_principle_form() -> None:
    principle = "JO_7110.65 §3-10-3 (Air Traffic Control — SAME RUNWAY SEPARATION)"
    assert _extract_anchor(principle) == "3-10-3"


def test_extract_anchor_handles_aim() -> None:
    assert _extract_anchor("AIM 5-3-8") == "5-3-8"


def test_extract_anchor_handles_cfr() -> None:
    assert _extract_anchor("14 CFR §91.155") == "91.155"


def test_extract_anchor_unicode_minus() -> None:
    assert _extract_anchor("§10−2−5") == "10-2-5"  # noqa: RUF001


def test_extract_anchor_empty_on_no_match() -> None:
    assert _extract_anchor("no citation here") == ""
    assert _extract_anchor("") == ""


# ---------- check_cite_groundedness ----------


def test_grounded_when_cite_in_top_k(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        # Question about runway threshold landing → §3-10-3 should
        # surface in top-K and the §3-10-3 cite should be grounded.
        result = check_cite_groundedness(
            question="How soon after a plane lands can the next cross the threshold?",
            reply="JO 7110.65 §3-10-3 — once the preceding aircraft is clear "
            "of the runway, the next can cross the threshold.",
            episodic_store=store,
        )
        assert len(result.checks) == 1
        assert result.checks[0].section == "3-10-3"
        assert result.checks[0].grounded is True
        assert result.checks[0].suggested is None
        assert result.has_ungrounded is False
    finally:
        store.close()


def test_ungrounded_when_cite_not_in_top_k(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        # Question about runway threshold landing — model cited §3-10-4
        # (Taxi/Ground Movement, real-but-wrong). The catcher should
        # detect ungrounded and suggest §3-10-3 (top-1 retrieval).
        result = check_cite_groundedness(
            question="How soon after a plane lands can the next cross the threshold?",
            reply="JO 7110.65 §3-10-4 — taxi the next aircraft to the threshold.",
            episodic_store=store,
            k=1,  # tight K: only top-1 grounds; §3-10-4 isn't top-1 here
        )
        assert len(result.checks) == 1
        check = result.checks[0]
        assert check.section == "3-10-4"
        assert check.grounded is False
        assert check.suggested == "3-10-3"
        assert result.has_ungrounded is True
    finally:
        store.close()


def test_no_citations_returns_empty(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        result = check_cite_groundedness(
            question="generic question",
            reply="reply with no citation pattern at all.",
            episodic_store=store,
        )
        assert result.checks == ()
        assert result.has_ungrounded is False
    finally:
        store.close()


def test_dedup_by_anchor(tmp_path: Path) -> None:
    """Multiple surface forms of the same section dedupe by anchor —
    one CiteCheck per unique section, not per surface mention."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        result = check_cite_groundedness(
            question="hijack squawk question",
            reply="JO 7110.65 §10-2-6 covers it. Per §10-2-6 verify the squawk.",
            episodic_store=store,
        )
        # Two surface mentions of §10-2-6 → one check entry
        sections = [c.section for c in result.checks]
        assert sections.count("10-2-6") == 1
    finally:
        store.close()


def test_suggested_none_when_grounded(tmp_path: Path) -> None:
    """Grounded cites carry suggested=None even if the top-1 anchor is
    a different section — we only suggest replacements for ungrounded."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        result = check_cite_groundedness(
            question="hijacked aircraft squawk seven five zero zero",
            reply="JO 7110.65 §10-2-6 — verify the squawk.",
            episodic_store=store,
            k=10,
        )
        check = result.checks[0]
        assert check.grounded is True
        assert check.suggested is None
    finally:
        store.close()


def test_suggested_omitted_when_top1_equals_cite(tmp_path: Path) -> None:
    """When the top-1 retrieval IS the cited section, we don't suggest
    a replacement that would be a no-op."""
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        # k=0 forces every cite to be ungrounded; we still shouldn't
        # suggest the cite back at itself.
        result = check_cite_groundedness(
            question="hijacked aircraft squawk seven five zero zero",
            reply="JO 7110.65 §10-2-6 — verify the squawk.",
            episodic_store=store,
            k=0,  # nothing retrieved → no top-1 candidate
        )
        check = result.checks[0]
        assert check.grounded is False
        # k=0 → no top-1 candidate → suggested is None
        assert check.suggested is None
    finally:
        store.close()


def test_cite_grounding_result_helpers(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_LexicalEmbedder())
    _seed_atc_corpus(store)
    try:
        result = check_cite_groundedness(
            question="threshold landing question",
            reply="JO 7110.65 §3-10-4 ground movement. Also JO 7110.65 §3-10-3 same runway.",
            episodic_store=store,
            k=1,
        )
        # §3-10-3 grounded (top-1), §3-10-4 not in top-1 → ungrounded
        assert result.has_ungrounded is True
        ungrounded_sections = {c.section for c in result.ungrounded}
        assert "3-10-4" in ungrounded_sections
        assert "3-10-3" not in ungrounded_sections
    finally:
        store.close()


def test_empty_result_helpers(tmp_path: Path) -> None:
    """Empty CiteGroundingResult.has_ungrounded is False, ungrounded is ()."""
    result = CiteGroundingResult(checks=())
    assert result.has_ungrounded is False
    assert result.ungrounded == ()
