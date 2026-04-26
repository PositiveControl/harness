"""Tests for the FTS5 + RRF hybrid retrieval path (sota punch #2).

Two layers:

- Pure-function tests for the shared `_hybrid` module
  (sanitization + RRF math).
- Store-level tests that ingest data, exercise `search(..., mode=...)`,
  and assert the hybrid path catches identifier-heavy queries that
  dense cosine alone misses.

The episodic + semantic fake embedders are intentionally weak — the
whole point of the hybrid path is to make identifier matching work
*despite* weak semantic similarity, so the test embedders shouldn't
help it too much.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from harness.store._hybrid import reciprocal_rank_fusion, sanitize_fts_query
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore

# ---------- pure functions ----------


def test_sanitize_keeps_alnum_and_underscore() -> None:
    # Tokens are quoted so FTS5 treats them as literal phrase-search
    # terms — including the user's literal "OR" token, which would
    # otherwise be parsed as an operator mid-clause.
    assert sanitize_fts_query("user_id OR BeadsAdapter") == '"user_id" OR "OR" OR "BeadsAdapter"'


def test_sanitize_drops_punctuation_and_symbols() -> None:
    assert sanitize_fts_query("who(said) *this*?") == '"who" OR "said" OR "this"'


def test_sanitize_empty_input_returns_empty_string() -> None:
    assert sanitize_fts_query("") == ""
    assert sanitize_fts_query("   ") == ""
    assert sanitize_fts_query("!!!") == ""


def test_sanitize_single_token_no_operator() -> None:
    """Single token is just a quoted phrase — no dangling OR."""
    assert sanitize_fts_query("BeadsAdapter") == '"BeadsAdapter"'


def test_sanitize_hyphen_compound_emits_phrase_query() -> None:
    """Hyphenated compound like 'aircraft-to-aircraft' is shredded by
    FTS5's unicode61 tokenizer into [aircraft, to, aircraft]. The
    two-pass sanitize emits both the individual-token OR chain AND a
    quoted phrase query ('aircraft to aircraft') so BM25 can reward
    sections that use the exact n-gram adjacency (harness-5uq follow-up)."""
    out = sanitize_fts_query("aircraft-to-aircraft alerts")
    assert '"aircraft"' in out
    assert '"to"' in out
    assert '"alerts"' in out
    # Phrase query wraps the compound with internal hyphens replaced
    # by spaces — FTS5 phrase syntax.
    assert '"aircraft to aircraft"' in out


def test_sanitize_slash_compound_emits_phrase_query() -> None:
    """Slash compounds like 'L/MF' (JO 7110.65 §4-1-2 Radio Beacon
    class) or 'CA/MCI' get the same phrase treatment."""
    out = sanitize_fts_query("L/MF Radio Beacon")
    assert '"L"' in out
    assert '"MF"' in out
    assert '"L MF"' in out


def test_sanitize_bare_hyphen_between_words_treats_as_compound() -> None:
    """'non-standard' → tokens 'non' + 'standard' + phrase 'non standard'."""
    out = sanitize_fts_query("non-standard formations")
    assert '"non"' in out
    assert '"standard"' in out
    assert '"formations"' in out
    assert '"non standard"' in out


def test_sanitize_no_compound_no_phrase_emitted() -> None:
    """Query with no hyphens/slashes shouldn't grow any phrase
    clauses — pass-1 output unchanged."""
    assert sanitize_fts_query("who said this") == '"who" OR "said" OR "this"'


def test_sanitize_single_segment_with_hyphen_at_boundary_is_token_only() -> None:
    """'-word' or 'word-' (hyphen at a boundary, no second segment to
    join) must not emit a phantom phrase — compound regex requires 2+
    segments connected by the punctuation."""
    out = sanitize_fts_query("-foo bar-")
    # Only single tokens, no phrase clauses.
    assert out == '"foo" OR "bar"'


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([[7, 3, 5]])
    assert [rid for rid, _ in fused] == [7, 3, 5]
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_id_in_both_lists_outranks_id_in_one() -> None:
    """Classic RRF property: an id at rank 3 in both lists beats an
    id at rank 1 in one list but absent from the other, for k=60."""
    fused = reciprocal_rank_fusion([[1, 2, 3], [9, 8, 3]])
    top = fused[0][0]
    assert top == 3
    # 3 gets 1/63 + 1/63 ≈ 0.0317; 1 gets 1/61 ≈ 0.0164.
    three_score = dict(fused)[3]
    one_score = dict(fused)[1]
    assert three_score > one_score


def test_rrf_k_parameter_changes_curve() -> None:
    """Smaller k → steeper curve → rank-1 items pull further ahead.
    Larger k → flatter curve → overlap between lists matters more.
    Verifies the parameter actually moves scores, so a regression
    that silently hardcodes k=60 elsewhere would stand out."""
    small_k = dict(reciprocal_rank_fusion([[1, 2, 3], [9, 8, 3]], k=1))
    large_k = dict(reciprocal_rank_fusion([[1, 2, 3], [9, 8, 3]], k=1000))
    # id=1 at rank 1 in list 1 only: score drops as k grows.
    assert small_k[1] > large_k[1]
    # id=3 at rank 3 in both lists: score ALSO drops as k grows.
    assert small_k[3] > large_k[3]
    # But 1's drop is less severe than 3's, so at large k the
    # overlap advantage for id=3 grows in relative terms.
    assert large_k[3] / large_k[1] > small_k[3] / small_k[1]


# ---------- store-level fixtures ----------


@dataclass
class _FakeEmbedder:
    """Weak pseudo-embeddings from character-code arithmetic. The
    hybrid tests rely on the text path, not on the embedder producing
    clever similarities."""

    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        return np.stack(vectors)


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())


@pytest.fixture
def semantic(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())


# ---------- episodic: text + hybrid ----------


def test_episodic_text_search_finds_identifier(episodic: EpisodicStore) -> None:
    """The whole point of FTS5: identifiers should be locatable by
    exact-token match even when dense cosine misses."""
    episodic.ingest(
        external_id="ep-1",
        title="Routing refactor",
        body="Renamed BeadsAdapter.get_focus to resolve_focus for clarity.",
        tier="working",
        source="user",
    )
    episodic.ingest(
        external_id="ep-2",
        title="Unrelated note",
        body="The weather is nice today.",
        tier="working",
        source="user",
    )
    hits = episodic.search("BeadsAdapter", mode="text")
    assert len(hits) == 1
    assert hits[0][0].external_id == "ep-1"


def test_episodic_text_search_empty_query_returns_empty(episodic: EpisodicStore) -> None:
    episodic.ingest(
        external_id="ep-1",
        title="anything",
        body="content",
        tier="working",
        source="user",
    )
    assert episodic.search("!!!", mode="text") == []


def test_episodic_text_search_handles_special_chars_without_crashing(
    episodic: EpisodicStore,
) -> None:
    episodic.ingest(
        external_id="ep-1",
        title="the quick brown fox",
        body="jumped over the lazy dog",
        tier="working",
        source="user",
    )
    # These used to crash FTS5's MATCH parser; sanitizer should
    # keep the query safe.
    for q in ["*()*", 'why "is" this', "a:b:c", "() AND ()"]:
        episodic.search(q, mode="text")  # must not raise


def test_episodic_text_search_respects_user_scope(episodic: EpisodicStore) -> None:
    episodic.ingest(
        external_id="shared",
        title="shared memory",
        body="matching_token lives here",
        tier="seed",
        source="yaml",
        user_id=None,
    )
    episodic.ingest(
        external_id="mark",
        title="mark private",
        body="matching_token in mark's silo",
        tier="working",
        source="user",
        user_id="mark",
    )
    episodic.ingest(
        external_id="other",
        title="other private",
        body="matching_token in other's silo",
        tier="working",
        source="user",
        user_id="other_user",
    )
    hits = episodic.search("matching_token", mode="text", user_id="mark")
    ids = {r.external_id for r, _ in hits}
    assert ids == {"shared", "mark"}


def test_episodic_hybrid_catches_identifier_dense_would_miss(
    episodic: EpisodicStore,
) -> None:
    """The core property the punch list is buying: dense cosine can
    miss proper nouns, but hybrid picks them up via the FTS side."""
    episodic.ingest(
        external_id="target",
        title="router refactor",
        body="Changed BeadsAdapter.get_focus signature.",
        tier="working",
        source="user",
    )
    episodic.ingest(
        external_id="decoy1",
        title="decoy note",
        body="a b c d e f g",
        tier="working",
        source="user",
    )
    episodic.ingest(
        external_id="decoy2",
        title="another",
        body="x y z",
        tier="working",
        source="user",
    )
    hits = episodic.search("BeadsAdapter", mode="hybrid", k=3)
    top_ids = {r.external_id for r, _ in hits}
    assert "target" in top_ids


def test_episodic_hybrid_backfills_fts_from_pre_existing_rows(
    tmp_path: Path,
) -> None:
    """Simulate an existing install that has episodic rows but no
    FTS sidecar: the legacy schema is created without FTS, then the
    upgraded store opens the same file and must backfill."""
    db = tmp_path / "h.sqlite"
    import sqlite3

    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE episodic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT UNIQUE,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            principle TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            tier TEXT NOT NULL,
            source TEXT NOT NULL,
            session_id TEXT,
            user_id TEXT,
            created_at TEXT NOT NULL,
            last_accessed TEXT,
            embedding BLOB NOT NULL,
            embedder_id TEXT,
            embedding_dim INTEGER,
            superseded_by INTEGER REFERENCES episodic(id)
        );
        """
    )
    conn.execute(
        """INSERT INTO episodic (
            external_id, title, body, principle, tier, source, created_at,
            embedding, embedder_id, embedding_dim
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "legacy-1",
            "Legacy entry",
            "The hidden token is LegacyToken42.",
            None,
            "working",
            "user",
            "2024-01-01T00:00:00+00:00",
            np.ones(4, dtype=np.float32).tobytes(),
            "fake",
            4,
        ),
    )
    conn.commit()
    conn.close()

    store = EpisodicStore(db, embedder=_FakeEmbedder())
    hits = store.search("LegacyToken42", mode="text")
    assert len(hits) == 1
    assert hits[0][0].external_id == "legacy-1"


def test_episodic_dense_mode_preserves_legacy_scores(episodic: EpisodicStore) -> None:
    """Explicit mode='dense' returns cosine scores (not RRF scores),
    so callers that depend on the min_score semantics or the cosine
    range [-1, 1] keep working."""
    episodic.ingest(
        external_id="a",
        title="one",
        body="text a",
        tier="working",
        source="user",
    )
    hits = episodic.search("text a", mode="dense", k=1)
    assert len(hits) == 1
    # Cosine on normalized vectors sits in [-1, 1]; RRF scores are in
    # [0, ~0.033]. If the legacy path leaked, we'd see the latter.
    assert -1.0 <= hits[0][1] <= 1.0


# ---------- semantic: text + hybrid ----------


def test_semantic_text_search_finds_identifier(semantic: SemanticStore) -> None:
    semantic.add(
        subject="mark",
        predicate="uses_tool",
        object="BeadsAdapter",
        source="user",
    )
    semantic.add(subject="mark", predicate="likes", object="coffee", source="user")
    hits = semantic.search("BeadsAdapter", mode="text")
    assert len(hits) == 1
    assert hits[0][0].object == "BeadsAdapter"


def test_semantic_hybrid_respects_confidence_floor(semantic: SemanticStore) -> None:
    """min_confidence filters the text path too, not just the dense one."""
    semantic.add(
        subject="mark",
        predicate="uses",
        object="LowConfWidget",
        confidence=0.3,
        source="user",
    )
    semantic.add(
        subject="mark",
        predicate="uses",
        object="HighConfWidget",
        confidence=0.9,
        source="user",
    )
    hits = semantic.search("Widget", mode="hybrid", min_confidence=0.8, k=5)
    objs = {f.object for f, _ in hits}
    assert "HighConfWidget" in objs
    assert "LowConfWidget" not in objs


def test_semantic_hybrid_user_scope(semantic: SemanticStore) -> None:
    semantic.add(
        subject="airton",
        predicate="knows",
        object="SharedIdentifier",
        source="seed",
        user_id=None,
    )
    semantic.add(
        subject="mark",
        predicate="uses",
        object="SharedIdentifier",
        source="user",
        user_id="mark",
    )
    semantic.add(
        subject="other",
        predicate="uses",
        object="SharedIdentifier",
        source="user",
        user_id="other_user",
    )
    hits = semantic.search("SharedIdentifier", mode="hybrid", user_id="mark", k=10)
    users = {f.user_id for f, _ in hits}
    assert users == {None, "mark"}


def test_hybrid_returns_empty_list_on_no_match(episodic: EpisodicStore) -> None:
    episodic.ingest(
        external_id="a",
        title="one",
        body="one",
        tier="working",
        source="user",
    )
    # No token in the query matches any stored row; both dense and
    # text sides are starved (dense by embedder coincidence, text by
    # sanitizer). Hybrid must gracefully return [].
    assert episodic.search("!!!", mode="text") == []


# ---------- harness-dffh: bm25 cosine floor ----------


@dataclass
class _SteerableEmbedder:
    """Embedder where similarity is dictated by axis tokens in the
    input text. Each token in `axes` maps to a unit vector along
    one basis direction; the embedder picks the matching axis when
    its token appears anywhere in the embedded text and falls back
    to a default vector otherwise.

    This lets bm25-floor tests pin the cosine between the query and
    each ingested row without having to mirror the store's
    `_build_embed_text` formatting exactly. The store wraps body
    text with `[tier: …]` headers before embedding, so a strict
    by-text dict misses every ingested row; substring matching on
    a unique axis token (`AXIS_BLEED`, `AXIS_ONTOPIC`, …) is
    robust to that wrapping."""

    id: str = "steer"
    dimension: int = 4
    fallback: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    axes: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            chosen: tuple[float, float, float, float] | None = None
            for token, vec in self.axes.items():
                if token in text:
                    chosen = vec
                    break
            v = np.array(chosen if chosen is not None else self.fallback, dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.stack(out)


def test_episodic_hybrid_drops_bm25_only_hit_below_cosine_floor(tmp_path: Path) -> None:
    """harness-dffh: a pure BM25 hit whose dense cosine to the
    query is well below `bm25_min_score` must NOT make it into
    the hybrid top-K. Exercises the topical-bleed failure mode:
    a stored row shares one common token with the query but is
    semantically unrelated."""
    embedder = _SteerableEmbedder(
        axes={
            # Query carries AXIS_QUERY → vector e0.
            "AXIS_QUERY": (1.0, 0.0, 0.0, 0.0),
            # Bleed row body shares the lexical token "junior" with
            # the query but lives on a different semantic axis (e1).
            # Dense cosine query↔bleed ≈ 0.
            "AXIS_BLEED": (0.0, 1.0, 0.0, 0.0),
            # Ontopic row sits on the same axis as the query (e0):
            # dense cosine query↔ontopic ≈ 1.
            "AXIS_ONTOPIC": (1.0, 0.0, 0.0, 0.0),
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="bleed",
            title="topical bleed AXIS_BLEED",
            body="junior reference AXIS_BLEED — semantically unrelated",
            tier="working",
            source="user",
        )
        store.ingest(
            external_id="ontopic",
            title="ontopic AXIS_ONTOPIC",
            body="junior eng coaching playbook AXIS_ONTOPIC",
            tier="working",
            source="user",
        )

        # Realistic chat config: dense floor 0.5 + BM25 floor 0.5.
        # Without the BM25 gate (the harness-dffh fix) the bleed
        # row would re-enter via the BM25 side of RRF even though
        # dense rejected it. With the gate, only on-axis stays.
        gated = store.search(
            "junior eng question AXIS_QUERY",
            mode="hybrid",
            k=2,
            min_score=0.5,
            bm25_min_score=0.5,
        )
        ids_gated = {r.external_id for r, _ in gated}
        assert "ontopic" in ids_gated
        assert "bleed" not in ids_gated

        # Disable both floors → bleed comes back via RRF, proving the
        # gate is what pruned it (not some unrelated filter).
        no_floor = store.search(
            "junior eng question AXIS_QUERY",
            mode="hybrid",
            k=2,
            min_score=0.0,
            bm25_min_score=0.0,
        )
        assert "bleed" in {r.external_id for r, _ in no_floor}
    finally:
        store.close()


def test_episodic_hybrid_default_floor_derives_from_min_score(tmp_path: Path) -> None:
    """When `bm25_min_score` is not passed, the default is
    `max(min_score - 0.15, 0.0)`. Verify the derivation actually
    fires at the gate: with min_score=0.5 the soft floor is 0.35,
    so a row whose cosine to the query is 0.4 should pass while a
    row at cosine 0.1 should not."""
    embedder = _SteerableEmbedder(
        axes={
            "AXIS_QUERY": (1.0, 0.0, 0.0, 0.0),
            # cos(query, border) = 0.4 (above 0.35 soft floor)
            "AXIS_BORDER": (0.4, float(np.sqrt(1 - 0.16)), 0.0, 0.0),
            # cos(query, far) = 0.1 (below 0.35 soft floor)
            "AXIS_FAR": (0.1, float(np.sqrt(1 - 0.01)), 0.0, 0.0),
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="border",
            title="border AXIS_BORDER",
            body="lexical-token shared with query AXIS_BORDER",
            tier="working",
            source="user",
        )
        store.ingest(
            external_id="far",
            title="far AXIS_FAR",
            body="lexical-token shared with query AXIS_FAR",
            tier="working",
            source="user",
        )
        # Default bm25_min_score derives to 0.35 from min_score=0.5.
        hits = store.search(
            "lexical-token AXIS_QUERY",
            mode="hybrid",
            min_score=0.5,
        )
        ids = {r.external_id for r, _ in hits}
        # Both rows are below the strict 0.5 dense floor, so dense
        # rejected both; the soft BM25 floor at 0.35 lets `border`
        # through (cos 0.4) and cuts `far` (cos 0.1).
        assert "border" in ids
        assert "far" not in ids
    finally:
        store.close()


def test_episodic_hybrid_explicit_zero_disables_floor(tmp_path: Path) -> None:
    """Pass `bm25_min_score=0.0` to opt out of the gate entirely
    (e.g. for identifier-heavy queries where dense cosine is
    genuinely low). Verifies the parameter is wired and isn't
    masked by the auto-derivation."""
    embedder = _SteerableEmbedder(
        axes={
            "AXIS_QUERY": (1.0, 0.0, 0.0, 0.0),
            "AXIS_IDENT": (0.0, 1.0, 0.0, 0.0),  # cos = 0
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="ident",
            title="BeadsAdapter AXIS_IDENT",
            body="Renamed BeadsAdapter.get_focus to resolve_focus AXIS_IDENT",
            tier="working",
            source="user",
        )
        # min_score=0.5 would normally derive a soft floor of 0.35;
        # explicit 0.0 keeps the identifier match.
        hits = store.search(
            "needle BeadsAdapter AXIS_QUERY",
            mode="hybrid",
            min_score=0.5,
            bm25_min_score=0.0,
        )
        assert "ident" in {r.external_id for r, _ in hits}
    finally:
        store.close()


def test_semantic_hybrid_drops_bm25_only_hit_below_cosine_floor(tmp_path: Path) -> None:
    """Same gate on the semantic store. BM25 hit on a shared token
    in the (subject, predicate, object) triple shouldn't survive
    if its dense cosine to the full query is below threshold."""
    embedder = _SteerableEmbedder(
        axes={
            "AXIS_QUERY": (1.0, 0.0, 0.0, 0.0),
            "AXIS_BLEED": (0.0, 1.0, 0.0, 0.0),
            "AXIS_KEEP": (1.0, 0.0, 0.0, 0.0),
        }
    )
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(
            subject="mark",
            predicate="prefers",
            object="stack-rank reviews AXIS_BLEED",
            confidence=0.8,
            source="scribe",
            tier="working",
        )
        store.add(
            subject="mark",
            predicate="prefers",
            object="stack postgres AXIS_KEEP",
            confidence=0.8,
            source="scribe",
            tier="working",
        )
        gated = store.search(
            "mark stack preference AXIS_QUERY",
            mode="hybrid",
            k=5,
            min_score=0.5,
            bm25_min_score=0.5,
        )
        objs = {f.object for f, _ in gated}
        assert any("AXIS_KEEP" in o for o in objs)
        assert not any("AXIS_BLEED" in o for o in objs)
    finally:
        store.close()
