"""Tests for contextual chunking on episodic embeddings (sota punch
#6, harness-2am).

The `_build_embed_text` helper prepends a structured
`[tier: X; principle: Y; date: Z]` header to the text we feed to the
embedder. These tests exercise:

- Pure-function shape of the helper (header format, date slicing,
  missing-field paths).
- Ingest wires the helper (the embedder sees the tagged text, not
  the bare body).
- `rebuild_embeddings` migrates existing rows to the new format, so
  a fresh install that upgrades mid-life can run one rebuild and
  recover the retrieval lift.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from harness.store.episodic import EpisodicStore, _build_embed_text

# ---------- pure function ----------


def test_header_carries_tier_principle_and_date() -> None:
    out = _build_embed_text(
        title="Routing refactor",
        body="Renamed BeadsAdapter.get_focus.",
        principle="identity",
        tier="working",
        created_at_iso="2026-04-22T14:30:00+00:00",
    )
    # Header appears first, content follows.
    assert out.startswith("[tier: working; principle: identity; date: 2026-04-22]\n\n")
    # Full body + principle still present so semantic matching still works.
    assert "BeadsAdapter.get_focus" in out
    assert out.count("identity") >= 2  # once in tag, once as standalone line


def test_header_omits_missing_principle() -> None:
    out = _build_embed_text(
        title="note",
        body="b",
        principle=None,
        tier="seed",
        created_at_iso="2026-04-22T00:00:00Z",
    )
    assert out.startswith("[tier: seed; date: 2026-04-22]\n\n")
    assert "principle" not in out.split("\n\n")[0]  # not in the header line


def test_header_omits_missing_date() -> None:
    out = _build_embed_text(
        title="t",
        body="b",
        principle=None,
        tier="working",
        created_at_iso=None,
    )
    assert out.startswith("[tier: working]\n\n")


def test_date_uses_day_granularity_only() -> None:
    """ISO 8601 dates carry time zones + microseconds that add noise
    to embeddings. The header should keep only YYYY-MM-DD."""
    out = _build_embed_text(
        title="t",
        body="b",
        principle=None,
        tier="working",
        created_at_iso="2026-04-22T14:30:59.123456+00:00",
    )
    assert "date: 2026-04-22]" in out
    assert "14:30" not in out


def test_body_preserved_verbatim() -> None:
    """Downstream FTS5 indexes the body column directly, so the
    embed-text transform must not mutate the narrative content."""
    body = "line 1\nline 2\nline 3"
    out = _build_embed_text(
        title="t",
        body=body,
        principle=None,
        tier="working",
        created_at_iso=None,
    )
    assert body in out


# ---------- ingest integration ----------


@dataclass
class _CapturingEmbedder:
    """Records every .embed() call so tests can assert what text the
    store actually fed the embedder."""

    id: str = "fake"
    dimension: int = 4
    seen: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        self.seen.append(batch)
        vectors: list[np.ndarray] = []
        for _ in batch:
            vectors.append(np.ones(4, dtype=np.float32) / 2.0)
        return np.stack(vectors)


@pytest.fixture
def embedder() -> _CapturingEmbedder:
    return _CapturingEmbedder()


@pytest.fixture
def store(tmp_path: Path, embedder: _CapturingEmbedder) -> EpisodicStore:
    return EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)


def test_ingest_embeds_the_tagged_header(
    store: EpisodicStore, embedder: _CapturingEmbedder
) -> None:
    store.ingest(
        external_id="ep-1",
        title="Routing refactor",
        body="renamed BeadsAdapter.get_focus",
        principle="identity",
        tier="working",
        source="user",
    )
    assert embedder.seen, "embedder was never called"
    embedded = embedder.seen[0][0]
    assert embedded.startswith("[tier: working; principle: identity; date: ")


def test_ingest_handles_missing_principle(
    store: EpisodicStore, embedder: _CapturingEmbedder
) -> None:
    store.ingest(
        external_id="ep-2",
        title="Unrelated note",
        body="b",
        principle=None,
        tier="working",
        source="user",
    )
    embedded = embedder.seen[0][0]
    assert embedded.startswith("[tier: working; date: ")
    # No principle field even when the caller omitted it.
    first_line = embedded.split("\n")[0]
    assert "principle" not in first_line


# ---------- rebuild_embeddings migration ----------


def test_rebuild_embeddings_re_applies_contextual_chunking(
    tmp_path: Path,
) -> None:
    """Simulate an install whose rows were embedded under the pre-
    chunking format: after rebuild_embeddings, the embedder sees
    every row's text with the new tagged header. Users running the
    existing `harness memory rebuild-embeddings` command pick up the
    retrieval lift without any other migration step."""
    embedder_v1 = _CapturingEmbedder()
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder_v1)
    # Ingest under the (now contextual) format.
    store.ingest(
        external_id="a",
        title="Alpha",
        body="body of a",
        principle="identity",
        tier="seed",
        source="yaml",
    )
    store.ingest(
        external_id="b",
        title="Beta",
        body="body of b",
        principle=None,
        tier="working",
        source="user",
    )
    # Clear the capture log so we only see what rebuild itself sent.
    embedder_v1.seen = []
    updated, _ = store.rebuild_embeddings()
    assert updated == 2
    # Rebuild batches all rows into a single .embed() call.
    assert len(embedder_v1.seen) == 1
    batch = embedder_v1.seen[0]
    assert len(batch) == 2
    # Both rows carry the header; row A has principle, row B doesn't.
    assert batch[0].startswith("[tier: seed; principle: identity; date: ")
    assert batch[1].startswith("[tier: working; date: ")


def test_rebuild_passes_over_superseded_rows(tmp_path: Path) -> None:
    """Superseded rows are retired from retrieval; rebuild should
    skip them so we don't waste embedder work."""
    embedder = _CapturingEmbedder()
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    keep = store.ingest(
        external_id="keep",
        title="Keep",
        body="b",
        tier="working",
        source="user",
    )
    retired = store.ingest(
        external_id="old",
        title="Old",
        body="b",
        tier="working",
        source="user",
    )
    store.mark_superseded(retired.id, by=keep.id)
    embedder.seen = []
    updated, _ = store.rebuild_embeddings()
    assert updated == 1
    assert len(embedder.seen[0]) == 1
