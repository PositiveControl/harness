from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore


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


# ---------- Episodic user scoping ----------


def test_episodic_alice_cannot_see_bob_memories(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    try:
        store.ingest(
            external_id="alice-mem",
            title="private for alice",
            body="alice had a thought",
            tier="working",
            source="t",
            user_id="alice",
        )
        store.ingest(
            external_id="bob-mem",
            title="private for bob",
            body="bob had a thought",
            tier="working",
            source="t",
            user_id="bob",
        )
        store.ingest(
            external_id="shared",
            title="shared",
            body="everyone sees this",
            tier="seed",
            source="yaml",
            user_id=None,
        )

        alice_hits = store.search("anything", k=10, user_id="alice")
        alice_ids = {r.external_id for r, _ in alice_hits}
        assert "alice-mem" in alice_ids
        assert "shared" in alice_ids
        assert "bob-mem" not in alice_ids

        bob_hits = store.search("anything", k=10, user_id="bob")
        bob_ids = {r.external_id for r, _ in bob_hits}
        assert "bob-mem" in bob_ids
        assert "shared" in bob_ids
        assert "alice-mem" not in bob_ids
    finally:
        store.close()


def test_episodic_owner_view_sees_everything(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    try:
        store.ingest(
            external_id="a", title="a", body="a", tier="working", source="t", user_id="alice"
        )
        store.ingest(
            external_id="b", title="b", body="b", tier="working", source="t", user_id="bob"
        )

        # user_id=None is the owner-tier view — no scope filter applied
        hits = store.search("anything", k=10, user_id=None)
        assert len({r.external_id for r, _ in hits}) == 2
    finally:
        store.close()


# ---------- Semantic user scoping ----------


def test_semantic_alice_cannot_see_bob_facts(tmp_path: Path) -> None:
    store = SemanticStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    try:
        store.add(subject="alice", predicate="prefers", object="tea", source="t", user_id="alice")
        store.add(subject="bob", predicate="prefers", object="coffee", source="t", user_id="bob")
        store.add(subject="airton", predicate="runs_on", object="mlx", source="t", user_id=None)

        alice_hits = store.search("anything", k=10, user_id="alice")
        alice_subjects = {f.subject for f, _ in alice_hits}
        assert "alice" in alice_subjects
        assert "airton" in alice_subjects  # shared
        assert "bob" not in alice_subjects

        bob_hits = store.search("anything", k=10, user_id="bob")
        bob_subjects = {f.subject for f, _ in bob_hits}
        assert "bob" in bob_subjects
        assert "airton" in bob_subjects
        assert "alice" not in bob_subjects
    finally:
        store.close()


def test_semantic_shared_facts_visible_to_every_user(tmp_path: Path) -> None:
    store = SemanticStore(tmp_path / "h.sqlite", embedder=_FakeEmbedder())
    try:
        store.add(subject="project", predicate="name", object="harness", source="t", user_id=None)
        for user in ("alice", "bob", "carol"):
            hits = store.search("anything", k=10, user_id=user)
            assert any(f.subject == "project" for f, _ in hits)
    finally:
        store.close()
