"""Per-character memory DB path resolution — harness-aq9.

Verifies that airton and airton_b resolve to distinct database files so
ab's episodic/semantic silo never merges with Airton's, and that the
environment override (HARNESS_AB_MEMORY_DIR) is honoured. Also
integration-tests the isolation end to end: writing a fact through ab's
SemanticStore must not appear when reading Airton's store.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from harness.config import Settings
from harness.store.semantic import SemanticStore


class _StubEmbedder:
    """Deterministic embedder — seeded by text so results are stable.
    Keeps the isolation test tight on the storage layer, not on the
    embedder's runtime cost."""

    id = "stub-384"
    dimension = 384

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm else vec)
        return np.vstack(rows).astype(np.float32)


def test_airton_resolves_to_default_db_path(tmp_path: Path) -> None:
    s = Settings(root=tmp_path, character_name="airton")
    # Default character keeps the repo-relative data/harness.sqlite —
    # no relocation of existing Airton memory.
    assert s.db_path_for("airton") == s.db_path
    assert s.character_db_path == s.db_path


def test_airton_b_resolves_to_isolated_path(tmp_path: Path) -> None:
    s = Settings(
        root=tmp_path,
        character_name="airton_b",
        ab_bd_dir=tmp_path / "ab",
    )
    resolved = s.db_path_for("airton_b")
    # ab's DB lands under <ab_bd_dir>/memory/ by default, so bd data
    # and memory live in the same per-character dir for unified backup.
    assert resolved == tmp_path / "ab" / "memory" / "harness.sqlite"
    assert resolved != s.db_path
    # The parent dir is eagerly created so stores can open without
    # having to mkdir themselves.
    assert resolved.parent.exists()


def test_ab_memory_dir_override_respected(tmp_path: Path) -> None:
    explicit = tmp_path / "custom" / "memory"
    s = Settings(
        root=tmp_path,
        character_name="airton_b",
        ab_bd_dir=tmp_path / "ab",
        ab_memory_dir=explicit,
    )
    assert s.ab_memory_dir_resolved == explicit
    assert s.db_path_for("airton_b") == explicit / "harness.sqlite"


def test_unknown_character_falls_back_to_default(tmp_path: Path) -> None:
    s = Settings(root=tmp_path, character_name="some_other")
    # Only airton_b is isolated in v1; any other character shares the
    # default path. New isolation cohorts land deliberately, not by
    # string-matching accident.
    assert s.db_path_for("some_other") == s.db_path
    assert s.db_path_for("random_char") == s.db_path


def test_character_db_path_tracks_character_name(tmp_path: Path) -> None:
    # Switching character_name flips the resolved path without touching
    # anything else in config.
    airton_settings = Settings(root=tmp_path, character_name="airton")
    ab_settings = Settings(
        root=tmp_path,
        character_name="airton_b",
        ab_bd_dir=tmp_path / "ab",
    )
    assert airton_settings.character_db_path != ab_settings.character_db_path


def test_semantic_stores_at_distinct_paths_are_isolated(tmp_path: Path) -> None:
    """End-to-end: write a fact through ab's store; Airton's store must
    still report empty. Uses real SQLite files at tmp_path to confirm
    the isolation invariant holds at the filesystem layer, not just in
    the path resolver."""
    airton_db = tmp_path / "airton" / "harness.sqlite"
    ab_db = tmp_path / "ab" / "memory" / "harness.sqlite"
    embedder = _StubEmbedder()

    airton_store = SemanticStore(airton_db, embedder=embedder)
    ab_store = SemanticStore(ab_db, embedder=embedder)

    ab_store.add(
        subject="mark",
        predicate="prefers",
        object="deep work 9-12",
        source="test",
    )

    ab_facts = ab_store.all()
    airton_facts = airton_store.all()

    assert len(ab_facts) == 1
    assert airton_facts == []


def test_ab_memory_dir_resolved_default_under_bd_dir(tmp_path: Path) -> None:
    s = Settings(root=tmp_path, ab_bd_dir=tmp_path / "ab_root")
    # Default colocation: memory dir hangs off the bd dir so one
    # <ab dir> backup captures both planes.
    assert s.ab_memory_dir_resolved == tmp_path / "ab_root" / "memory"


def test_settings_monkeypatch_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Env-var-driven override via HARNESS_AB_MEMORY_DIR — verifies the
    pydantic-settings prefix wiring holds for the new keys."""
    override = tmp_path / "env" / "mem"
    monkeypatch.setenv("HARNESS_AB_MEMORY_DIR", str(override))
    monkeypatch.setenv("HARNESS_CHARACTER_NAME", "airton_b")
    monkeypatch.setenv("HARNESS_AB_BD_DIR", str(tmp_path / "env" / "bd"))
    s = Settings(root=tmp_path)
    assert s.ab_memory_dir_resolved == override
    assert s.db_path_for("airton_b") == override / "harness.sqlite"
