"""Behavioral contract for the store/embedder helpers in the CLI.

These four helpers (`_load_embedder`, `_maybe_retriever`,
`_open_episodic_store`, `_open_semantic_store`) are step 1 of
docs/cli-extraction-plan.md — they move out of `cli.py` into
`cli_store.py`. Before harness-z4k1.1 they had exactly one test touching
them, and that one only monkeypatched `_load_embedder` away.

The tests pin BEHAVIOR, not location: they import through the
`harness.cli` surface, which keeps working after the move via
re-export, so they characterize the code before the extraction and
regression-guard it after. The last test is the extraction's landmine
guard — the embedder cache is process-wide, and a second cache created
during the move would silently load the model twice.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harness import cli
from harness.character import load_character

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AIRTON = _REPO_ROOT / "character" / "airton"


@dataclass
class _CountingEmbedder:
    """Stands in for SentenceTransformersEmbedder without loading a model."""

    id: str = "counting-fake"
    dimension: int = 4
    constructions: list[str] = field(default_factory=list)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        return np.stack(vectors) if vectors else np.zeros((0, self.dimension), dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_embedder_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a cold cache and leaves the real one alone.

    `_cached_embedder` is a module global that survives for the life of
    the process; without this the first test to load an embedder would
    decide the outcome of all the others.

    Resolved through `_load_embedder.__module__` rather than named
    directly, so the reset follows the helper to whichever module owns
    it (cli.py before the step-1 extraction, cli_store.py after).
    """
    owner = sys.modules[cli._load_embedder.__module__]
    monkeypatch.setattr(owner, "_cached_embedder", owner._EMBEDDER_SENTINEL)


@pytest.fixture
def fake_embedder_class(monkeypatch: pytest.MonkeyPatch) -> list[_CountingEmbedder]:
    """Patch the class `_load_embedder` imports, and record instances."""
    built: list[_CountingEmbedder] = []

    def _factory() -> _CountingEmbedder:
        instance = _CountingEmbedder()
        built.append(instance)
        return instance

    monkeypatch.setattr(
        "harness.retrieval.st_embedder.SentenceTransformersEmbedder", _factory, raising=True
    )
    return built


@pytest.fixture
def retrieval_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a checkout without the retrieval extra installed.

    Patching the module rather than `_load_embedder` keeps these tests
    independent of which module the helper currently lives in — they
    hold across the cli.py -> cli_store.py move.
    """
    monkeypatch.setitem(sys.modules, "harness.retrieval.st_embedder", None)


@pytest.fixture
def embedder_must_not_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to construct an embedder fails the test."""

    def _explode() -> object:
        raise AssertionError("embedder must not load on this path")

    monkeypatch.setitem(
        sys.modules,
        "harness.retrieval.st_embedder",
        SimpleNamespace(SentenceTransformersEmbedder=_explode),
    )


# --- _load_embedder ------------------------------------------------


def test_load_embedder_constructs_once_and_memoizes(
    fake_embedder_class: list[_CountingEmbedder],
) -> None:
    """Three callers, one model load — the whole point of the cache."""
    first = cli._load_embedder()
    second = cli._load_embedder()
    third = cli._load_embedder()

    assert first is second is third
    assert len(fake_embedder_class) == 1, "embedder was constructed more than once"


def test_load_embedder_returns_none_when_retrieval_extra_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing retrieval extra degrades to None plus one warning."""
    monkeypatch.setitem(sys.modules, "harness.retrieval.st_embedder", None)

    assert cli._load_embedder() is None
    assert "retrieval extra not installed" in capsys.readouterr().out


def test_load_embedder_caches_the_negative_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_embedder_class: list[_CountingEmbedder],
) -> None:
    """The import isn't retried once it has failed — and the user isn't
    warned a second time. `None` and "not yet loaded" are distinct
    states; that's what the sentinel is for."""
    monkeypatch.setitem(sys.modules, "harness.retrieval.st_embedder", None)
    assert cli._load_embedder() is None
    capsys.readouterr()

    # Make the module importable again: a live process that recovered
    # its retrieval extra still must not re-enter the import path.
    monkeypatch.setitem(
        sys.modules,
        "harness.retrieval.st_embedder",
        SimpleNamespace(SentenceTransformersEmbedder=_CountingEmbedder),
    )

    assert cli._load_embedder() is None
    assert capsys.readouterr().out == ""
    assert fake_embedder_class == []


# --- _maybe_retriever ----------------------------------------------


@pytest.mark.parametrize("top_k", [0, -1])
@pytest.mark.usefixtures("embedder_must_not_load")
def test_maybe_retriever_declines_without_loading_an_embedder(top_k: int) -> None:
    """top_k <= 0 short-circuits BEFORE the embedder is touched, so
    `--top-k 0` costs nothing."""
    character = load_character(_AIRTON)

    assert cli._maybe_retriever(character, top_k) is None


@pytest.mark.usefixtures("retrieval_extra_missing")
def test_maybe_retriever_returns_none_when_embedder_unavailable() -> None:
    """Caller falls back to full-set few-shot rather than failing."""
    character = load_character(_AIRTON)

    assert cli._maybe_retriever(character, 6) is None


def test_maybe_retriever_wires_the_shared_embedder(
    fake_embedder_class: list[_CountingEmbedder],
) -> None:
    character = load_character(_AIRTON)

    retriever = cli._maybe_retriever(character, 6)

    assert retriever is not None
    assert retriever.embedder is fake_embedder_class[0]
    assert retriever.character is character


# --- _open_episodic_store / _open_semantic_store --------------------


@pytest.mark.usefixtures("retrieval_extra_missing")
def test_open_episodic_store_returns_none_without_embedder() -> None:
    character = load_character(_AIRTON)

    assert cli._open_episodic_store(character, ingest=False) is None


def test_open_episodic_store_honors_db_path_override(
    tmp_path: Path, fake_embedder_class: list[_CountingEmbedder]
) -> None:
    """`--character` redirects memory subcommands to another silo
    (harness-5t53); the override must reach the store, not just the
    banner."""
    db_path = tmp_path / "elsewhere.db"
    character = load_character(_AIRTON)

    store = cli._open_episodic_store(character, ingest=False, db_path=db_path)

    assert store is not None
    assert db_path.exists()


def test_open_episodic_store_ingest_flag_controls_seeding(
    tmp_path: Path, fake_embedder_class: list[_CountingEmbedder]
) -> None:
    """ingest=False opens a store without seeding it; ingest=True seeds
    the character's seed memories and is idempotent on a second open."""
    character = load_character(_AIRTON)

    unseeded = cli._open_episodic_store(character, ingest=False, db_path=tmp_path / "cold.db")
    assert unseeded is not None
    assert unseeded.count() == 0

    seed_db = tmp_path / "seeded.db"
    seeded = cli._open_episodic_store(character, ingest=True, db_path=seed_db)
    assert seeded is not None
    first_count = seeded.count()
    assert first_count > 0

    again = cli._open_episodic_store(character, ingest=True, db_path=seed_db)
    assert again is not None
    assert again.count() == first_count


@pytest.mark.usefixtures("retrieval_extra_missing")
def test_open_semantic_store_returns_none_without_embedder() -> None:
    assert cli._open_semantic_store() is None


def test_open_semantic_store_honors_db_path_override(
    tmp_path: Path, fake_embedder_class: list[_CountingEmbedder]
) -> None:
    db_path = tmp_path / "facts.db"

    store = cli._open_semantic_store(db_path=db_path)

    assert store is not None
    assert db_path.exists()


def test_every_store_helper_shares_one_embedder(
    tmp_path: Path, fake_embedder_class: list[_CountingEmbedder]
) -> None:
    """Retriever, episodic store and semantic store all hold the SAME
    embedder instance. This is the contract that makes the ~1.3 GB model
    load once per process instead of three times."""
    character = load_character(_AIRTON)

    retriever = cli._maybe_retriever(character, 6)
    episodic = cli._open_episodic_store(character, ingest=False, db_path=tmp_path / "e.db")
    semantic = cli._open_semantic_store(db_path=tmp_path / "s.db")

    assert retriever is not None
    assert episodic is not None
    assert semantic is not None
    assert len(fake_embedder_class) == 1
    shared = fake_embedder_class[0]
    assert retriever.embedder is shared
    assert episodic.embedder is shared
    assert semantic.embedder is shared


def test_exactly_one_module_owns_the_embedder_cache() -> None:
    """Extraction landmine guard (docs/cli-extraction-plan.md, step 1).

    The cache is process-wide state. If the move to `cli_store.py`
    leaves a second `_cached_embedder` behind in `cli.py`, callers split
    across two caches and the model loads twice — with no test failure
    anywhere else to reveal it.
    """
    owners = sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in (_REPO_ROOT / "src" / "harness").rglob("*.py")
        if "_cached_embedder: object =" in path.read_text()
    )

    assert len(owners) == 1, f"the process-wide embedder cache is defined in {owners}"
