"""Embedder + store construction for the CLI.

Step 1 of docs/cli-extraction-plan.md. These four helpers are what every
other CLI surface needs before it can do anything with memory: the tool
registry, the memory subcommands, the eval handlers, and the sibling UIs
(`cli_classic`, `cli_repl`, `cli_tui`) all reach for them.

The embedder cache is process-wide ON PURPOSE. `cmd_chat` wires the
retriever, the episodic store and the semantic store from one instance
so the ~1.3 GB model loads once instead of three times. This module is
its single owner — a second `_cached_embedder` anywhere else silently
splits callers across two caches (guarded by
tests/test_cli_store_contract.py).
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.status import Status

from harness.character import Character
from harness.config import settings
from harness.retrieval import VoiceRetriever
from harness.store.episodic import EpisodicStore, ensure_seeds_ingested
from harness.store.semantic import SemanticStore

console = Console()

_EMBEDDER_SENTINEL: object = object()
_cached_embedder: object = _EMBEDDER_SENTINEL


def _load_embedder() -> object | None:
    """Lazy-import and memoize the default embedder. Returns None (with
    a warning) if the retrieval extra isn't installed.

    The result is cached process-wide — `cmd_chat` wires retriever +
    episodic store + semantic store from the same instance, so the 1.3
    GB embedder model loads once instead of three times."""
    global _cached_embedder
    if _cached_embedder is not _EMBEDDER_SENTINEL:
        return None if _cached_embedder is None else _cached_embedder
    try:
        from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    except ImportError:
        console.print(
            "[yellow]retrieval extra not installed. "
            "Run `uv sync --extra all` to enable retrieval + memory.[/yellow]"
        )
        _cached_embedder = None
        return None
    _cached_embedder = SentenceTransformersEmbedder()
    return _cached_embedder


def _maybe_retriever(character: Character, top_k: int) -> VoiceRetriever | None:
    """Build a VoiceRetriever if retrieval is requested and the optional
    sentence-transformers dep is installed. Returns None to signal the
    caller to fall back to full-set few-shot."""
    if top_k <= 0:
        return None
    embedder = _load_embedder()
    if embedder is None:
        return None
    with Status(f"warming embedder ({embedder.id})…", console=console):  # type: ignore[attr-defined]
        return VoiceRetriever(embedder=embedder, character=character)  # type: ignore[arg-type]


def _open_episodic_store(
    character: Character,
    *,
    ingest: bool = True,
    db_path: Path | None = None,
) -> EpisodicStore | None:
    """Open the episodic store, ingesting seeds on first run. Returns
    None if the retrieval extra isn't installed.

    `db_path` overrides the default `settings.character_db_path` — used
    by the `memory` subcommands when `--character` redirects the
    target store to a non-default silo (harness-5t53)."""
    embedder = _load_embedder()
    if embedder is None:
        return None
    path = db_path if db_path is not None else settings.character_db_path
    store = EpisodicStore(path, embedder=embedder)  # type: ignore[arg-type]
    if ingest:
        inserted = ensure_seeds_ingested(character, store)
        if inserted > 0:
            console.print(f"[dim]seeded {inserted} episodic memories from character.[/dim]")
    return store


def _open_semantic_store(*, db_path: Path | None = None) -> SemanticStore | None:
    """Open the semantic-facts store. `db_path` overrides the default
    `settings.character_db_path` for the same reason as
    `_open_episodic_store` (harness-5t53)."""
    embedder = _load_embedder()
    if embedder is None:
        return None
    path = db_path if db_path is not None else settings.character_db_path
    return SemanticStore(path, embedder=embedder)  # type: ignore[arg-type]
