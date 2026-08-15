"""Behavioral contract for the `harness memory` and `harness voice` commands.

Step 5b of docs/cli-extraction-plan.md moves these handlers into
cli_memory.py / cli_voice.py. They were the last group in cli.py that
MUTATES STORES with no CLI-level test at all: 509 LOC of memory
handlers at 6% coverage, writing episodic rows, semantic facts, scribe
watermarks — and `wipe`, which deletes all three.

Everything here drives the real Typer app through CliRunner against a
tmp-path store, in the style of tests/test_session_cli.py (which is why
`session` sits at 81% coverage while `memory` sat at 6%).

Two redirections make that safe and offline:
  * `settings.root` is mutated on the shared Settings instance, so every
    module that did `from harness.config import settings` sees the tmp
    root — including the handler modules after they move.
  * `_load_embedder` is stubbed at its owning module so no model loads.
Both are resolved dynamically, so these tests are unchanged by the move.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import app
from harness.config import settings

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _StubEmbedder:
    """Deterministic 4-dim embedder. No model load, stable vectors."""

    id: str = "stub"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.stack(out) if out else np.zeros((0, 4), dtype=np.float32)


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at a tmp root holding a COPY of the airton
    character, plus a stub embedder. Returns the character db path.

    A copy, not a symlink: `voice capture` writes into
    `character/<name>/voice/captured.yaml`, and a symlink would put test
    samples in the real corpus."""
    shutil.copytree(_REPO_ROOT / "character" / "airton", tmp_path / "character" / "airton")
    monkeypatch.setattr(settings, "root", tmp_path)

    from harness.cli import _load_embedder

    owner = sys.modules[_load_embedder.__module__]
    monkeypatch.setattr(owner, "_cached_embedder", _StubEmbedder())

    return settings.character_db_path


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


# ---------- read paths ----------


def test_memory_list_seeds_the_store_on_first_open(cli_env: Path) -> None:
    """Surprising but real: `memory list` is not read-only. It opens the
    episodic store with ingest=True, so a first run against an empty db
    ingests the character's seed memories and then lists them. Pinned
    because the extraction must not quietly change it — and because
    anyone reading the handler expects a plain read."""
    result = _run("memory", "list")

    assert result.exit_code == 0, result.output
    assert "seeded" in result.output
    assert "Episodic memory" in result.output


def test_memory_ingest_then_list_shows_the_seeded_records(cli_env: Path) -> None:
    ingest = _run("memory", "ingest")
    assert ingest.exit_code == 0, ingest.output

    listed = _run("memory", "list")

    assert listed.exit_code == 0, listed.output
    assert "Episodic memory" in listed.output
    assert "seed" in listed.output


def test_memory_ingest_is_idempotent(cli_env: Path) -> None:
    """Second pass inserts nothing new — external_id dedup. The command
    reports 0 new against an unchanged total."""
    first = _run("memory", "ingest")
    second = _run("memory", "ingest")

    assert second.exit_code == 0, second.output
    assert "0" in second.output
    total_first = first.output.split("new,")[1]
    total_second = second.output.split("new,")[1]
    assert total_first == total_second


def test_memory_list_filters_by_tier(cli_env: Path) -> None:
    _run("memory", "ingest")

    seeds = _run("memory", "list", "--tier", "seed")
    working = _run("memory", "list", "--tier", "working")

    assert "Episodic memory" in seeds.output
    # Nothing has been scribed, so the working tier is genuinely empty.
    assert "(empty)" in working.output


def test_memory_search_also_seeds_before_searching(cli_env: Path) -> None:
    """Same ingest=True path as `list`. A search against a fresh db
    therefore searches the seeds, never an empty store."""
    result = _run("memory", "search", "anything")

    assert result.exit_code == 0, result.output
    assert "seeded" in result.output


def test_memory_search_returns_scored_hits(cli_env: Path) -> None:
    _run("memory", "ingest")

    result = _run("memory", "search", "tests", "--k", "2")

    assert result.exit_code == 0, result.output
    assert "no matches" not in result.output


def test_memory_target_header_names_the_character_and_store(cli_env: Path) -> None:
    """harness-5t53: the implicit default must never be silent — every
    memory command prints which character's store it opened."""
    result = _run("memory", "list")

    assert "(memory: airton store at" in result.output


def test_unknown_character_is_a_parameter_error(cli_env: Path) -> None:
    result = _run("memory", "list", "--character", "nope")

    assert result.exit_code != 0
    assert "not found" in result.output


# ---------- fact write path ----------


def test_fact_add_writes_a_triple_and_echoes_the_id(cli_env: Path) -> None:
    result = _run("memory", "fact-add", "mark", "uses", "harness")

    assert result.exit_code == 0, result.output
    assert "added" in result.output
    assert "mark uses harness" in result.output

    listed = _run("memory", "fact-list")
    assert "mark" in listed.output
    assert "harness" in listed.output


def test_fact_add_defaults_to_a_shared_fact(cli_env: Path) -> None:
    """No --user means user_id IS NULL — visible to everyone. Getting
    this backwards silos a fact meant to be shared."""
    _run("memory", "fact-add", "airton", "runs-on", "mlx")

    from harness.store.semantic import SemanticStore

    store = SemanticStore(cli_env, embedder=_StubEmbedder())
    try:
        (fact,) = store.all()
        assert fact.user_id is None
    finally:
        store.close()


def test_fact_add_scopes_to_a_user_when_asked(cli_env: Path) -> None:
    _run("memory", "fact-add", "mark", "prefers", "terse", "--user", "mark")

    from harness.store.semantic import SemanticStore

    store = SemanticStore(cli_env, embedder=_StubEmbedder())
    try:
        (fact,) = store.all()
        assert fact.user_id == "mark"
    finally:
        store.close()


def test_fact_add_threads_confidence_tier_and_source(cli_env: Path) -> None:
    _run(
        "memory",
        "fact-add",
        "airton",
        "likes",
        "tdd",
        "--confidence",
        "0.42",
        "--tier",
        "consolidated",
        "--source",
        "eval",
    )

    from harness.store.semantic import SemanticStore

    store = SemanticStore(cli_env, embedder=_StubEmbedder())
    try:
        (fact,) = store.all()
        assert fact.confidence == pytest.approx(0.42)
        assert fact.tier == "consolidated"
        assert fact.source == "eval"
    finally:
        store.close()


def test_fact_add_parses_temporal_validity_flags(cli_env: Path) -> None:
    """--valid-from / --valid-to / --asserted-at accept plain dates and
    land as real datetimes, which is what `search(as_of=...)` filters on."""
    _run(
        "memory",
        "fact-add",
        "mark",
        "lived-in",
        "berlin",
        "--valid-from",
        "2019-01-01",
        "--valid-to",
        "2021-06-30",
        "--asserted-at",
        "2024-03-15",
    )

    from harness.store.semantic import SemanticStore

    store = SemanticStore(cli_env, embedder=_StubEmbedder())
    try:
        (fact,) = store.all()
        assert fact.valid_from is not None
        assert fact.valid_from.year == 2019
        assert fact.valid_to is not None
        assert fact.valid_to.year == 2021
        assert fact.asserted_at is not None
        assert fact.asserted_at.year == 2024
    finally:
        store.close()


def test_fact_list_filters_by_subject(cli_env: Path) -> None:
    _run("memory", "fact-add", "mark", "uses", "harness")
    _run("memory", "fact-add", "airton", "runs-on", "mlx")

    only_mark = _run("memory", "fact-list", "--subject", "mark")

    assert "harness" in only_mark.output
    assert "mlx" not in only_mark.output


def test_fact_search_reports_empty_store(cli_env: Path) -> None:
    result = _run("memory", "fact-search", "anything")

    assert result.exit_code == 0, result.output


# ---------- destructive paths ----------


def test_wipe_requires_confirmation_and_aborts_without_it(cli_env: Path) -> None:
    """No --yes and a declined prompt must leave the data alone. This is
    the only command in the group that destroys rows."""
    _run("memory", "ingest")
    _run("memory", "fact-add", "mark", "uses", "harness")

    result = CliRunner().invoke(app, ["memory", "wipe"], input="n\n")

    assert result.exit_code != 0  # abort
    assert "(empty)" not in _run("memory", "list").output
    assert "mark" in _run("memory", "fact-list").output


def test_wipe_with_yes_clears_episodic_and_semantic(cli_env: Path) -> None:
    _run("memory", "ingest")
    _run("memory", "fact-add", "mark", "uses", "harness")

    result = _run("memory", "wipe", "--yes")

    assert result.exit_code == 0, result.output
    assert "wiped" in result.output

    # Read the stores directly: `memory list` would re-seed episodic on
    # open (see test_memory_list_seeds_the_store_on_first_open), which
    # would mask the delete. fact-list has no such side effect.
    from harness.store.episodic import EpisodicStore

    episodic = EpisodicStore(cli_env, embedder=_StubEmbedder())
    try:
        assert episodic.all() == []
    finally:
        episodic.close()
    assert "(empty)" in _run("memory", "fact-list").output


def test_wipe_preserves_transcripts(cli_env: Path) -> None:
    """Documented contract: wipe clears memory, not the transcript."""
    from harness.store.transcript import Transcript

    transcript = Transcript(cli_env)
    try:
        transcript.append(
            session="alpha", channel="cli", speaker="mark", role="user", content="hello"
        )
    finally:
        transcript.close()

    _run("memory", "wipe", "--yes")

    after = Transcript(cli_env)
    try:
        assert after.list_sessions()
    finally:
        after.close()


# ---------- rebuild + consolidate ----------


def test_rebuild_embeddings_reports_mismatches_and_rebuilds(cli_env: Path) -> None:
    _run("memory", "ingest")

    result = _run("memory", "rebuild-embeddings")

    assert result.exit_code == 0, result.output
    assert "mismatched" in result.output
    assert "rebuilt" in result.output
    assert "stub" in result.output


def test_consolidate_runs_on_an_empty_store(cli_env: Path) -> None:
    """Nothing to merge is a normal outcome, not an error."""
    result = _run("memory", "consolidate")

    assert result.exit_code == 0, result.output


# ---------- voice ----------


def test_voice_list_captured_handles_an_empty_corpus(cli_env: Path) -> None:
    result = _run("voice", "list-captured")

    assert result.exit_code == 0, result.output


# ---------- scribe ----------


def test_scribe_walks_unprocessed_turns_with_the_echo_adapter(cli_env: Path) -> None:
    """The echo adapter can't produce well-formed extractions, so this
    exercises the whole scribe path AND its parse-error reporting —
    which is the branch that must not raise."""
    from harness.store.transcript import Transcript

    transcript = Transcript(cli_env)
    try:
        transcript.append(
            session="alpha", channel="cli", speaker="mark", role="user", content="I use MLX daily"
        )
        transcript.append(
            session="alpha",
            channel="cli",
            speaker="airton",
            role="assistant",
            content="noted",
        )
    finally:
        transcript.close()

    result = _run("memory", "scribe", "--session", "alpha", "--model", "echo")

    assert result.exit_code == 0, result.output
    assert "processed" in result.output


def test_scribe_on_an_unknown_session_processes_nothing(cli_env: Path) -> None:
    result = _run("memory", "scribe", "--session", "nope", "--model", "echo")

    assert result.exit_code == 0, result.output
    assert "processed 0 turns" in result.output


# ---------- harvest (no bd dir under the tmp root) ----------


@pytest.mark.parametrize("command", ["harvest-skills", "harvest-memories"])
def test_harvest_exits_nonzero_when_bd_is_unavailable(cli_env: Path, command: str) -> None:
    """Note the asymmetry with the session-start harvest, which warns and
    continues (a flaky bd must not block opening chat). The explicit
    `memory harvest-*` commands instead exit 1, because the user asked
    for a harvest and didn't get one. Pinned, not judged."""
    result = _run("memory", command)

    assert result.exit_code == 1, result.output


# ---------- voice capture ----------


def test_voice_capture_appends_a_sample_and_list_shows_it(cli_env: Path) -> None:
    from harness.store.transcript import Transcript

    transcript = Transcript(cli_env)
    try:
        transcript.append(
            session="alpha",
            channel="cli",
            speaker="mark",
            role="user",
            content="what do you think of the rewrite?",
        )
        transcript.append(
            session="alpha", channel="cli", speaker="airton", role="assistant", content="It's fine."
        )
    finally:
        transcript.close()

    captured = _run(
        "voice", "capture", "--session", "alpha", "--gold", "Ship it.", "--id", "cap-test"
    )

    assert captured.exit_code == 0, captured.output

    import yaml

    doc = yaml.safe_load((settings.character_path / "voice" / "captured.yaml").read_text())
    sample = next(s for s in doc["samples"] if s["id"] == "cap-test")
    assert sample["gold"] == "Ship it."
    assert sample["prompt"] == "what do you think of the rewrite?"
    assert sample["captured_from"] == "session=alpha"

    listed = _run("voice", "list-captured")
    assert "cap-test" in listed.output


def test_voice_capture_prompt_override_wins_over_the_transcript(cli_env: Path) -> None:
    from harness.store.transcript import Transcript

    transcript = Transcript(cli_env)
    try:
        transcript.append(
            session="alpha", channel="cli", speaker="mark", role="user", content="from transcript"
        )
    finally:
        transcript.close()

    _run(
        "voice",
        "capture",
        "--session",
        "alpha",
        "--gold",
        "reply",
        "--prompt",
        "explicit prompt",
        "--id",
        "cap-override",
    )

    import yaml

    doc = yaml.safe_load((settings.character_path / "voice" / "captured.yaml").read_text())
    sample = next(s for s in doc["samples"] if s["id"] == "cap-override")
    assert sample["prompt"] == "explicit prompt"
