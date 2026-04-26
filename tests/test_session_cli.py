from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.store.transcript import Transcript


class _StubEmbedder:
    """Minimal embedder fixture for tests that need to seed
    EpisodicStore / SemanticStore through their public ingest API
    but don't otherwise care about retrieval quality. Hash-based,
    deterministic, no model load."""

    id: str = "stub"
    dimension: int = 4

    def embed(self, texts):  # type: ignore[no-untyped-def]
        out: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.stack(out)


def _seed_db(db: Path) -> None:
    """Drop two recorded sessions into `db`."""
    db.parent.mkdir(parents=True, exist_ok=True)
    ts = Transcript(db)
    try:
        ts.append(session="alpha", channel="cli", speaker="mark", role="user", content="hi")
        ts.append(session="alpha", channel="cli", speaker="airton", role="assistant", content="hey")
        ts.append(session="beta", channel="cli", speaker="mark", role="user", content="other")
        ts.append(
            session="beta", channel="cli", speaker="airton", role="assistant", content="reply"
        )
    finally:
        ts.close()


def _redirect_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Replace `harness.cli.settings` with a fresh `Settings(root=tmp_path)`
    so any test-local sqlite reads land under tmp instead of the real
    repo data dir. Returns the resolved character db path."""
    from harness.config import Settings

    fresh = Settings(root=tmp_path)
    monkeypatch.setattr("harness.cli.settings", fresh)
    return fresh.character_db_path


def test_list_sessions_groups_and_orders(tmp_path: Path) -> None:
    """`Transcript.list_sessions()` returns one row per session,
    newest-last-activity first, with role-split turn counts. Backbone
    of `harness session list`."""
    db = tmp_path / "harness.sqlite"
    _seed_db(db)
    ts = Transcript(db)
    try:
        rows = ts.list_sessions()
    finally:
        ts.close()
    assert {r.session for r in rows} == {"alpha", "beta"}
    by_id = {r.session: r for r in rows}
    assert by_id["alpha"].total_rows == 2
    assert by_id["alpha"].user_turns == 1
    assert by_id["alpha"].assistant_turns == 1
    assert by_id["alpha"].channel == "cli"
    # Beta was appended last so it sorts first.
    assert rows[0].session == "beta"


def test_session_list_command_prints_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`harness session list` opens the character db and prints a
    table covering both seeded sessions."""
    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    runner = CliRunner()
    result = runner.invoke(app, ["session", "list"])
    assert result.exit_code == 0, result.output
    assert "alpha" in result.output
    assert "beta" in result.output
    # Header row labels are rendered.
    assert "turns" in result.output


def test_session_show_dumps_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`harness session show <id>` emits one markdown section per
    turn, in insertion order. Output is paste-friendly: no ANSI,
    role + speaker + timestamp on the heading line."""
    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    runner = CliRunner()
    result = runner.invoke(app, ["session", "show", "alpha"])
    assert result.exit_code == 0, result.output
    assert "### user (mark)" in result.output
    assert "### assistant (airton)" in result.output
    assert "hi" in result.output
    assert "hey" in result.output
    # Order is preserved.
    assert result.output.index("hi") < result.output.index("hey")


def test_session_show_defaults_to_most_recent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a session id, `session show` falls back to the
    most-recently-active session (`beta` per seed order)."""
    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    runner = CliRunner()
    result = runner.invoke(app, ["session", "show"])
    assert result.exit_code == 0, result.output
    assert "other" in result.output
    assert "reply" in result.output
    # `hi`/`hey` belong to alpha; the default fallback shouldn't pull them.
    assert "hi" not in result.output


def test_session_show_json_emits_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` switches to one-JSON-object-per-line so the dump
    can be piped into jq or another transcript loader."""
    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    runner = CliRunner()
    result = runner.invoke(app, ["session", "show", "alpha", "--json"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip().startswith("{")]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["content"] for p in parsed] == ["hi", "hey"]
    assert parsed[0]["role"] == "user"
    assert parsed[1]["role"] == "assistant"


def test_session_show_unknown_id_is_quiet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown session id exits 0 with a stderr hint and no body —
    nothing to dump, but not a hard error."""
    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    runner = CliRunner()
    result = runner.invoke(app, ["session", "show", "nope"])
    assert result.exit_code == 0, result.output
    # No turns means no `### ` markdown headings.
    assert "### " not in result.output


def test_coin_session_id_format() -> None:
    """`_coin_session_id` mints a daily-rotated, launch-unique id of
    the form `cli-YYYY-MM-DD-HHMMSS` (UTC). Pin the format because
    `harness session list` ordering and downstream regexes (eval +
    docs) rely on it. harness-y7ua."""
    from datetime import UTC, datetime

    from harness.cli import _coin_session_id

    fixed = datetime(2026, 4, 26, 14, 32, 5, tzinfo=UTC)
    assert _coin_session_id(now=fixed) == "cli-2026-04-26-143205"


def test_chat_default_session_is_unique_per_launch() -> None:
    """Two `_coin_session_id()` calls a second apart produce
    different ids, so two consecutive `harness chat` launches do
    not collide on the prior session's compaction summary."""
    import time

    from harness.cli import _coin_session_id

    first = _coin_session_id()
    time.sleep(1.01)
    second = _coin_session_id()
    assert first != second


def test_session_compact_reset_drops_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`harness session compact-reset <id> --yes` deletes the
    compaction summary so the next chat invocation falls through
    to the raw-transcript tail path. Transcript rows stay
    intact for audit (harness-xf8d)."""
    from harness.compaction import CompactionStore

    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)
    store = CompactionStore(db)
    try:
        store.append(
            session_id="alpha",
            summary="folded-old-threads",
            up_to_turn_id=2,
            covered_turns=4,
            model_id="echo",
        )
        assert store.latest_for_session("alpha") is not None
    finally:
        store.close()

    runner = CliRunner()
    result = runner.invoke(app, ["session", "compact-reset", "alpha", "--yes"])
    assert result.exit_code == 0, result.output
    assert "dropped" in result.output

    store2 = CompactionStore(db)
    try:
        assert store2.latest_for_session("alpha") is None
    finally:
        store2.close()

    # Transcript rows survive the reset — verify by re-running show.
    show = runner.invoke(app, ["session", "show", "alpha"])
    assert show.exit_code == 0, show.output
    assert "hi" in show.output
    assert "hey" in show.output


def test_session_reset_full_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`harness session reset <id> --yes` (harness-k7m9) must:
    1) drop the compaction summary,
    2) record a /clear watermark at the current tip,
    3) delete tier=working episodic + semantic rows tagged to that
       session,
    4) leave the transcript intact.

    Other sessions' rows + shared seeds + consolidated rows survive."""
    from harness.compaction import CompactionStore
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore

    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    # The CLI builds an embedder via _load_embedder() which would
    # try to download bge-small. Patch that off so the test stays
    # offline; the delete path doesn't need vectors.
    monkeypatch.setattr("harness.cli._load_embedder", lambda: None)

    # Seed memory + summary state for session "alpha".
    compaction = CompactionStore(db)
    compaction.append(
        session_id="alpha",
        summary="folded threads",
        up_to_turn_id=2,
        covered_turns=4,
        model_id="echo",
    )
    compaction.close()
    embedder = _StubEmbedder()
    episodic = EpisodicStore(db, embedder=embedder)
    semantic = SemanticStore(db, embedder=embedder)
    try:
        episodic.ingest(
            external_id=None,
            title="alpha-working-1",
            body="from alpha",
            tier="working",
            source="scribe",
            session_id="alpha",
        )
        episodic.ingest(
            external_id=None,
            title="beta-working-1",
            body="from beta",
            tier="working",
            source="scribe",
            session_id="beta",
        )
        episodic.ingest(
            external_id="seed-x",
            title="shared seed",
            body="seed body",
            tier="seed",
            source="yaml",
        )
        semantic.add(
            subject="mark",
            predicate="prefers",
            object="raw sql",
            confidence=0.8,
            source="scribe",
            tier="working",
            session_id="alpha",
        )
        semantic.add(
            subject="mark",
            predicate="lives_in",
            object="austin",
            confidence=0.9,
            source="scribe",
            tier="working",
            session_id="beta",
        )
    finally:
        episodic.close()
        semantic.close()

    runner = CliRunner()
    result = runner.invoke(app, ["session", "reset", "alpha", "--yes"])
    assert result.exit_code == 0, result.output
    assert "summary=1" in result.output
    assert "episodic_working=1" in result.output
    assert "semantic_working=1" in result.output

    # Post-reset state: alpha summary gone, watermark set, alpha
    # working rows gone, others intact.
    compaction2 = CompactionStore(db)
    try:
        assert compaction2.latest_for_session("alpha") is None
        assert compaction2.latest_clear_after_id("alpha") is not None
    finally:
        compaction2.close()
    episodic2 = EpisodicStore(db, embedder=embedder)
    semantic2 = SemanticStore(db, embedder=embedder)
    try:
        ep_titles = {r.title for r in episodic2.all()}
        assert "alpha-working-1" not in ep_titles
        assert "beta-working-1" in ep_titles
        assert "shared seed" in ep_titles
        sm_subjects = {(f.subject, f.object) for f in semantic2.all()}
        assert ("mark", "raw sql") not in sm_subjects
        assert ("mark", "austin") in sm_subjects
    finally:
        episodic2.close()
        semantic2.close()

    # Transcript untouched.
    show = runner.invoke(app, ["session", "show", "alpha"])
    assert show.exit_code == 0, show.output
    assert "hi" in show.output
    assert "hey" in show.output


def test_session_compact_reset_quiet_when_nothing_to_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a session has no compaction summary, the command exits 0
    with a stderr hint and skips the confirm prompt — `--yes`
    isn't even required for the no-op path."""
    db = _redirect_settings(monkeypatch, tmp_path)
    _seed_db(db)

    runner = CliRunner()
    result = runner.invoke(app, ["session", "compact-reset", "alpha"])
    assert result.exit_code == 0, result.output
    assert "no compaction summary" in result.output


def test_session_list_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`session list` against a fresh db prints a (no sessions) hint
    and exits cleanly."""
    db = _redirect_settings(monkeypatch, tmp_path)
    Transcript(db).close()  # touch the file so the cli opens cleanly

    runner = CliRunner()
    result = runner.invoke(app, ["session", "list"])
    assert result.exit_code == 0, result.output
    assert "no sessions" in result.output
