from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.store.transcript import Transcript


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


def test_session_list_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`session list` against a fresh db prints a (no sessions) hint
    and exits cleanly."""
    db = _redirect_settings(monkeypatch, tmp_path)
    Transcript(db).close()  # touch the file so the cli opens cleanly

    runner = CliRunner()
    result = runner.invoke(app, ["session", "list"])
    assert result.exit_code == 0, result.output
    assert "no sessions" in result.output
