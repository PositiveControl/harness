"""Unit tests for the CLI confirm + stream-filter helpers added as part
of harness-di7. Exercises the pre-validation, the call-description, and
the sentence-level stream filter without spinning up a whole chat loop."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from harness.cli import (
    _describe_call,
    _pre_validate_write_call,
    _StreamRenderer,
)
from harness.tools.base import ToolCall

# ---------- _pre_validate_write_call ----------


def test_pre_validate_passes_on_create(tmp_path: Path) -> None:
    call = ToolCall(
        name="write_file",
        arguments={"path": "new.py", "content": "print('x')\n"},
    )
    assert _pre_validate_write_call(call, tmp_path) is None


def test_pre_validate_refuses_shrink_clobber(tmp_path: Path) -> None:
    target = tmp_path / ".gitignore"
    target.write_text(".venv/\n.mypy_cache/\ndist/\nbuild/\n.env\n" * 5)
    call = ToolCall(
        name="write_file",
        arguments={
            "path": ".gitignore",
            "content": "scratch\n",
            "overwrite": True,
        },
    )
    msg = _pre_validate_write_call(call, tmp_path)
    assert msg is not None
    assert "append disguised as overwrite" in msg
    assert "edit_file" in msg


def test_pre_validate_allows_legitimate_small_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    target.write_text("old = 1\n")
    call = ToolCall(
        name="write_file",
        arguments={
            "path": "config.toml",
            "content": "new = 2\nother = 3\n",
            "overwrite": True,
        },
    )
    assert _pre_validate_write_call(call, tmp_path) is None


def test_pre_validate_noop_for_edit_file(tmp_path: Path) -> None:
    call = ToolCall(
        name="edit_file",
        arguments={
            "path": ".gitignore",
            "old_string": "",
            "new_string": "scratch\n",
        },
    )
    assert _pre_validate_write_call(call, tmp_path) is None


def test_pre_validate_noop_for_non_overwrite(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("existing\n" * 50)
    call = ToolCall(
        name="write_file",
        arguments={"path": "x.txt", "content": "hi\n"},
    )
    # overwrite missing — the tool itself will refuse on 'already exists';
    # no need for the pre-validator to second-guess here.
    assert _pre_validate_write_call(call, tmp_path) is None


# ---------- _describe_call ----------


def test_describe_create_new_file(tmp_path: Path) -> None:
    call = ToolCall(
        name="write_file",
        arguments={"path": "new.py", "content": "print('x')\n"},
    )
    assert _describe_call(call, tmp_path) == "create new.py (11B)"


def test_describe_overwrite_shows_delta(tmp_path: Path) -> None:
    target = tmp_path / "old.txt"
    target.write_text("a" * 200)
    call = ToolCall(
        name="write_file",
        arguments={"path": "old.txt", "content": "b" * 50, "overwrite": True},
    )
    out = _describe_call(call, tmp_path)
    assert "overwrite old.txt" in out
    assert "200B" in out
    assert "50B" in out


def test_describe_edit_append(tmp_path: Path) -> None:
    call = ToolCall(
        name="edit_file",
        arguments={
            "path": ".gitignore",
            "old_string": "",
            "new_string": "scratch\n",
        },
    )
    assert _describe_call(call, tmp_path) == "append 8B to .gitignore"


def test_describe_edit_replace(tmp_path: Path) -> None:
    call = ToolCall(
        name="edit_file",
        arguments={
            "path": "README.md",
            "old_string": "# Title",
            "new_string": "# New Title",
        },
    )
    out = _describe_call(call, tmp_path)
    assert "edit README.md" in out
    assert "1 match" in out


def test_describe_shell_truncates(tmp_path: Path) -> None:
    long = "a" * 200
    call = ToolCall(name="shell", arguments={"cmd": long})
    out = _describe_call(call, tmp_path)
    assert out.startswith("run: ")
    assert "..." in out
    assert len(out) < len(long) + 20


def test_describe_remember_fact(tmp_path: Path) -> None:
    call = ToolCall(
        name="remember_fact",
        arguments={"subject": "mark", "predicate": "uses", "object": "harness"},
    )
    assert _describe_call(call, tmp_path) == "mark uses harness"


# ---------- _StreamRenderer sentence filter ----------


def _render(deltas: list[str]) -> tuple[str, str]:
    """Run deltas through the renderer; return (visible_output, captured_stdout)."""
    import io

    out = io.StringIO()
    console = Console(file=out, force_terminal=False, color_system=None, width=200)
    renderer = _StreamRenderer(console)
    renderer.start()
    for d in deltas:
        renderer.append(d)
    visible = renderer.stop()
    return visible, out.getvalue()


def test_renderer_emits_plain_prose() -> None:
    visible, captured = _render(["The ", "weather ", "is ", "fine. "])
    assert "weather is fine" in visible
    assert "weather is fine" in captured


def test_renderer_drops_meta_confirm_sentence() -> None:
    visible, captured = _render(
        [
            "Fine. ",  # keep
            "Would you like me to proceed with the change? ",  # drop
            "Anything else? ",  # keep (no meta-confirm pattern)
        ]
    )
    assert "Fine." in visible
    assert "Would you like" not in captured
    assert "suppressed 1 line" in captured


def test_renderer_drops_false_success_sentence() -> None:
    visible, captured = _render(
        [
            "I'll investigate. ",  # keep
            "The scratch directory has been added to .gitignore. ",  # drop
        ]
    )
    assert "investigate" in visible
    assert "has been added" not in captured
    assert "suppressed" in captured


def test_renderer_flushes_trailing_non_terminated_fragment() -> None:
    visible, captured = _render(["trailing with no period"])
    assert "trailing with no period" in visible
    assert "trailing with no period" in captured


def test_renderer_suppresses_trailing_meta_confirm_fragment() -> None:
    visible, captured = _render(["Please confirm your approval"])
    assert "Please confirm" not in visible
    assert "Please confirm" not in captured
    assert "suppressed" in captured


def test_renderer_force_flushes_runaway_paragraph() -> None:
    """Regression: the 7B went into a degenerate loop that didn't emit
    sentence terminators. Previously the renderer would buffer silently
    until max_tokens fired, looking 'stuck' to the user. Now we force-
    flush at _MAX_BUFFER chars so the tokens hit the screen."""
    # A single 'sentence' of 500 chars with no terminator. Exceeds the
    # 400-char buffer cap — must render rather than stay hidden.
    junk = "alpha beta " * 50  # ~550 chars, no period
    visible, captured = _render([junk])
    assert junk.rstrip() in visible or "alpha beta" in captured


def test_renderer_force_flush_still_drops_meta_confirm() -> None:
    """The force-flush path must still apply the meta-confirm regex —
    otherwise a runaway 'would you like me to would you like me to…'
    loop would dump the whole mess to the user."""
    runaway = "would you like me to proceed " * 20  # ~580 chars, no period
    _, captured = _render([runaway])
    assert "would you like" not in captured.lower()
    assert "suppressed" in captured


def test_renderer_handles_token_level_fragmentation() -> None:
    """Tokens often arrive sub-word. A meta-confirm phrase split across
    several tiny deltas must still be dropped as a unit at the sentence
    boundary."""
    deltas = [
        "Wo",
        "uld ",
        "you ",
        "like ",
        "me ",
        "to ",
        "pro",
        "ceed",
        "? ",
        "Ready.",
    ]
    visible, captured = _render(deltas)
    assert "Would you like" not in captured
    assert "Ready" in visible
