"""Tests for the pre-close verify hook (harness-nlj7)."""

from __future__ import annotations

from harness.driver.precommit_verify_hook import (
    PreCloseVerifyHook,
    _is_close_of_issue,
    _strip_cd_prefix,
)
from harness.orchestrator.hooks import Continue, PreToolContext, Skip
from harness.tools.base import ToolCall


def _ctx(cmd: str) -> PreToolContext:
    return PreToolContext(
        call=ToolCall(name="shell", arguments={"cmd": cmd}),
        seen_calls={},
    )


def test_strip_cd_prefix_removes_one_level() -> None:
    assert _strip_cd_prefix("cd /tmp && bd close harness-a") == "bd close harness-a"


def test_strip_cd_prefix_leaves_bare_command() -> None:
    assert _strip_cd_prefix("bd close harness-a") == "bd close harness-a"


def test_is_close_matches_basic() -> None:
    assert _is_close_of_issue("bd close harness-a", "harness-a")


def test_is_close_matches_with_cd_prefix() -> None:
    assert _is_close_of_issue("cd /tmp && bd close harness-a", "harness-a")


def test_is_close_matches_with_flags() -> None:
    assert _is_close_of_issue("bd close harness-a --reason=done", "harness-a")
    assert _is_close_of_issue("bd close harness-a --suggest-next", "harness-a")


def test_is_close_matches_multi_id_when_focal_present() -> None:
    assert _is_close_of_issue("bd close harness-a harness-b", "harness-b")


def test_is_close_does_not_match_other_issue() -> None:
    assert not _is_close_of_issue("bd close harness-other", "harness-a")


def test_is_close_does_not_match_non_close() -> None:
    assert not _is_close_of_issue("bd show harness-a", "harness-a")
    assert not _is_close_of_issue("bd ready", "harness-a")
    assert not _is_close_of_issue("ls -la", "harness-a")


def test_hook_passes_through_non_shell_calls() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: ("verify failed", 1),
        _current_issue_id="harness-a",
    )
    ctx = PreToolContext(
        call=ToolCall(name="read_file", arguments={"path": "foo.js"}),
        seen_calls={},
    )
    assert isinstance(hook.check(ctx), Continue)


def test_hook_passes_through_non_close_shell_calls() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: ("verify failed", 1),
        _current_issue_id="harness-a",
    )
    assert isinstance(hook.check(_ctx("ls -la")), Continue)


def test_hook_passes_through_close_of_other_issue() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: ("verify failed", 1),
        _current_issue_id="harness-a",
    )
    assert isinstance(hook.check(_ctx("bd close harness-other")), Continue)


def test_hook_continues_when_verify_passes() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: (None, 2),
        _current_issue_id="harness-a",
    )
    assert isinstance(hook.check(_ctx("bd close harness-a")), Continue)


def test_hook_skips_with_verify_blocked_when_verify_fails() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: ("syntax error on game.js:42", 1),
        _current_issue_id="harness-a",
    )
    outcome = hook.check(_ctx("bd close harness-a"))
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "verify_blocked"
    assert "VERIFY_BLOCKED" in outcome.result.output
    assert "syntax error on game.js:42" in outcome.result.output


def test_hook_skips_when_close_has_flags_and_verify_fails() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: ("verify failed", 1),
        _current_issue_id="harness-a",
    )
    outcome = hook.check(_ctx("bd close harness-a --reason=done"))
    assert isinstance(outcome, Skip)


def test_hook_handles_non_string_cmd_argument() -> None:
    hook = PreCloseVerifyHook(
        _verify=lambda: ("fail", 1),
        _current_issue_id="harness-a",
    )
    ctx = PreToolContext(
        call=ToolCall(name="shell", arguments={"cmd": 123}),
        seen_calls={},
    )
    assert isinstance(hook.check(ctx), Continue)
