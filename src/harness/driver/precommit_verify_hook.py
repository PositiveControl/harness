"""Pre-close verify gate (harness-nlj7).

A pre-tool hook that intercepts ``bd close <current_issue>`` shell calls
and runs the workspace verify steps BEFORE the close lands. On verify
fail, the close is Skipped with a synthesized failure ToolResult — the
model never sees a successful close ack, so the wrap_up_forced
narration spiral (loop_run=3e295564 turn 20) never starts.

The post-close verify in `loop.py` is kept as defense in depth — it
catches workspace degradation between pre-verify and bd close, or any
bd close path that bypasses this hook.

Matches ``bd close <current_id>`` as the first or trailing token, with
optional ``cd <path> &&`` prefix and optional ``--reason=...`` /
``--suggest-next`` flags. ``bd close <other_id>`` (closing ancillary
beads) passes through untouched — only the focal issue is gated.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harness.orchestrator.hooks import Continue, PreToolContext, PreToolOutcome, Skip
from harness.tools.base import ToolResult

if TYPE_CHECKING:
    from harness.driver.bd import DriverBd
    from harness.driver.planner import VerifyStep

VerifyFn = Callable[[], tuple[str | None, int]]


def _verify_noop() -> tuple[str | None, int]:
    """Default verify closure for the dataclass — a no-op that never
    blocks. Callers must replace via `make_pre_close_verify_hook`."""
    return None, 0


# Strip one leading ``cd <path> &&`` wrapper (the executor habitually
# prefixes one; see harness-jmkc / ShellEchoNoopHook for the same
# treatment). Match is non-greedy on the path so a chained ``cd a && cd
# b && bd close`` only loses one level — the inner cmd is then re-checked.
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+\S+\s*&&\s*(.+)$", re.DOTALL)


def _strip_cd_prefix(cmd: str) -> str:
    """Remove one leading ``cd <path> &&`` wrapper from `cmd`."""
    m = _CD_PREFIX_RE.match(cmd)
    return m.group(1).strip() if m else cmd.strip()


def _is_close_of_issue(cmd: str, issue_id: str) -> bool:
    """True iff `cmd` is a ``bd close`` shell call whose target list
    contains `issue_id`. Accepts the basic form and the common flag
    variants (``--reason=...``, ``--suggest-next``, ``--force``).

    Multi-id closes (``bd close a b c``) match if `issue_id` is in the
    list — closing the focal issue alongside scratch is still gated."""
    stripped = _strip_cd_prefix(cmd)
    tokens = stripped.split()
    if len(tokens) < 3:
        return False
    if tokens[0] != "bd" or tokens[1] != "close":
        return False
    return issue_id in tokens[2:]


@dataclass
class PreCloseVerifyHook:
    """Pre-tool hook that runs workspace verify BEFORE ``bd close
    <current_issue>`` is allowed to execute. On verify pass: Continue
    (bd close runs normally). On verify fail: Skip with a synthesized
    failed ToolResult — the model sees an explicit verify_blocked
    failure instead of a close-success ack."""

    name: str = "pre_close_verify"
    _verify: VerifyFn = field(default=_verify_noop)
    _current_issue_id: str = ""

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if ctx.call.name != "shell":
            return Continue()
        cmd = ctx.call.arguments.get("cmd")
        if not isinstance(cmd, str):
            return Continue()
        if not _is_close_of_issue(cmd, self._current_issue_id):
            return Continue()
        failure, _ = self._verify()
        if failure is None:
            return Continue()
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=(
                    f"[VERIFY_BLOCKED] bd close was NOT executed: "
                    f"workspace verify failed first.\n{failure}\n"
                    f"Fix the workspace, then retry the close."
                ),
                success=False,
                error="verify_blocked",
            )
        )


def make_pre_close_verify_hook(
    *,
    verify_map: Mapping[str, Sequence[VerifyStep]],
    bd: DriverBd,
    current_issue_id: str,
    workspace: Path,
    default_steps: Sequence[VerifyStep],
    regression_snapshot: Path | None,
) -> PreCloseVerifyHook:
    """Wire a PreCloseVerifyHook against the driver's verify config and
    bd handle. Returns the hook ready to install in the pre_tool phase.

    The closure binds the per-turn verify config — current_issue_id and
    regression_snapshot change between turns, so the factory must run
    inside `_run_executor_turn`, not at run startup.
    """
    # Local import — `_run_issue_verify` is the existing verify driver
    # and lives in loop.py. The hook is the seam that lets the same
    # function run BEFORE bd close instead of only after.
    from harness.driver.loop import _run_issue_verify

    def verify() -> tuple[str | None, int]:
        return _run_issue_verify(
            verify_map,
            bd,
            current_issue_id,
            workspace,
            default_steps=default_steps,
            regression_snapshot=regression_snapshot,
        )

    return PreCloseVerifyHook(_verify=verify, _current_issue_id=current_issue_id)


__all__ = ["PreCloseVerifyHook", "make_pre_close_verify_hook"]
