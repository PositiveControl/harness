"""Shared factories for stateful orchestrator hooks (harness-lefw).

`WriteFileRedirectHook` carries three closures — `read_existing`,
`ensure_edit_file_active`, `invoke_edit_file` — that bridge the
pure-data hook to the session's live workspace + registry. The wiring
is identical between the chat REPL (cli_classic.py) and the driver's
executor turns (driver/loop.py); pulling it here avoids drift between
the two call sites.

A single function lives here today; future per-session hook factories
(e.g. an AutoLoadOnUnknownHook factory) should join it rather than
sprout new modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.orchestrator.hooks import WriteFileRedirectHook
    from harness.tools.base import ToolRegistry, ToolResult


def make_write_file_redirect_hook(
    *,
    registry: ToolRegistry | None,
    workspace_path: Path | None,
    targeted_fix: bool = False,
) -> WriteFileRedirectHook | None:
    """Wire a WriteFileRedirectHook against a live registry + workspace.

    Returns None when either input is None (tool-less chat sessions, or
    callers that haven't materialized a workspace yet). The hook itself
    is constructed regardless of whether write_file is currently in the
    registry — harness-hf4r: sessions with `--tool-set minimal` start
    without write_file, then load it lazily; an eager 'not in registry'
    short-circuit would skip those lazy-load turns. The hook's own
    `check()` gates on `ctx.call.name == "write_file"` so the always-
    constructed hook is a no-op cost on every non-write_file call.

    Three closures:

      - `read_existing(path)` reads `<workspace>/<path>` as UTF-8 text
        and returns the content. Non-existent files and decode failures
        yield None so the hook treats them as 'not redirectable' and
        Continues. Reject paths that escape the workspace root.
      - `ensure_edit_file_active()` adds edit_file to the active set
        when it's registered-but-inactive (the load_tool companion
        pairing usually means it's already active alongside
        write_file). Returns True iff edit_file is callable after the
        call. Returns False (causing the hook to fall through to
        Continue) when edit_file isn't even in the catalog.
      - `invoke_edit_file(path, old, new)` dispatches the registry's
        edit_file tool and returns its ToolResult so the hook can feed
        it back as the Skip payload.
    """
    if registry is None or workspace_path is None:
        return None

    from harness.orchestrator.hooks import WriteFileRedirectHook

    root = workspace_path.resolve()

    def read_existing(path: str) -> str | None:
        try:
            target = (root / path).resolve()
            target.relative_to(root)
        except (OSError, ValueError):
            return None
        if not target.exists() or not target.is_file():
            return None
        try:
            return target.read_text()
        except (OSError, UnicodeDecodeError):
            return None

    def ensure_edit_file_active() -> bool:
        if "edit_file" not in registry:
            return False
        if "edit_file" in registry.active_names():
            return True
        try:
            registry.set_active(set(registry.active_names()) | {"edit_file"})
        except Exception:
            return False
        return "edit_file" in registry.active_names()

    def invoke_edit_file(path: str, old: str, new: str) -> ToolResult:
        return registry.call(
            "edit_file",
            {"path": path, "old_string": old, "new_string": new},
        )

    return WriteFileRedirectHook(
        read_existing=read_existing,
        ensure_edit_file_active=ensure_edit_file_active,
        invoke_edit_file=invoke_edit_file,
        targeted_fix=targeted_fix,
    )


__all__ = ["make_write_file_redirect_hook"]
