"""Thin wrapper around the `bd` (beads) CLI — the single chokepoint
for ab's data-plane operations.

bd is treated as a black-box subprocess. Every call runs with
`cwd=<ab_bd_dir>` so the adapter targets ab's isolated beads database
and never touches the dev database co-located with the harness repo.
One-time setup (bd init / bd bootstrap in the ab_bd_dir) is the user's
responsibility; the adapter verifies on first use and raises with a
clear message if the dir hasn't been initialized.

Scope is enforced on every create. ab supports exactly two scopes —
`professional` and `personal` — applied as a `scope:<value>` label on
the bead. The adapter refuses to create an item without a scope.

Custom types (`project`, `event`, `habit`) are registered once on first
use via `bd config set types.custom`, merging with any pre-existing
custom types the user has configured.

The implementation is split across domain mixins (harness-vhoj):

- `_bd_runner.py` — base state + `_run` + verify + ensure_custom_types
- `_bd_crud.py`   — create / close / reopen / delete / update /
                    list / search / ready / show / stale
- `_bd_focus.py`  — get_focus / set_focus / persist_to_focus
- `_bd_graph.py`  — dep_* / label_* / comment_* / find_duplicates
- `_bd_memory.py` — remember / memories / forget

BeadsAdapter composes them via MRO so the public API is identical.
"""

from __future__ import annotations

# These imports are unused in this module, but tests monkeypatch
# `harness.store.bd_adapter.shutil.which` and
# `harness.store.bd_adapter.subprocess.run` to stub the bd CLI at
# the adapter boundary. Python modules are singletons — patching an
# attr on the module-level name reaches into the actual shutil /
# subprocess module the mixin code uses too. Keep these imports for
# the test surface.
import shutil  # noqa: F401
import subprocess  # noqa: F401

from harness.store._bd_crud import _extract_created_id, _parse_issue_list
from harness.store._bd_focus import BeadsFocusMixin
from harness.store._bd_graph import BeadsGraphMixin
from harness.store._bd_memory import BeadsMemoryMixin
from harness.store._bd_runner import (
    AB_CUSTOM_TYPES,
    BeadsAdapterError,
    BeadsRunner,
    InflightCapExceededError,
    TurnCapExceededError,
)
from harness.store._bd_types import ALLOWED_SCOPES, ALLOWED_TYPES, BeadsIssue, _issue_from_json


class BeadsAdapter(
    BeadsFocusMixin,  # extends BeadsCrudMixin extends BeadsRunner
    BeadsGraphMixin,
    BeadsMemoryMixin,
):
    """Subprocess wrapper around the bd CLI. Construct once per session
    and pass to ab's tools; every method dispatches to bd with
    `cwd=bd_dir` so the isolation invariant holds without the caller
    having to think about it.

    See the module docstring for the mixin layout — this class is
    intentionally empty; behavior lives in the topical mixins."""


__all__ = [
    "AB_CUSTOM_TYPES",
    "ALLOWED_SCOPES",
    "ALLOWED_TYPES",
    "BeadsAdapter",
    "BeadsAdapterError",
    "BeadsIssue",
    "BeadsRunner",
    "InflightCapExceededError",
    "TurnCapExceededError",
    "_extract_created_id",
    "_issue_from_json",
    "_parse_issue_list",
]
