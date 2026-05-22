"""Post-write syntax gate for file-mutating tools — harness-h6wa.

Surfaced by loop run 26c39558: a single `edit_file` call landed a
JavaScript file with two `const tileX` declarations in the same `for`
loop block scope, which is a parse-time SyntaxError. The harness had
no defense — `edit_file` just spliced strings and reported byte counts.
Six subsequent beads closed on a DOA artifact before §15 surfaced a
symptom (input listeners "missing" — they could not have worked since
the script never parsed).

This module supplies an extension-dispatched parser invocation that
`EditFileTool` and `WriteFileTool` run after every successful write.
A failed parse turns the tool result from success → failure with the
parser's stderr embedded, so the model sees the SyntaxError in the
same turn it wrote the bad code and the next round can fix or back out.

The file stays on disk in the broken state (surface-only, no rollback)
because rolling back forces the model to re-derive the entire edit from
scratch, whereas the broken state plus parser output gives it a precise
target to repair.

Skip policy: unsupported extensions, missing parser binaries, and parser
timeouts all return `(True, "")` — the gate adds signal where it can
and stays silent where it can't, so it never blocks a write a developer
could have made by hand."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Per-extension parser command. Each entry is the argv prefix; the
# absolute file path is appended at call time. A `None` value means the
# command is constructed inline (e.g. JSON via a Python one-liner).
_PARSERS: dict[str, tuple[str, ...]] = {
    ".js": ("node", "--check"),
    ".mjs": ("node", "--check"),
    ".cjs": ("node", "--check"),
    ".py": ("python", "-m", "py_compile"),
    ".json": (
        "python",
        "-c",
        # Two-arg form keeps the command tuple uniform: argv[0] is the
        # path appended below. `sys.exit(...)` propagates a non-zero
        # rc on parse failure; the JSONDecodeError's text reaches
        # stderr via the default traceback formatter.
        "import json,sys; json.load(open(sys.argv[1]))",
    ),
}

# How long a single parser invocation may run before we treat it as
# inconclusive. 5s is generous for any file under ~1 MB; if a parser
# hangs past that, returning ok=True lets the write proceed rather
# than blocking on a misbehaving tool.
_PARSE_CHECK_TIMEOUT_SECONDS: float = 5.0


def parse_check(path: Path) -> tuple[bool, str]:
    """Run an extension-specific syntax check on `path`.

    Returns `(ok, detail)`:
      - `ok=True, detail=""` — the parser ran and accepted the file, OR
        no parser is configured for this extension, OR the parser binary
        isn't installed, OR the parser timed out. All four cases are
        treated as "no signal" so the gate never blocks a write that a
        developer could have made by hand.
      - `ok=False, detail=<stderr+stdout>` — the parser ran and rejected
        the file. The combined stderr/stdout text is returned verbatim
        so the caller can embed it in the tool result for the model."""
    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        return (True, "")
    if shutil.which(parser[0]) is None:
        return (True, "")
    try:
        # S603: parser argv head is a hardcoded constant; path is the
        # workspace-bounded file the calling tool just wrote (validated
        # against escape attempts in EditFileTool/WriteFileTool). No
        # shell, no untrusted input — silenced deliberately.
        result = subprocess.run(  # noqa: S603
            [*parser, str(path)],
            capture_output=True,
            text=True,
            timeout=_PARSE_CHECK_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (True, "")
    if result.returncode == 0:
        return (True, "")
    detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return (False, detail)


__all__ = ["parse_check"]
