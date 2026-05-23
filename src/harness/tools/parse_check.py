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

import re
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


# Node's `Identifier 'X' has already been declared` is the canonical
# pattern we want to enrich. Captures the identifier name so we can
# grep the file for ALL of its declaration sites — the model sees only
# the NEW offender's line by default and has to search for the prior
# decl, costing a round. Pre-compiled at module import so the hot path
# pays only the search cost.
_ALREADY_DECLARED_RE = re.compile(r"Identifier '([^']+)' has already been declared")

# Declaration-shape regex used by the dup-decl enrichment. Catches the
# four JS keywords + class/function statements. Bound to the identifier
# at format time via a precompiled-once-per-ident inline pattern below.
_DECL_KEYWORDS = ("let", "const", "var", "function", "class")


def _enrich_with_duplicate_locations(detail: str, path: Path) -> str:
    """harness-0tni: when the parser flags a duplicate-declaration
    error, append a grep of where the conflicting identifier is
    declared elsewhere in the file. Saves the model a search round on
    the loop run's most expensive failure mode (block-scope blindness
    introducing redeclarations).

    Returns the original `detail` untouched when:
      - the parser output doesn't match the dup-decl shape,
      - the file can't be read (e.g. concurrent writer),
      - fewer than 2 declaration sites are found (single match = the
        parser's own line is the better signal already).
    """
    match = _ALREADY_DECLARED_RE.search(detail)
    if not match:
        return detail
    ident = match.group(1)
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return detail
    keyword_alt = "|".join(_DECL_KEYWORDS)
    decl_re = re.compile(rf"^\s*({keyword_alt})\s+{re.escape(ident)}\b")
    locations: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if decl_re.search(line):
            locations.append(f"  line {lineno}: {line.strip()}")
    if len(locations) <= 1:
        return detail
    return (
        f"{detail}\n\nAll declarations of `{ident}` in {path.name}:\n"
        + "\n".join(locations)
        + f"\nRemove or rename {len(locations) - 1} of these to resolve the duplicate."
    )


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
    detail = _enrich_with_duplicate_locations(detail, path)
    return (False, detail)


__all__ = ["parse_check"]
