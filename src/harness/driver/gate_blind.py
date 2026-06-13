"""Deterministic eval-blind gate tell (harness-815wm).

loop_run=069d6172 (harness-xdgtb): every gate the model authored
stubbed the browser globals, loaded the source with
``eval(fs.readFileSync('game.js'))``, then asserted on top-level
bindings (``traffic``, ``peds``). Sloppy-mode direct eval hoists
FUNCTION declarations (and ``var``) into the calling scope, but
``let`` / ``const`` bindings stay trapped inside the eval — so the
gate fails ``ReferenceError: traffic is not defined`` even while the
source declares ``let traffic = []``. That red is structural, not the
gap: no IMPLEMENT pass can move it.

The byte-identical-failure detector (loop_run=dae002aa) eventually
catches the shape, but only after the verify-retry ceiling
(harness-hs50i) plus a re-authored gate that hits the same wall — and
its remediation text ("author a gate that LOADS the source") steers
the model back into the blind idiom. This module is the first-failure
tell: a ReferenceError on a name the eval'd source DOES declare at top
level with let/const proves the gate cannot observe the code under
test — diagnose immediately.

Conservative by construction: the test must direct-eval a source file
it reads itself (``readFileSync``), and the name must be let/const-
declared in that file. A ReferenceError on an UNDECLARED name is the
genuine gap (``drawTile is not defined`` where drawTile IS the
deliverable) and is never flagged; every unreadable / unparseable
case returns None.
"""

from __future__ import annotations

import re
from pathlib import Path

_REFERENCE_ERROR_RE = re.compile(r"ReferenceError: ([A-Za-z_$][\w$]*) is not defined")

# Source files the test reads — the eval-the-source idiom always goes
# through fs.readFileSync('<source>').
_READ_SOURCE_RE = re.compile(r"""readFileSync\(\s*['"]([^'"]+\.[cm]?js)['"]""")

# Bound reads; mirrors fsm_turn's _GATE_LINT_READ_CAP rationale.
_READ_CAP = 262_144

# Shared remediation text — the WRITE_TEST hint, the submit-time lint,
# and the suspect-gate halt all teach the same two working idioms.
GATE_BLIND_IDIOM_NOTE = (
    "direct eval hoists function declarations into the test scope, but "
    "top-level `let`/`const` bindings stay trapped inside the eval'd "
    "string — `eval(src); traffic.length` throws 'traffic is not defined' "
    "even when the source declares `let traffic = []`. Append the "
    "assertions INTO the eval'd string (eval(src + '\\n;if (...) "
    "process.exit(1)')) so they run in the same scope, or assert on the "
    "source text itself (readFileSync + regex)"
)


def _top_level_declares(source_text: str, name: str) -> bool:
    """True when `source_text` has a top-level ``let``/``const``
    declaration whose FIRST declarator is `name`. Later declarators in a
    multi-declarator statement are missed — a conservative miss means no
    flag, never a false one."""
    pattern = re.compile(rf"^(?:let|const)\s+{re.escape(name)}\b", re.MULTILINE)
    return bool(pattern.search(source_text))


def eval_blind_reference(output: str, test_path: str | None, workspace: Path) -> str | None:
    """Diagnostic when `output` shows the gate throwing ReferenceError on
    a binding the eval'd source declares at top level with let/const;
    None otherwise.

    `output` is the test run's combined output (the fuller the better —
    a Node stack trace pushes the ReferenceError line out of a short
    tail). `test_path` is the test script, workspace-relative or
    absolute; None means the caller couldn't resolve one."""
    if test_path is None:
        return None
    match = _REFERENCE_ERROR_RE.search(output)
    if match is None:
        return None
    name = match.group(1)
    resolved = Path(test_path)
    if not resolved.is_absolute():
        resolved = workspace / resolved
    try:
        test_text = resolved.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
    except OSError:
        return None
    if "eval(" not in test_text:
        return None
    for source_rel in _READ_SOURCE_RE.findall(test_text):
        source_path = Path(source_rel)
        if not source_path.is_absolute():
            source_path = workspace / source_path
        try:
            source_text = source_path.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
        except OSError:
            continue
        if _top_level_declares(source_text, name):
            return (
                f"the test throws ReferenceError on `{name}`, but {source_rel} "
                f"declares it at top level with let/const — the gate is "
                f"structurally blind to the source's state, not red on the "
                f"gap: {GATE_BLIND_IDIOM_NOTE}"
            )
    return None
