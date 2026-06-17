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

# The `typeof`-guarded sibling of the ReferenceError trap (loop_run=55568949,
# harness-1ttpl / harness-k18er): the model writes `eval(src); if (typeof foot
# !== 'object') process.exit(1)`. `typeof` on a trapped binding returns
# 'undefined' rather than THROWING, so eval_blind_reference (which keys on the
# ReferenceError) never fires — the gate slips past submit, reds out
# byte-identically across every IMPLEMENT pass, and only the VERIFY
# byte-identical detector catches it, after the whole attempt budget is gone.
# Detect it statically from the test text: an eval call that does NOT append
# the assertions into the eval string, plus a `typeof <name>` guard on a name
# the eval'd source declares at top level with let/const. The working idiom
# `eval(src + '\n;…assertions…')` carries a `+` inside the eval parens before
# the close; the trapped forms `eval(src)` / `eval(fs.readFileSync(…))` never
# do — so a `+`-free eval call is the trap tell.
_EVAL_CALL_RE = re.compile(r"\beval\s*\(")
_EVAL_APPEND_RE = re.compile(r"\beval\s*\([^)\n]*\+")
_TYPEOF_RE = re.compile(r"\btypeof\s+([A-Za-z_$][\w$]*)\b")

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

# The function-body-truncation trap (loop_run=7b5de2af, harness-k18er): the
# model reads the source, extracts a function body with a `\{[^}]*\}` regex,
# then asserts the captured body contains (or lacks) some token. `[^}]*` is a
# NON-NESTING char class — it stops at the FIRST inner `}`, so for any
# function with nested blocks (if / for / object literal) the capture is a
# truncated fragment ending at that first brace, never the whole body. The
# real implementation lives past it, so a "body lacks X" assertion is red
# regardless of what the model writes. Unlike the eval traps this throws
# nothing and DOES observe the source (readFileSync) — only the byte-identical
# detector caught it, after the whole attempt budget was gone (k18er parked
# this way across 4 attempts, the model re-authoring the identical regex each
# time). Detect it statically: a `{[^}]*}` (or `[^}]+`) body capture in a test
# that reads source. The shared note steers re-authoring AWAY from another
# body-extraction regex — assert on the full source text, not a sliced body.
_BODY_TRUNCATION_RE = re.compile(r"\{\s*\[\^}\]\s*[*+]\??\s*\}")

GATE_BODY_TRUNCATION_NOTE = (
    "a `{[^}]*}` regex captures only up to the FIRST inner `}`, so a function "
    "with any nested block (if / for / object literal) yields a truncated "
    "fragment, not its full body — an assertion on that fragment is red no "
    "matter what the implementation does. Assert on the FULL source text "
    "directly (e.g. src.includes(\"player.mode === 'foot'\")), not on a "
    "regex-sliced function body"
)


# The phantom-member trap (loop_run=77f684f9, harness-o4cbj): the test reads
# the source and asserts (required-present) on a dotted member path the source
# does NOT use, because the test author invented an intermediate segment — the
# gate demands `player.car.speed` while the source models speed flat as
# `player.speed` (no `player.car.speed` exists, and a correct implementation
# never adds one). Red against the CORRECT implementation, so no IMPLEMENT pass
# can move it: the gate parks byte-identically once the verify detector catches
# it, after the whole attempt budget is gone (o4cbj parked this way across 4
# attempts). Unlike the eval traps it throws nothing and DOES read the source;
# unlike body-truncation the regex is well-formed — it just names a path that
# can't exist. Detect it statically with a tight discriminator: the full path
# `A.B.C` is absent from the source AND the middle-dropped path `A.C` IS present
# — proof the concept `C` already lives directly on `A`, so requiring the longer
# spelling contradicts the source. When `A.C` is also absent (genuine greenfield
# add), no flag. The shared note steers re-authoring toward asserting on the
# spelling the source actually uses.
_DOTTED_PATH_RE = re.compile(r"[A-Za-z_$][\w$]*(?:\\?\.[A-Za-z_$][\w$]*)+")

GATE_PHANTOM_MEMBER_NOTE = (
    "the gate requires the member path `{full}`, but {source} never uses it — "
    "the source expresses that field one level up as `{short}` (a correct "
    "implementation writes `{short}`, not `{full}`), so the assertion is red "
    "no matter what the implementation does. Assert on the member path the "
    "source actually uses (`{short}`), not an invented intermediate segment"
)


def _member_path_present(source_text: str, path: str) -> bool:
    """True when `source_text` contains `path` as a literal dotted member
    access (e.g. ``player.speed``), word-bounded so ``player.speed`` does
    not match inside ``player.speedometer``."""
    pattern = re.compile(r"\b" + r"\.".join(re.escape(p) for p in path.split(".")) + r"\b")
    return bool(pattern.search(source_text))


def _path_required_present(test_text: str, raw_path: str) -> bool:
    """True when at least one line mentioning `raw_path` asserts it
    PRESENT — i.e. the line carries no `!` negation before the path. A
    required-present assertion on a phantom path is the trap; a
    should-be-absent assertion (`!src.includes(path)`) goes green when the
    path is absent, so paths used only under negation are never flagged."""
    for line in test_text.splitlines():
        idx = line.find(raw_path)
        if idx == -1:
            continue
        if "!" not in line[:idx]:
            return True
    return False


def phantom_member_assertion(test_path: str | None, workspace: Path) -> str | None:
    """Static diagnostic for the phantom-member trap; None otherwise. Like
    the other static guards this reads the test TEXT — the trap throws
    nothing and the regex is well-formed, it just names a member path the
    source can't produce.

    Fires only when the test reads source (``readFileSync``) AND a
    required-present assertion references a 3-segment path ``A.B.C`` that is
    absent from the source while the middle-dropped path ``A.C`` is present
    — proof the concept lives one level up and the longer spelling
    contradicts the source. Conservative throughout: no source read, no such
    contradicted path, every occurrence negated, or an unreadable file →
    None."""
    if test_path is None:
        return None
    resolved = Path(test_path)
    if not resolved.is_absolute():
        resolved = workspace / resolved
    try:
        test_text = resolved.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
    except OSError:
        return None
    source_rels = _READ_SOURCE_RE.findall(test_text)
    if not source_rels:
        return None
    # Normalize regex-escaped dots (`player\.car\.speed`) to plain paths,
    # keeping the raw spelling for the per-line negation check.
    candidates: dict[str, str] = {}
    for raw in _DOTTED_PATH_RE.findall(test_text):
        norm = raw.replace("\\", "")
        if norm.count(".") == 2:  # exactly A.B.C
            candidates.setdefault(norm, raw)
    if not candidates:
        return None
    for source_rel in source_rels:
        source_path = Path(source_rel)
        if not source_path.is_absolute():
            source_path = workspace / source_path
        try:
            source_text = source_path.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
        except OSError:
            continue
        for full, raw in sorted(candidates.items()):
            a, _, c = full.split(".")
            short = f"{a}.{c}"
            if (
                not _member_path_present(source_text, full)
                and _member_path_present(source_text, short)
                and _path_required_present(test_text, raw)
            ):
                return GATE_PHANTOM_MEMBER_NOTE.format(full=full, short=short, source=source_rel)
    return None


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


def eval_blind_typeof_guard(test_path: str | None, workspace: Path) -> str | None:
    """Static diagnostic for the `typeof`-guarded eval-blind trap; None
    otherwise. Unlike `eval_blind_reference`, this reads the test TEXT (not
    runtime output) — the trap it catches produces no ReferenceError, so there
    is nothing in the output to key on.

    Fires only on the precise trap shape: an eval call that does NOT append
    the assertions into the eval string (the working idiom ``eval(src +
    '…assertions…')`` carries a ``+`` inside the parens and is never matched)
    plus a ``typeof <name>`` guard on a binding the eval'd source declares at
    top level with let/const. Such a guard reads 'undefined' for the trapped
    binding regardless of the implementation, so the gate is red forever.
    Conservative throughout: no eval, an append-idiom eval, no typeof guard,
    no readFileSync source, or an unreadable file → None."""
    if test_path is None:
        return None
    resolved = Path(test_path)
    if not resolved.is_absolute():
        resolved = workspace / resolved
    try:
        test_text = resolved.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
    except OSError:
        return None
    if not _EVAL_CALL_RE.search(test_text) or _EVAL_APPEND_RE.search(test_text):
        return None
    guarded = set(_TYPEOF_RE.findall(test_text))
    if not guarded:
        return None
    for source_rel in _READ_SOURCE_RE.findall(test_text):
        source_path = Path(source_rel)
        if not source_path.is_absolute():
            source_path = workspace / source_path
        try:
            source_text = source_path.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
        except OSError:
            continue
        for name in sorted(guarded):
            if _top_level_declares(source_text, name):
                return (
                    f"the test guards `typeof {name}` after eval-loading {source_rel}, "
                    f"but {source_rel} declares `{name}` at top level with let/const — "
                    f"`typeof` reads 'undefined' for the trapped binding (no "
                    f"ReferenceError), so the gate is red regardless of the "
                    f"implementation: {GATE_BLIND_IDIOM_NOTE}"
                )
    return None


def regex_body_truncation(test_path: str | None, workspace: Path) -> str | None:
    """Static diagnostic for the function-body-truncation trap; None
    otherwise. Like `eval_blind_typeof_guard` this reads the test TEXT — the
    trap produces no runtime error, just a structurally-red assertion on a
    truncated capture.

    Fires only when the test reads source (``readFileSync``) AND contains a
    ``{[^}]*}`` / ``{[^}]+}`` non-nesting body-capture regex. That char class
    stops at the first inner ``}``, so any function with a nested block
    captures a fragment, never the full body — an assertion on it is red
    regardless of the implementation. Conservative: no source read, or no
    truncating body capture, or an unreadable file → None."""
    if test_path is None:
        return None
    resolved = Path(test_path)
    if not resolved.is_absolute():
        resolved = workspace / resolved
    try:
        test_text = resolved.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
    except OSError:
        return None
    if not _READ_SOURCE_RE.search(test_text):
        return None
    if not _BODY_TRUNCATION_RE.search(test_text):
        return None
    return (
        f"the test slices a function body with a non-nesting `{{[^}}]*}}` regex "
        f"after reading the source, but {GATE_BODY_TRUNCATION_NOTE}"
    )


def first_blind_tell(output: str, test_path: str | None, workspace: Path) -> str | None:
    """First-failure blind-gate tell: run the full trap roster in priority
    order and return the first non-None diagnostic (None if the gate is
    clean).

    Single source of truth for the roster so the submit-time lint
    (`fsm_turn._lint_submitted_gate`) and the carried-gate VERIFY check
    (`fsm_turn._resolve_verify_outcome`) stay in lock-step. They drifted
    once (loop_run=135f0d99): the VERIFY site ran only the first three
    traps, so a phantom-member gate CARRIED from a prior attempt — which
    never re-submits, so it never sees the submit-time lint — got no
    first-failure tell and burned the verify-retry ceiling before the
    byte-identical detector finally caught it. Both callers route through
    here now; adding a trap updates both at once.

    `output` is the test run's combined output, consumed only by the
    runtime ReferenceError tell (`eval_blind_reference`); the other three
    traps read the test text statically and ignore it."""
    return (
        eval_blind_reference(output, test_path, workspace)
        or eval_blind_typeof_guard(test_path, workspace)
        or regex_body_truncation(test_path, workspace)
        or phantom_member_assertion(test_path, workspace)
    )
