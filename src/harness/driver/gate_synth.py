"""Deterministic source-text gate synthesis (harness-l3tgq).

When WRITE_TEST would halt "no test" — the model never submitted a runnable
red gate and no existing on-disk gate could be adopted — the small executor
has usually authored a structurally-blind gate (eval-scope / body-slice) and
re-authored the SAME trap every attempt, despite the WRITE_TEST hint that
spells out the working idiom (loop_run=77f684f9: harness-l3tgq parked this way
across 4 attempts). The gap is real and the assessment already NAMES the code
shapes that close it; only the gate's STRUCTURE was blind.

Rather than park, synthesize a structure-safe gate the harness owns: read the
target source and regex the FULL text for code shapes pulled from the
assessment's own `approach` / `gap`. Three properties keep it strictly
no-worse-than-halt:

  1. Tokens come from the model's OWN assessment, so they are self-consistent
     with what the same model writes in the IMPLEMENT pass that follows — the
     gate asserts the spelling the implementer is about to produce, not a
     third party's guess.
  2. Only tokens ABSENT from the source right now are kept — the gate is
     red-now by construction and goes green exactly when the implementation
     adds them. If every candidate is already present there is nothing to
     gate, so synthesis declines.
  3. The scaffold is fixed and harness-authored (``readFileSync`` + a regex
     list; never an eval, never a ``{[^}]*}`` body slice), so it cannot be
     structurally blind. The caller still runs the result through the same
     submit-lint a real gate passes and only adopts a red, runnable, lint-clean
     result; anything else is discarded and the caller halts exactly as before.

This module is PURE: it resolves the source, extracts tokens, reads the source
to drop already-present shapes, and renders the gate file's text. Writing,
running, linting, and adopting belong to the caller (fsm_executor), mirroring the
gate_blind analysis/orchestration split.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bound source reads; mirrors gate_blind's _READ_CAP rationale.
_READ_CAP = 262_144

# A member compound/simple assignment LHS the assessment names as the change
# shape: `car.x +=`, `player.speed =`, `this.angle -=`. Requires at least one
# dot (a bare `speed = 0` is too generic to gate on) and a single `=` not part
# of `==`/`===`/`<=`/`>=`. The captured operator is normalized to `\s*` in the
# rendered regex so source whitespace doesn't matter.
_ASSIGN_RE = re.compile(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+)\s*([-+*/|&^]?=)(?!=)")

# A distinctive multi-hump camelCase identifier the assessment names — e.g.
# `nextDecisionAt`, `stateTimer`. Single-hump common words (`speed`, `angle`)
# are excluded by requiring an interior capital; this keeps the token set to
# deliberate symbol names, not prose.
_CAMEL_RE = re.compile(r"\b([a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+)\b")

# A source filename the assessment references ("in game.js (lines 680-714)").
_SOURCE_FILE_RE = re.compile(r"\b([\w./-]+\.(?:js|mjs|cjs))\b")

_TEST_NAME_RE = re.compile(r"(?:^|[_./])test[_.]|[_.]test\.", re.IGNORECASE)
_LOCAL_JS_SUFFIXES = frozenset({".js", ".mjs", ".cjs"})
_EXCLUDED_DIRS = frozenset({"node_modules", ".git", ".harness", "dist", "build"})

# Bounds on the synthesized REQUIRED list. At least two shapes — a single-shape
# gate is a weak, easily-spurious signal; cap at six to keep the gate focused.
_MIN_SHAPES = 2
_MAX_SHAPES = 6


@dataclass(frozen=True)
class SynthesizedGate:
    """A ready-to-write source-text gate. `test_filename` and `content` are
    workspace-relative / file text; `test_cmd` runs it; `shape_count` is how
    many absent-now shapes it asserts (for the adoption note)."""

    test_filename: str
    content: str
    test_cmd: str
    shape_count: int


def _looks_like_test(rel: Path) -> bool:
    return bool(_TEST_NAME_RE.search(rel.as_posix()))


def _resolve_source(assessment_text: str, workspace: Path) -> Path | None:
    """The source file the gate should read. Prefer a `.js` filename the
    assessment names that exists and isn't a test; else the largest non-test
    local-JS file in the workspace. None when the workspace has no such file."""
    for name in _SOURCE_FILE_RE.findall(assessment_text):
        rel = str(name)
        candidate = workspace / rel
        if candidate.is_file() and not _looks_like_test(Path(rel)):
            return candidate
    largest: tuple[int, Path] | None = None
    for path in workspace.rglob("*"):
        if path.suffix not in _LOCAL_JS_SUFFIXES or not path.is_file():
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if _looks_like_test(path.relative_to(workspace)):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if largest is None or size > largest[0]:
            largest = (size, path)
    return largest[1] if largest is not None else None


def _shape_regexes(assessment_text: str) -> list[tuple[str, re.Pattern[str]]]:
    """Ordered, de-duplicated (js_source, compiled) pairs for each code shape
    named in the assessment. `js_source` is the regex body for a JS literal
    `/.../`; the compiled Python pattern (same syntax) drives the absent-now
    check. First occurrence wins; order is preserved for a stable gate."""
    out: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()

    def add(js_source: str) -> None:
        if js_source in seen:
            return
        try:
            compiled = re.compile(js_source)
        except re.error:
            return
        seen.add(js_source)
        out.append((js_source, compiled))

    for lhs, op in _ASSIGN_RE.findall(assessment_text):
        # `car.x` + `+=` -> `car\.x\s*\+=` : literal dots, op-agnostic whitespace.
        add(re.escape(lhs) + r"\s*" + re.escape(op))
    for ident in _CAMEL_RE.findall(assessment_text):
        add(r"\b" + re.escape(ident) + r"\b")
    return out


def build_source_text_gate(
    assessment: Mapping[str, Any] | None,
    workspace: Path,
    *,
    is_browser_js: bool,
) -> SynthesizedGate | None:
    """Synthesize a structure-safe source-text gate from `assessment`, or None
    to decline (caller then halts unchanged).

    Declines when: the workspace is not browser-JS (the scaffold is a Node
    script), there is no assessment text, no resolvable source file, or fewer
    than `_MIN_SHAPES` of the assessment's named code shapes are ABSENT from
    the source (nothing red-now to gate)."""
    if not is_browser_js or assessment is None:
        return None
    assessment_text = "\n".join(
        str(assessment.get(field, "")) for field in ("approach", "gap", "current_state")
    ).strip()
    if not assessment_text:
        return None
    source = _resolve_source(assessment_text, workspace)
    if source is None:
        return None
    try:
        source_text = source.read_text(encoding="utf-8", errors="replace")[:_READ_CAP]
    except OSError:
        return None

    absent: list[str] = []
    for js_source, compiled in _shape_regexes(assessment_text):
        if compiled.search(source_text):
            continue  # already implemented — gating it adds no red-now signal
        absent.append(js_source)
        if len(absent) >= _MAX_SHAPES:
            break
    if len(absent) < _MIN_SHAPES:
        return None

    source_rel = source.relative_to(workspace).as_posix()
    required = ",\n  ".join(f"/{body}/" for body in absent)
    content = (
        "// Synthesized source-text gate (harness gate-synth, harness-l3tgq).\n"
        "// The executor never submitted a runnable gate; this asserts the\n"
        "// code shapes the assessment named are present in the source.\n"
        "const fs = require('fs');\n"
        f"const src = fs.readFileSync({source_rel!r}, 'utf8');\n"
        "const REQUIRED = [\n"
        f"  {required},\n"
        "];\n"
        "const missing = REQUIRED.filter((re) => !re.test(src));\n"
        "if (missing.length) {\n"
        "  console.error('FAIL: source missing required shapes: ' "
        "+ missing.map(String).join(', '));\n"
        "  process.exit(1);\n"
        "}\n"
        "console.log('OK: all required shapes present');\n"
    )
    test_filename = f"test_synth_{source.stem}.js"
    return SynthesizedGate(
        test_filename=test_filename,
        content=content,
        test_cmd=f"node {test_filename}",
        shape_count=len(absent),
    )
