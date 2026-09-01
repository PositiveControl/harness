"""Deterministic runtime behavioral-gate synthesis (harness gate-synth, runtime arm).

The source-text synthesizer (``gate_synth``) asserts that named code *shapes*
appear in the file. It cannot see *behavior*: "NPC cars never move" or "foot
mode is unreachable" are positional/runtime gaps whose tokens (``car.x +=``,
``mode = 'foot'``) already exist *somewhere* in the source, so a source-text
regex is green even while the behavior is broken. Those are exactly the gaps
that false-closed and reopened across drives — the headless smoke gate loads
the page but never steps the loop, so it can't observe motion over time.

This module synthesizes a RUNTIME gate from the assessment's own words: a
``--setup`` / ``--assert`` pair for ``smoke_runner`` (the primitive already
exists, harness-5vn6t / harness-u1il5). The setup drives input / stashes a
baseline before the settle window; the assert checks the resulting state after
the game loop has had frames to run, returning an array of ``RGFAIL:`` strings
(empty = pass).

Two recognized invariant patterns, both red-now-or-decline:

  1. **collection motion** — when the assessment names a position-integration
     gap on an entity collection (``traffic`` / ``cars`` / …), stash each
     element's ``(x, y)`` at load and assert at least one moved over the settle
     window. Red while integration is missing (positions frozen), green once
     ``e.x += …`` lands in the loop.
  2. **state reached after input** — when the assessment names a
     ``state = 'literal'`` transition plus a trigger key, dispatch the key(s)
     and assert the state reaches the literal. Red while the transition is
     absent (state stuck), green once the handler lands.

This module is PURE: it extracts tokens and renders JS strings. Writing the
probe files, running ``smoke_runner``, and the red-now / runnable / no-probe-
error adoption gauntlet belong to the caller (``fsm_executor``), mirroring the
source-text split. The caller adopts ONLY a candidate whose smoke run is red on
the gap (``RGFAIL:`` in the tail) and free of probe errors (a mis-extracted
collection / state symbol throws a ReferenceError that surfaces as a
``scenario error`` and is discarded, never a false red).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Common game-entity collection nouns. A bare identifier matching one of these
# (case-insensitively) is a candidate array to assert motion on. Kept curated —
# a generic "any plural identifier" net would emit motion probes on collections
# that legitimately don't move (e.g. spawn points), wasting smoke launches.
_COLLECTION_NOUNS = frozenset(
    {
        "traffic",
        "cars",
        "vehicles",
        "npcs",
        "npc",
        "peds",
        "pedestrians",
        "bullets",
        "projectiles",
        "enemies",
        "entities",
        "particles",
        "obstacles",
        "civilians",
    }
)

# A motion / position-integration gap signal in the assessment. Only when one
# of these fires do we emit a collection-motion candidate — otherwise an issue
# that names a collection for an unrelated reason wouldn't get a spurious
# motion gate (the adoption gate would discard it, but skipping the smoke
# launch is cheaper).
_MOTION_SIGNAL_RE = re.compile(
    r"position integration|don'?t move|do(?:es)? not move|never move|not? move|"
    r"\.x\s*\+=|\.y\s*\+=|integrat|cos\([^)]*\)\s*\*",
    re.IGNORECASE,
)

# An identifier USED as a collection: `X.length`, `X.map(`, `X.forEach(`,
# `X[i]`, `for (… of X)`, or member position access `X.x`/`X.y`. Captures the
# base identifier so we pick up the array name even when it isn't a known noun.
_COLLECTION_USE_RE = re.compile(
    r"\b([A-Za-z_$][\w$]*)\s*(?:\.(?:length|map|forEach|filter|some|every)\b|\[)"
    r"|for\s*\(\s*(?:const|let|var)\s+[\w$]+\s+of\s+([A-Za-z_$][\w$]*)"
)

# A `state = 'literal'` (or `===`) pair the assessment names. The path may be
# dotted (`player.mode`); the literal is a bare word in single or double
# quotes. Operator is `=`, `==`, or `===` so both assignments and comparisons
# the prose quotes are captured.
_STATE_LITERAL_RE = re.compile(
    r"([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*={1,3}\s*['\"]([A-Za-z_]\w*)['\"]"
)

# Trigger key codes (KeyboardEvent.code spelling) the assessment names.
_KEY_RE = re.compile(
    r"\b(Enter|Space|Tab|Escape|Backspace|"
    r"Key[A-Z]|Arrow(?:Up|Down|Left|Right)|Digit[0-9])\b"
)

# Bounds: cap candidates so a noisy assessment can't fan out into dozens of
# ~2s smoke launches. Motion + state combined stay under this.
_MAX_COLLECTIONS = 3
_MAX_STATE_PAIRS = 3
_MAX_KEYS = 6


@dataclass(frozen=True)
class RuntimeGateCandidate:
    """One runtime probe the caller can write + run. ``kind`` is ``'motion'``
    or ``'reached'`` (for the adoption note); ``setup_js`` runs before the
    settle window (statements; may throw → discarded), ``assert_js`` runs after
    and must ``return`` an array of failure strings; ``description`` is the
    human note threaded into the adoption ``failure_output``."""

    kind: str
    setup_js: str
    assert_js: str
    description: str


def _assessment_text(assessment: Mapping[str, Any] | None) -> str:
    if assessment is None:
        return ""
    return "\n".join(
        str(assessment.get(field, "")) for field in ("approach", "gap", "current_state")
    ).strip()


def _collection_identifiers(text: str) -> list[str]:
    """Ordered, de-duplicated candidate collection identifiers: known entity
    nouns present in the text, plus any identifier USED as an array/collection.
    First occurrence wins; order preserved for a stable gate."""
    out: list[str] = []
    seen: set[str] = set()

    def add(ident: str) -> None:
        if ident and ident not in seen:
            seen.add(ident)
            out.append(ident)

    # Known nouns, preserving the source casing of the first occurrence.
    for tok in re.findall(r"\b([A-Za-z_$][\w$]*)\b", text):
        if tok.lower() in _COLLECTION_NOUNS:
            add(tok)
    # Identifiers used structurally as collections.
    for m in _COLLECTION_USE_RE.finditer(text):
        add(m.group(1) or m.group(2))
    return out[:_MAX_COLLECTIONS]


def _state_pairs(text: str) -> list[tuple[str, str]]:
    """Ordered, de-duplicated (path, literal) pairs. A path with no dot and a
    common-word literal would be too generic, so require the path to be dotted
    OR the literal to co-occur with a transition the prose flags — but we lean
    on the adoption gate (red-now + no probe error) for precision rather than
    over-filtering here. The already-true literal (current state) comes up
    green and is discarded by the caller; only the unreached literal stays red."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, literal in _STATE_LITERAL_RE.findall(text):
        key = (path, literal)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out[:_MAX_STATE_PAIRS]


def _trigger_keys(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for code in _KEY_RE.findall(text):
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out[:_MAX_KEYS]


def _motion_candidate(coll: str) -> RuntimeGateCandidate:
    # `coll` resolves by bare reference against the page's global lexical scope
    # (a classic <script>'s top-level `let traffic = []` is reachable by name
    # from page.evaluate). A non-array / empty / undefined `coll` throws in
    # setup → smoke reports a setup-scenario error → the caller discards it.
    setup_js = f"const __c = {coll};\nwindow.__rg0 = __c.map(function(e){{ return [e.x, e.y]; }});"
    assert_js = (
        f"const __c = {coll};\n"
        "const __b = window.__rg0 || [];\n"
        "const __moved = __c.some(function(e, i){\n"
        "  return __b[i] && (e.x !== __b[i][0] || e.y !== __b[i][1]);\n"
        "});\n"
        "return __moved ? [] : ["
        f"'RGFAIL: no element of `{coll}` changed position over the settle "
        f"window — position integration missing in the update loop'];"
    )
    return RuntimeGateCandidate(
        kind="motion",
        setup_js=setup_js,
        assert_js=assert_js,
        description=f"`{coll}` elements never move over the settle window",
    )


def _reached_candidate(path: str, literal: str, keys: list[str]) -> RuntimeGateCandidate:
    codes = ", ".join(repr(k) for k in keys)
    # Dispatch each trigger key as a held keydown on BOTH window and document
    # (games wire the listener to either). The settle window then gives the
    # update loop frames to process the (edge-triggered) transition. A `path`
    # that doesn't resolve throws in the assert → surfaces as a scenario error
    # → discarded by the caller (never a false red).
    setup_js = (
        f"const __codes = [{codes}];\n"
        "for (const __code of __codes) {\n"
        "  const __ev = {code: __code, key: __code, bubbles: true};\n"
        "  window.dispatchEvent(new KeyboardEvent('keydown', __ev));\n"
        "  document.dispatchEvent(new KeyboardEvent('keydown', __ev));\n"
        "}"
    )
    assert_js = (
        f"const __v = {path};\n"
        f"return __v === {literal!r} ? [] : ["
        f"'RGFAIL: `{path}` did not reach ' + {literal!r}.toString() + "
        "' after the trigger input (got ' + JSON.stringify(__v) + ') — "
        "transition missing'];"
    )
    return RuntimeGateCandidate(
        kind="reached",
        setup_js=setup_js,
        assert_js=assert_js,
        description=f"`{path}` never reaches '{literal}' after input",
    )


def build_runtime_gates(
    assessment: Mapping[str, Any] | None,
    *,
    is_browser_js: bool,
) -> list[RuntimeGateCandidate]:
    """Synthesize runtime behavioral-gate candidates from ``assessment``.

    Returns an ordered list (caller tries each, adopts the first that is
    red-now + probe-clean). Empty when: not a browser-JS workspace, no
    assessment text, or no recognized invariant pattern. Motion candidates are
    emitted only when a position/motion signal fires AND a collection is named;
    state-reached candidates only when both a ``state = 'literal'`` pair and a
    trigger key are named."""
    if not is_browser_js:
        return []
    text = _assessment_text(assessment)
    if not text:
        return []

    candidates: list[RuntimeGateCandidate] = []

    if _MOTION_SIGNAL_RE.search(text):
        for coll in _collection_identifiers(text):
            candidates.append(_motion_candidate(coll))

    keys = _trigger_keys(text)
    if keys:
        for path, literal in _state_pairs(text):
            candidates.append(_reached_candidate(path, literal, keys))

    return candidates
