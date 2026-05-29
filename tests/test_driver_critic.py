"""Tests for the post-drive critic (harness-dopb)."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable

import pytest

from harness.driver.critic import (
    CriticAdapterError,
    CriticFinding,
    _citation_in_workspace,
    _code_quote_grounded,
    _evidence_window,
    _extract_json_array,
    _finding_is_grounded,
    _locate_citation,
    _normalize_title,
    _numbered_cost,
    _parse_findings,
    _parse_verdict,
    _Slice,
    _slice_file,
    _slice_snapshot,
    _title_matches_any,
    _validate_finding,
    budget_snapshot,
    critic_char_budget,
    run_critic,
)
from harness.model.adapter import ChatMessage

_SPEC = (
    "# Spec\n\n"
    "Pressing Escape toggles paused. While paused, skip the update step "
    "but still render the world and draw 'PAUSED' centered in white.\n\n"
    "Keys are read from a `keys` map populated by the keydown / keyup "
    "listeners — both must update the same map.\n"
)
_GAME_SOURCE = "\n".join(f"line {i}" for i in range(1, 200))
_SNAP = {"game.js": _GAME_SOURCE, "smoke.js": "console.log('ok');"}


def _finding_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "title": "keys map never written — game is unresponsive",
        "description": "The keydown listener writes to keyStates, not keys. game.js:42",
        "acceptance": "All control reads observe keypresses; can drive the car with WASD.",
        "priority": 0,
        "spec_quote": "both must update the same map",
        "evidence_path": "game.js:42",
        "code_quote": "line 42",  # verbatim _GAME_SOURCE content at game.js:42
    }
    base.update(overrides)
    return base


# ---- helpers -----------------------------------------------------------


def test_normalize_title_lowercases_and_collapses_punctuation() -> None:
    assert _normalize_title("Pause Toggle (Escape)!") == "pause toggle escape"


def test_title_matches_any_dupe_detection() -> None:
    assert _title_matches_any("Pause toggle broken", ("Pause toggle is broken",))


def test_title_matches_any_distinct() -> None:
    assert not _title_matches_any("Pause toggle broken", ("HUD rendered under camera",))


def test_citation_in_workspace_resolves_existing_line() -> None:
    assert _citation_in_workspace("game.js:42", _SNAP)


def test_citation_in_workspace_rejects_out_of_bounds() -> None:
    assert not _citation_in_workspace("game.js:9999", _SNAP)


def test_citation_in_workspace_rejects_unknown_file() -> None:
    assert not _citation_in_workspace("nope.js:1", _SNAP)


def test_citation_in_workspace_tolerates_basename() -> None:
    """Model often drops the leading directory; we accept basename
    matches as a courtesy."""
    snap = {"src/game.js": _GAME_SOURCE}
    assert _citation_in_workspace("game.js:42", snap)


def test_citation_in_workspace_no_citation_returns_false() -> None:
    assert not _citation_in_workspace("just prose with no citation", _SNAP)


# ---- _extract_json_array / _parse_findings -----------------------------


def test_extract_json_array_passes_clean_array() -> None:
    raw = '[{"title": "x"}]'
    assert _extract_json_array(raw) == '[{"title": "x"}]'


def test_extract_json_array_strips_fences() -> None:
    raw = '```json\n[{"title": "x"}]\n```'
    extracted = _extract_json_array(raw)
    assert extracted is not None
    assert "{" in extracted
    assert "}" in extracted


def test_extract_json_array_finds_buried_array() -> None:
    raw = 'Here are my findings:\n\n[{"title": "x"}]\n\nThanks!'
    extracted = _extract_json_array(raw)
    assert extracted is not None
    assert "title" in extracted


def test_extract_json_array_returns_none_when_absent() -> None:
    assert _extract_json_array("no array here, just words") is None


def test_parse_findings_returns_dicts() -> None:
    raw = json.dumps([{"title": "a"}, {"title": "b"}])
    parsed = _parse_findings(raw)
    assert len(parsed) == 2
    assert all(isinstance(item, dict) for item in parsed)


def test_parse_findings_drops_non_dict_entries() -> None:
    raw = '[{"title": "a"}, "string", 42, null]'
    parsed = _parse_findings(raw)
    assert parsed == [{"title": "a"}]


def test_parse_findings_returns_empty_on_garbage() -> None:
    assert _parse_findings("not json at all") == []


def test_parse_findings_returns_empty_on_non_list_root() -> None:
    """The schema requires a list. A single object at the root is wrong."""
    assert _parse_findings('{"title": "x"}') == []


# ---- _validate_finding -------------------------------------------------


def test_validate_finding_accepts_well_formed() -> None:
    finding = _validate_finding(
        _finding_dict(),
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        open_titles=(),
    )
    assert isinstance(finding, CriticFinding)
    assert finding.title == "keys map never written — game is unresponsive"
    assert finding.priority == 0


def test_validate_finding_drops_when_no_file_line_citation() -> None:
    raw = _finding_dict(description="No file line here.", evidence_path="game.js")
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_drops_when_spec_quote_not_in_spec() -> None:
    raw = _finding_dict(spec_quote="this phrase is fabricated and absent")
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_drops_when_out_of_bounds_line() -> None:
    raw = _finding_dict(evidence_path="game.js:9999", description="see game.js:9999")
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_drops_dupe_title() -> None:
    raw = _finding_dict()
    open_titles = ("keys map never written: game is unresponsive",)
    assert (
        _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=open_titles)
        is None
    )


def test_validate_finding_drops_missing_required_field() -> None:
    raw = _finding_dict()
    del raw["acceptance"]
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_drops_priority_out_of_range() -> None:
    assert (
        _validate_finding(
            _finding_dict(priority=5),
            spec_text=_SPEC,
            workspace_snapshot=_SNAP,
            open_titles=(),
        )
        is None
    )


def test_validate_finding_drops_priority_wrong_type() -> None:
    assert (
        _validate_finding(
            _finding_dict(priority="critical"),
            spec_text=_SPEC,
            workspace_snapshot=_SNAP,
            open_titles=(),
        )
        is None
    )


def test_validate_finding_drops_spec_quote_too_short() -> None:
    raw = _finding_dict(spec_quote="paused")  # under _SPEC_QUOTE_MIN
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_skips_spec_check_when_no_spec_supplied() -> None:
    """When spec_text is None, spec_quote can be empty and still pass —
    the artifact-only critic path."""
    raw = _finding_dict(spec_quote="")
    finding = _validate_finding(raw, spec_text=None, workspace_snapshot=_SNAP, open_titles=())
    assert isinstance(finding, CriticFinding)


def test_validate_finding_accepts_citation_in_description_only() -> None:
    raw = _finding_dict(
        evidence_path="game.js",  # no line
        description="The bug is at game.js:42 — control wiring broken.",
    )
    finding = _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=())
    assert isinstance(finding, CriticFinding)


# ---- code_quote grounding gate (harness-mur6) --------------------------


def test_validate_finding_drops_missing_code_quote() -> None:
    """A finding with no code_quote is invalid — the model must copy the
    real source at the cited line."""
    raw = _finding_dict()
    del raw["code_quote"]
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_drops_fabricated_code_quote() -> None:
    """code_quote that doesn't appear near the cited line is dropped — the
    model cited a real line but invented what's on it (the harness-lpsq
    fabrication mode)."""
    raw = _finding_dict(code_quote="if (tile === '=') { passThroughBuilding(); }")
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_drops_too_short_code_quote() -> None:
    """A trivially short quote can't ground a finding."""
    raw = _finding_dict(code_quote="});")
    assert _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=()) is None


def test_validate_finding_code_quote_tolerates_whitespace() -> None:
    """Indentation / spacing differences don't matter — match is
    whitespace-normalized."""
    raw = _finding_dict(code_quote="   line    42   ")
    finding = _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=())
    assert isinstance(finding, CriticFinding)
    assert finding.code_quote == "line    42"  # stored stripped, not normalized


def test_validate_finding_code_quote_matches_within_window() -> None:
    """code_quote need not be exactly on the cited line — anywhere in the
    +/- _VERIFY_WINDOW band counts (the model often cites a block head)."""
    raw = _finding_dict(evidence_path="game.js:42", code_quote="line 46")
    finding = _validate_finding(raw, spec_text=_SPEC, workspace_snapshot=_SNAP, open_titles=())
    assert isinstance(finding, CriticFinding)


# ---- context budget (harness-zk3c) -------------------------------------


def test_critic_char_budget_reserves_output_and_spec() -> None:
    no_spec = critic_char_budget(context_window=32768, max_tokens=4096, spec_text=None)
    with_spec = critic_char_budget(context_window=32768, max_tokens=4096, spec_text="x" * 1000)
    assert no_spec > 0
    assert with_spec == no_spec - 1000  # spec chars come straight off the budget
    # A bigger window grants a bigger budget.
    bigger = critic_char_budget(context_window=65536, max_tokens=4096, spec_text=None)
    assert bigger > no_spec


def test_critic_char_budget_floors_at_zero() -> None:
    # Output reserve alone exceeds a tiny window → no room for source.
    assert critic_char_budget(context_window=100, max_tokens=4096, spec_text=None) == 0


def test_budget_snapshot_keeps_everything_under_budget() -> None:
    snap = {"a.js": "x\n" * 10, "b.js": "y\n" * 10}
    kept, notes = budget_snapshot(snap, char_budget=100_000)
    assert kept == snap
    assert notes == []


def test_budget_snapshot_truncates_overflow_file_on_line_boundary() -> None:
    big = "\n".join(f"line{i}" for i in range(100))  # 100 lines
    kept, notes = budget_snapshot({"big.js": big}, char_budget=200)
    assert "big.js" in kept
    kept_lines = kept["big.js"].count("\n") + 1
    assert kept_lines < 100  # truncated
    # Kept prefix is a whole-line prefix (line numbers stay accurate).
    assert big.startswith(kept["big.js"])
    assert any("truncated" in n for n in notes)


def test_budget_snapshot_drops_later_files_when_budget_spent() -> None:
    first = "\n".join(f"a{i}" for i in range(40))
    second = "\n".join(f"b{i}" for i in range(40))
    # Budget fits exactly the first file's numbered cost.
    kept, notes = budget_snapshot(
        {"a.js": first, "b.js": second}, char_budget=_numbered_cost(first)
    )
    assert "a.js" in kept
    assert "b.js" not in kept
    assert any(n.startswith("b.js") and "dropped" in n for n in notes)


# ---- symbol-aligned slicer (harness-a0yj) ------------------------------

_PY_SRC = (
    "import os\n"  # 1
    "\n"  # 2
    "def alpha():\n"  # 3
    "    return 1\n"  # 4
    "\n"  # 5
    "def beta():\n"  # 6
    "    return 2\n"  # 7
    "\n"  # 8
    "def gamma():\n"  # 9
    "    return 3\n"  # 10
)


def _covers_all_lines(slices: list[_Slice], n: int) -> bool:
    """Slices tile lines 1..n contiguously with no gaps or overlaps."""
    spans = sorted((s.start_line, s.end_line) for s in slices)
    if not spans:
        return n == 0
    if spans[0][0] != 1 or spans[-1][1] != n:
        return False
    return all(nxt[0] == cur[1] + 1 for cur, nxt in itertools.pairwise(spans))


def test_slice_file_covers_all_lines_no_gaps() -> None:
    slices = _slice_file("g.py", _PY_SRC, target_chars=10_000)
    assert _covers_all_lines(slices, len(_PY_SRC.splitlines()))


def test_slice_file_true_line_numbers() -> None:
    slices = _slice_file("g.py", _PY_SRC, target_chars=10_000)
    assert slices[0].start_line == 1
    assert "    1 | import os" in slices[0].numbered_text
    last_line = len(_PY_SRC.splitlines())
    assert f"{last_line:>5} | " in slices[-1].numbered_text  # true last line number


def test_slice_file_splits_on_symbol_boundaries_under_small_target() -> None:
    from harness.tools._symbols import SymbolsUnavailableError, outline

    try:
        tops = [s for s in outline(_PY_SRC, filename="g.py") if s.depth == 0]
    except SymbolsUnavailableError:
        pytest.skip("tree-sitter [code] extra not installed")
    slices = _slice_file("g.py", _PY_SRC, target_chars=1)  # never merge spans
    assert _covers_all_lines(slices, len(_PY_SRC.splitlines()))
    starts = {s.start_line for s in slices}
    # No definition is split: every top-level symbol begins a slice.
    assert all(t.start_line in starts for t in tops)


def test_slice_file_unknown_extension_degrades_to_single_slice() -> None:
    slices = _slice_file("notes.txt", "a\nb\nc\n", target_chars=1)
    assert len(slices) == 1
    assert (slices[0].start_line, slices[0].end_line) == (1, 3)


def test_slice_file_empty_file() -> None:
    assert _slice_file("g.py", "", target_chars=10) == []


def test_slice_snapshot_flattens_path_ordered() -> None:
    snap = {"b.py": _PY_SRC, "a.py": "def z():\n    return 0\n"}
    slices = _slice_snapshot(snap, target_chars=10_000)
    paths = [s.path for s in slices]
    assert paths == sorted(paths)
    assert {"a.py", "b.py"} <= set(paths)


# ---- run_critic --------------------------------------------------------


class _StubAdapter:
    """ModelAdapter stand-in that returns a canned response."""

    id = "stub"
    context_window = 32000

    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        msgs = list(messages)
        self.calls.append(msgs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_run_critic_returns_validated_findings() -> None:
    adapter = _StubAdapter(json.dumps([_finding_dict()]))
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=("harness-yd3m",),
        open_under_epic=(),
        verify_grounding=False,
    )
    assert len(findings) == 1
    assert findings[0].title.startswith("keys map never written")


def test_run_critic_drops_invalid_keeps_valid() -> None:
    """Mix one valid finding with two invalid ones — only the valid
    one survives validation."""
    candidates = [
        _finding_dict(spec_quote="fabricated text not in spec"),  # bad spec
        _finding_dict(  # good one
            title="HUD rendered under camera transform",
            description="HUD draws at world coords; see game.js:100",
            evidence_path="game.js:100",
            code_quote="line 100",
        ),
        _finding_dict(  # bad citation
            title="Wanted level starts at 5",
            description="No citation here.",
            evidence_path="game.js",
        ),
    ]
    adapter = _StubAdapter(json.dumps(candidates))
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
        verify_grounding=False,
    )
    assert len(findings) == 1
    assert findings[0].title == "HUD rendered under camera transform"


def test_run_critic_raises_on_adapter_error() -> None:
    """An adapter failure must NOT be swallowed as 'no bugs found' — it
    propagates as CriticAdapterError so the loop can't false-converge on
    an outage (harness-fote)."""
    adapter = _StubAdapter(RuntimeError("model unavailable"))
    with pytest.raises(CriticAdapterError):
        run_critic(
            adapter=adapter,
            spec_text=_SPEC,
            workspace_snapshot=_SNAP,
            closed_this_run=(),
            open_under_epic=(),
        )


def test_run_critic_raises_when_verify_call_fails() -> None:
    """A failure on the per-finding verify call also propagates — the
    generation succeeded but the model became unreachable mid-pass."""
    adapter = _SeqAdapter([json.dumps([_finding_dict()]), RuntimeError("verify down")])
    with pytest.raises(CriticAdapterError):
        run_critic(
            adapter=adapter,
            spec_text=_SPEC,
            workspace_snapshot=_SNAP,
            closed_this_run=(),
            open_under_epic=(),
        )


def test_run_critic_returns_empty_on_non_json_output() -> None:
    adapter = _StubAdapter("Sorry, I couldn't find any bugs in the code.")
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
    )
    assert findings == []


def test_run_critic_caps_at_max_findings() -> None:
    """Twenty valid candidates with default max=10 — only 10 returned."""
    candidates = [
        _finding_dict(
            title=f"distinct bug #{i:02d}",
            description=f"see game.js:{i + 10}",
            evidence_path=f"game.js:{i + 10}",
            code_quote=f"line {i + 10}",
        )
        for i in range(20)
    ]
    adapter = _StubAdapter(json.dumps(candidates))
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
        max_findings=10,
        verify_grounding=False,
    )
    assert len(findings) == 10


def test_run_critic_drops_dupes_against_open_titles() -> None:
    candidates = [
        _finding_dict(title="HUD rendered under camera transform"),
        _finding_dict(  # near-dupe of an open one
            title="keys map never written, game unresponsive"
        ),
    ]
    adapter = _StubAdapter(json.dumps(candidates))
    open_titles = ("keys map never written — game is unresponsive",)
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=open_titles,
        verify_grounding=False,
    )
    # Only the HUD finding survives — the keys finding dupes the open one.
    assert len(findings) == 1
    assert findings[0].title == "HUD rendered under camera transform"


def test_run_critic_works_without_spec() -> None:
    """spec_text=None path — findings still validated for citations
    but spec_quote is not checked."""
    candidates = [_finding_dict(spec_quote="")]
    adapter = _StubAdapter(json.dumps(candidates))
    findings = run_critic(
        adapter=adapter,
        spec_text=None,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
        verify_grounding=False,
    )
    assert len(findings) == 1


def test_run_critic_prompt_includes_spec_and_workspace() -> None:
    adapter = _StubAdapter("[]")
    run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=("a",),
        open_under_epic=("b",),
    )
    assert adapter.calls, "adapter.complete should have been called"
    user = adapter.calls[0][-1].content
    assert "[SPEC]" in user
    assert "Pressing Escape toggles paused" in user
    assert "=== game.js ===" in user
    assert "   42 | line 42" in user  # source is line-numbered (harness-mur6)
    assert "[CLOSED THIS RUN]" in user
    assert "- a" in user
    assert "[ALREADY OPEN UNDER EPIC]" in user
    assert "- b" in user


# ---- grounding-verify gate (harness-hdwp) ------------------------------


class _SeqAdapter:
    """ModelAdapter stand-in that returns queued responses in order — the
    first `complete` call gets the critic findings, each subsequent call
    gets a verify verdict. Raises if the queue is exhausted."""

    id = "seq"
    context_window = 32000

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(list(messages))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_parse_verdict_grounded() -> None:
    assert _parse_verdict("GROUNDED: line 42 compares against the wrong tile")


def test_parse_verdict_refuted() -> None:
    assert not _parse_verdict("REFUTED: the code already checks tile === 'B'")


def test_parse_verdict_refuted_first_wins() -> None:
    """A verdict that refutes first loses even if it later says grounded."""
    assert not _parse_verdict("REFUTED — this is not grounded in the code")


def test_parse_verdict_missing_keyword_rejects() -> None:
    """Default-reject: anything without an explicit GROUNDED is a no."""
    assert not _parse_verdict("I'm not sure, the code is ambiguous here.")


def test_evidence_window_builds_numbered_window() -> None:
    win = _evidence_window(_validate_finding_ok(), _SNAP)
    assert win is not None
    key, line, text = win
    assert key == "game.js"
    assert line == 42
    assert "> " in text  # cited line is marked
    assert "42 |" in text
    assert "line 42" in text  # _GAME_SOURCE content at the cited line
    assert "line 36" in text  # window lower bound
    assert "line 48" in text  # window upper bound


def test_evidence_window_none_when_citation_unresolvable() -> None:
    # Built directly — the validator would reject an unresolvable citation,
    # but the snapshot could shift between validate and verify in theory.
    finding = CriticFinding(
        title="ghost bug",
        description="no citation in this prose",
        acceptance="n/a",
        priority=2,
        spec_quote="",
        evidence_path="ghost.js",
        code_quote="irrelevant",
    )
    assert _evidence_window(finding, _SNAP) is None


def test_locate_citation_resolves_and_rejects() -> None:
    assert _locate_citation("game.js:42", _SNAP) == ("game.js", 42)
    assert _locate_citation("game.js:9999", _SNAP) is None
    assert _locate_citation("nope", _SNAP) is None


def test_finding_is_grounded_accepts_on_grounded_verdict() -> None:
    adapter = _SeqAdapter(["GROUNDED: yep, line 42 is wrong"])
    assert _finding_is_grounded(
        adapter,
        _validate_finding_ok(),
        _SNAP,
        max_tokens=64,
        temperature=0.0,
    )


def test_finding_is_grounded_rejects_on_refuted_verdict() -> None:
    adapter = _SeqAdapter(["REFUTED: the code already does the right thing"])
    assert not _finding_is_grounded(
        adapter,
        _validate_finding_ok(),
        _SNAP,
        max_tokens=64,
        temperature=0.0,
    )


def test_finding_is_grounded_rejects_when_no_window() -> None:
    """No resolvable citation → no code to show → reject without a call."""
    adapter = _SeqAdapter([])  # would raise if called
    finding = CriticFinding(
        title="ghost bug",
        description="no citation in this prose",
        acceptance="n/a",
        priority=2,
        spec_quote="",
        evidence_path="ghost.js",
        code_quote="irrelevant",
    )
    assert not _finding_is_grounded(
        adapter,
        finding,
        _SNAP,
        max_tokens=64,
        temperature=0.0,
    )


def test_finding_is_grounded_raises_on_adapter_error() -> None:
    """A verify-call adapter failure propagates as CriticAdapterError —
    an outage is not a silent default-reject (harness-fote). (A
    successful-but-unconfirmed response still default-rejects; that's the
    REFUTED/no-keyword path, covered separately.)"""
    adapter = _StubAdapter(RuntimeError("model down"))
    with pytest.raises(CriticAdapterError):
        _finding_is_grounded(
            adapter,
            _validate_finding_ok(),
            _SNAP,
            max_tokens=64,
            temperature=0.0,
        )


def test_run_critic_verify_gate_drops_refuted_finding() -> None:
    """End-to-end: a finding that clears the deterministic gates but whose
    cited code doesn't support the claim (verifier says REFUTED) is dropped
    — this is the harness-lpsq fabrication mode (6rai/l4je/draw-order)."""
    adapter = _SeqAdapter([json.dumps([_finding_dict()]), "REFUTED: code is fine"])
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
    )
    assert findings == []
    assert len(adapter.calls) == 2  # critic call + one verify call


def test_run_critic_verify_gate_keeps_grounded_finding() -> None:
    adapter = _SeqAdapter([json.dumps([_finding_dict()]), "GROUNDED: confirmed at line 42"])
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
    )
    assert len(findings) == 1
    assert findings[0].title.startswith("keys map never written")


def _validate_finding_ok(**overrides: object) -> CriticFinding:
    """Build a CriticFinding via the real validator so the window/verify
    helpers get a record shaped exactly like production."""
    finding = _validate_finding(
        _finding_dict(**overrides),
        spec_text=None,
        workspace_snapshot=_SNAP,
        open_titles=(),
    )
    assert finding is not None
    return finding


# ---- run_critic slice mode (harness-a0yj) ------------------------------


def test_run_critic_slice_mode_aggregates_across_slices() -> None:
    """_SNAP has 2 files (no extractable symbols -> one whole-file slice
    each) = 2 generation calls. Distinct findings from each slice are
    aggregated."""
    f_game = json.dumps(
        [
            _finding_dict(
                title="bug in game",
                description="see game.js:42",
                evidence_path="game.js:42",
                code_quote="line 42",
            )
        ]
    )
    f_smoke = json.dumps(
        [
            _finding_dict(
                title="bug in smoke",
                description="see smoke.js:1",
                evidence_path="smoke.js:1",
                code_quote="console.log('ok');",
            )
        ]
    )
    adapter = _SeqAdapter([f_game, f_smoke])  # path order: game.js, smoke.js
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
        slice_mode=True,
        verify_grounding=False,
    )
    assert {f.title for f in findings} == {"bug in game", "bug in smoke"}
    assert len(adapter.calls) == 2  # one generation call per slice


def test_run_critic_slice_mode_raises_on_adapter_error() -> None:
    """A slice generation failure propagates (an outage is not 'no bugs')."""
    adapter = _SeqAdapter([RuntimeError("slice gen down")])
    with pytest.raises(CriticAdapterError):
        run_critic(
            adapter=adapter,
            spec_text=_SPEC,
            workspace_snapshot=_SNAP,
            closed_this_run=(),
            open_under_epic=(),
            slice_mode=True,
        )


def test_run_critic_slice_mode_dedups_same_finding_across_slices() -> None:
    """Two slices surfacing the identical finding file it once."""
    same = json.dumps(
        [
            _finding_dict(
                title="same bug",
                description="see game.js:42",
                evidence_path="game.js:42",
                code_quote="line 42",
            )
        ]
    )
    adapter = _SeqAdapter([same, same])
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
        slice_mode=True,
        verify_grounding=False,
    )
    assert len(findings) == 1


def test_run_critic_slice_mode_verify_gate_still_fires() -> None:
    """Gates run against the full snapshot regardless of mode: a slice
    finding that reaches verify and is REFUTED is dropped."""
    adapter = _SeqAdapter([json.dumps([_finding_dict()]), "[]", "REFUTED: nope"])
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
        slice_mode=True,
    )
    assert findings == []
    assert len(adapter.calls) == 3  # 2 slice gen calls + 1 verify


# ---- harness-ascz7.3: dedup before the verify gate ---------------------


def test_run_critic_dedups_before_verify_gate() -> None:
    """A duplicate-title candidate must be deduped BEFORE the verify call,
    not after — so each distinct title is verified at most once, even when
    the first instance is REFUTED (harness-ascz7.3). Pre-fix, a refuted
    finding's near-dupes each paid a fresh verify round-trip because the
    seen-set was only populated on a passing verdict."""
    candidates = [_finding_dict(), _finding_dict()]  # identical title
    # gen + exactly ONE verify (REFUTED). If the dupe re-verified, the
    # SeqAdapter queue would be exhausted and raise.
    adapter = _SeqAdapter([json.dumps(candidates), "REFUTED: not grounded"])
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
    )
    assert findings == []
    assert len(adapter.calls) == 2  # 1 generation + 1 verify (dupe skipped)


# ---- harness-ascz7.1: verify-outage salvage + retry --------------------


def test_run_critic_salvages_verified_findings_on_verify_outage() -> None:
    """A verify-stage outage mid-pass must NOT discard the whole pass:
    findings already cleared by the verify gate are salvaged and returned
    (harness-ascz7.1)."""
    a = _finding_dict(
        title="A distinct bug",
        description="see game.js:42",
        evidence_path="game.js:42",
        code_quote="line 42",
    )
    b = _finding_dict(
        title="B distinct bug",
        description="see game.js:100",
        evidence_path="game.js:100",
        code_quote="line 100",
    )
    # A verifies GROUNDED; B's verify fails on every retry attempt.
    responses: list[str | Exception] = [
        json.dumps([a, b]),
        "GROUNDED",
        RuntimeError("verify down"),
        RuntimeError("verify down"),
        RuntimeError("verify down"),
    ]
    adapter = _SeqAdapter(responses)
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
    )
    assert [f.title for f in findings] == ["A distinct bug"]


def test_run_critic_verify_retry_recovers_transient_failure() -> None:
    """A transient verify failure is retried — two failures then a GROUNDED
    keeps the finding (harness-ascz7.1)."""
    adapter = _SeqAdapter(
        [
            json.dumps([_finding_dict()]),
            RuntimeError("blip"),
            RuntimeError("blip"),
            "GROUNDED",
        ]
    )
    findings = run_critic(
        adapter=adapter,
        spec_text=_SPEC,
        workspace_snapshot=_SNAP,
        closed_this_run=(),
        open_under_epic=(),
    )
    assert len(findings) == 1


def test_run_critic_verify_outage_with_nothing_salvaged_still_raises() -> None:
    """When a verify outage leaves NOTHING salvaged, it must still surface
    as CriticAdapterError — an outage with zero filed findings must not
    masquerade as a clean empty pass (harness-fote)."""
    adapter = _SeqAdapter(
        [
            json.dumps([_finding_dict()]),
            RuntimeError("down"),
            RuntimeError("down"),
            RuntimeError("down"),
        ]
    )
    with pytest.raises(CriticAdapterError):
        run_critic(
            adapter=adapter,
            spec_text=_SPEC,
            workspace_snapshot=_SNAP,
            closed_this_run=(),
            open_under_epic=(),
        )


# ---- harness-ascz7.2: code_quote tolerates injected editorial text -----

_BUG_SRC = "\n".join(
    [
        "function update() {",  # 1
        "  chainTimer = 3;",  # 2
        '  const u = "https://x";',  # 3
        "  lastShotAt;",  # 4
        "}",  # 5
    ]
)
_BUG_SNAP = {"game.js": _BUG_SRC}


def test_code_quote_grounded_tolerates_injected_trailing_comment() -> None:
    assert _code_quote_grounded(
        "chainTimer = 3; // BUG: should be 3000 milliseconds", ("game.js", 2), _BUG_SNAP
    )


def test_code_quote_grounded_tolerates_elision_markers() -> None:
    quote = "chainTimer = 3;\n// ... existing code ...\nlastShotAt;"
    assert _code_quote_grounded(quote, ("game.js", 3), _BUG_SNAP)


def test_code_quote_grounded_rejects_pure_editorial_quote() -> None:
    assert not _code_quote_grounded(
        "// this entire function is missing the decay mechanism", ("game.js", 2), _BUG_SNAP
    )


def test_code_quote_grounded_rejects_fabricated_with_trailing_comment() -> None:
    assert not _code_quote_grounded(
        "notReal = 5; // BUG: fabricated line", ("game.js", 2), _BUG_SNAP
    )


def test_code_quote_grounded_leaves_url_slashes_intact() -> None:
    """A `//` inside a string/URL must not be treated as a comment — the
    real source line still matches verbatim."""
    assert _code_quote_grounded('const u = "https://x"; // note', ("game.js", 3), _BUG_SNAP)
