"""Tests for the post-drive critic (harness-dopb)."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from harness.driver.critic import (
    CriticAdapterError,
    CriticFinding,
    _citation_in_workspace,
    _evidence_window,
    _extract_json_array,
    _finding_is_grounded,
    _locate_citation,
    _normalize_title,
    _parse_findings,
    _parse_verdict,
    _title_matches_any,
    _validate_finding,
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
