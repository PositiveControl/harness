"""Phraseology lint eval — replays a fixture of (utterance, expected
verdict, expected §) cases through `lint_utterance()` and scores
verdict accuracy + citation accuracy + per-scenario breakdown.

Phase-1 ship gate for the cite-grounded ATC transmission verifier
(epic harness-0pte). Sibling to the lint tool itself
(`src/harness/tools/phraseology_lint.py`, harness-q35t) — this module
turns the fixture (`character/<name>/phraseology_eval.yaml`,
harness-h2iz) into a measured pass-rate.

Decoupled from the model adapter via a `LintFn` callable, mirroring
`evals/atc_retrieval.py`'s SearchFn pattern. The CLI wires in a real
adapter + episodic store; tests script a deterministic LintFn so the
scorer stays pure.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.evals._corpus import load_yaml_cases, opt_str_field, str_field
from harness.tools.phraseology_lint import PhraseologyVerdict

# (utterance, scenario_hint) → PhraseologyVerdict.
# CLI wires this to lint_utterance(...) with adapter + store closed
# over; tests pass a deterministic stub.
LintFn = Callable[[str, str | None], PhraseologyVerdict]


@dataclass(frozen=True)
class PhraseologyFixtureRow:
    """One row from `phraseology_eval.yaml`. Mirrors the YAML schema
    documented in the fixture's `purpose:` block."""

    id: str
    scenario: str  # departure | arrival | handoff | emergency
    utterance: str
    expected_verdict: str  # ok | wrong | incomplete | out_of_scope
    expected_section: str | None  # "3-9-10" or None for oos
    expected_phraseology: str | None
    mismatch: str | None
    notes: str | None


@dataclass(frozen=True)
class PhraseologyEvalCase:
    """One scored case. Fixture row + actual verdict + per-case pass
    flags. Field-by-field comparison with the fixture row keeps the
    eval contract tight; if the model emits a verdict the fixture
    didn't anticipate (e.g. citing a section we never showed it),
    the lint tool's cite-or-silent gates already collapse it to oos."""

    row: PhraseologyFixtureRow
    actual: PhraseologyVerdict

    @property
    def verdict_pass(self) -> bool:
        return self.actual.verdict == self.row.expected_verdict

    @property
    def citation_pass(self) -> bool:
        # out_of_scope cases assert citation == None on both sides.
        # Symmetry matters: a model that says oos but parrots a section
        # has been normalized at the lint-tool layer (see
        # `lint_utterance` — oos verdict ⇒ null citation fields).
        if self.row.expected_section is None:
            return self.actual.expected_section is None
        return self.actual.expected_section == self.row.expected_section

    @property
    def passed(self) -> bool:
        return self.verdict_pass and self.citation_pass


@dataclass(frozen=True)
class PhraseologyEvalResult:
    cases: tuple[PhraseologyEvalCase, ...]

    @property
    def verdict_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.verdict_pass) / len(self.cases)

    @property
    def citation_accuracy(self) -> float:
        """Citation accuracy across ALL cases (oos cases included —
        their citation_pass tests for null on both sides). Distinct
        from `citation_accuracy_excluding_oos` because the bead's gate
        rolls oos into the combined metric."""
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.citation_pass) / len(self.cases)

    @property
    def combined_accuracy(self) -> float:
        """Both verdict AND citation must pass. This is the ship-gate
        metric (≥0.80)."""
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    def by_scenario(self) -> dict[str, dict[str, float]]:
        """Per-scenario {verdict_accuracy, citation_accuracy,
        combined_accuracy, count}. Helps spot whether a regression
        favors one operational class over another (e.g. emergency
        cases swung but departure held)."""
        buckets: dict[str, list[PhraseologyEvalCase]] = {}
        for case in self.cases:
            buckets.setdefault(case.row.scenario, []).append(case)
        out: dict[str, dict[str, float]] = {}
        for scenario, cases in buckets.items():
            n = len(cases)
            out[scenario] = {
                "verdict_accuracy": sum(1 for c in cases if c.verdict_pass) / n,
                "citation_accuracy": sum(1 for c in cases if c.citation_pass) / n,
                "combined_accuracy": sum(1 for c in cases if c.passed) / n,
                "count": float(n),
            }
        return out

    def failures(self) -> tuple[PhraseologyEvalCase, ...]:
        return tuple(c for c in self.cases if not c.passed)


def default_fixture_path(character_path: Path) -> Path:
    """`character/<name>/phraseology_eval.yaml` — same convention as
    `atc_eval.yaml` so a path-based test or a benchmark script can find
    it without hardcoding."""
    return character_path / "phraseology_eval.yaml"


def default_baseline_path(character_path: Path) -> Path:
    """Where `--save-baseline` writes the snapshot. Separate file from
    `phraseology_eval.yaml` so the fixture (the question set) and the
    baseline (the frozen pass-rate) move independently."""
    return character_path / "phraseology_baseline.json"


def load_fixture(path: Path) -> tuple[PhraseologyFixtureRow, ...]:
    """Parse `phraseology_eval.yaml` into validated rows.

    Required per case: id, scenario, utterance, expected_verdict.
    Optional: expected_section, expected_phraseology, mismatch, notes.

    Validates expected_verdict ∈ {ok, wrong, incomplete, out_of_scope}
    and that out_of_scope rows have null section + null phraseology.
    Other shape errors raise ValueError pointing at the offending row.
    """
    cases = load_yaml_cases(path, "phraseology")
    valid_verdicts = {"ok", "wrong", "incomplete", "out_of_scope"}
    rows: list[PhraseologyFixtureRow] = []
    for idx, entry in enumerate(cases):
        case_id = str_field(entry, "id", path, idx)
        scenario = str_field(entry, "scenario", path, idx)
        utterance = str_field(entry, "utterance", path, idx)
        verdict = str_field(entry, "expected_verdict", path, idx)
        if verdict not in valid_verdicts:
            raise ValueError(
                f"{path}[{idx}] expected_verdict {verdict!r} not in {sorted(valid_verdicts)}"
            )
        section = opt_str_field(entry, "expected_section")
        if section is not None:
            # Tolerate "§3-9-10" surface form in YAML; normalize.
            section = section.lstrip("§").strip().replace("−", "-")  # noqa: RUF001
        phraseology = opt_str_field(entry, "expected_phraseology")
        mismatch = opt_str_field(entry, "mismatch")
        notes = opt_str_field(entry, "notes")

        # Out-of-scope rows must null the citation fields. The fixture
        # author can document `mismatch` as a reason but a real cite
        # would mean the case isn't actually oos.
        if verdict == "out_of_scope":
            if section is not None or phraseology is not None:
                raise ValueError(
                    f"{path}[{idx}] out_of_scope row {case_id!r} must have "
                    "null expected_section AND expected_phraseology"
                )
        elif section is None:
            raise ValueError(f"{path}[{idx}] non-oos row {case_id!r} must carry expected_section")

        rows.append(
            PhraseologyFixtureRow(
                id=case_id,
                scenario=scenario,
                utterance=utterance,
                expected_verdict=verdict,
                expected_section=section,
                expected_phraseology=phraseology,
                mismatch=mismatch,
                notes=notes,
            )
        )
    return tuple(rows)


def run_phraseology_eval(
    fixture: Sequence[PhraseologyFixtureRow],
    lint_fn: LintFn,
) -> PhraseologyEvalResult:
    """Run every fixture row through `lint_fn` and aggregate the
    results. `lint_fn` takes (utterance, scenario_hint) and returns a
    PhraseologyVerdict; CLI passes a closure over `lint_utterance(...,
    adapter=..., episodic_store=...)`, tests pass a stub."""
    cases: list[PhraseologyEvalCase] = []
    for row in fixture:
        actual = lint_fn(row.utterance, row.scenario)
        cases.append(PhraseologyEvalCase(row=row, actual=actual))
    return PhraseologyEvalResult(cases=tuple(cases))


# ---------- Baseline comparator ----------
#
# Same shape as evals/atc_retrieval.py — per-case regression check
# (verdict / citation pass flips) plus aggregate-accuracy regression
# (verdict_accuracy / citation_accuracy / combined_accuracy). Pre-push
# hook calls `--compare-baseline`; non-zero exit blocks the push.


def load_baseline(path: Path) -> Mapping[str, Any]:
    """Read a saved `phraseology_baseline.json`. Raises FileNotFoundError
    when the file isn't present — caller surfaces that as a hint to
    run `--save-baseline` first."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


@dataclass(frozen=True)
class CasePassDelta:
    """Per-case pass movement between baseline and current run. We
    track verdict-pass and citation-pass separately so the diff
    explains WHICH dimension regressed (a verdict flip from `ok` to
    `wrong` is a different kind of failure than a citation that
    drifted §3-9-10 → §3-9-4)."""

    id: str
    old_verdict_pass: bool
    new_verdict_pass: bool
    old_citation_pass: bool
    new_citation_pass: bool

    @property
    def is_regression(self) -> bool:
        # Either dimension going from pass → fail counts as a regression.
        # A simultaneous improvement on the other dimension doesn't
        # cancel it; the gate is per-dimension.
        if self.old_verdict_pass and not self.new_verdict_pass:
            return True
        return self.old_citation_pass and not self.new_citation_pass

    @property
    def is_improvement(self) -> bool:
        if not self.old_verdict_pass and self.new_verdict_pass:
            return True
        return not self.old_citation_pass and self.new_citation_pass


@dataclass(frozen=True)
class AggregateAccuracyDelta:
    """One aggregate metric's old/new value pair. `metric` is the
    JSON key in the baseline file."""

    metric: str
    old: float
    new: float

    @property
    def is_regression(self) -> bool:
        return self.new < self.old


@dataclass(frozen=True)
class BaselineComparison:
    """Result of comparing a current eval result against a saved
    baseline. Mirrors `evals/atc_retrieval.BaselineComparison`."""

    case_deltas: tuple[CasePassDelta, ...]
    aggregate_deltas: tuple[AggregateAccuracyDelta, ...]
    new_cases: tuple[str, ...]
    dropped_cases: tuple[str, ...]

    @property
    def case_regressions(self) -> tuple[CasePassDelta, ...]:
        return tuple(d for d in self.case_deltas if d.is_regression)

    @property
    def case_improvements(self) -> tuple[CasePassDelta, ...]:
        return tuple(d for d in self.case_deltas if d.is_improvement)

    @property
    def aggregate_regressions(self) -> tuple[AggregateAccuracyDelta, ...]:
        return tuple(d for d in self.aggregate_deltas if d.is_regression)

    def has_regression(self, *, regression_budget: int = 0) -> bool:
        if self.aggregate_regressions:
            return True
        return len(self.case_regressions) > regression_budget


_AGGREGATE_METRICS: tuple[str, ...] = (
    "verdict_accuracy",
    "citation_accuracy",
    "combined_accuracy",
)


def compare_baselines(
    baseline: Mapping[str, Any],
    result: PhraseologyEvalResult,
) -> BaselineComparison:
    """Diff a saved baseline against a fresh eval result.

    The baseline JSON is whatever `--save-baseline` wrote; we read
    only the fields we need and tolerate missing optionals (an older
    snapshot without per-scenario detail still compares cleanly on
    the headline metrics)."""
    base_cases_raw = baseline.get("cases", []) or []
    base_cases: dict[str, Mapping[str, Any]] = {
        str(c["id"]): c for c in base_cases_raw if "id" in c
    }
    new_cases_map: dict[str, PhraseologyEvalCase] = {c.row.id: c for c in result.cases}

    shared_ids = sorted(base_cases.keys() & new_cases_map.keys())
    case_deltas = tuple(
        CasePassDelta(
            id=case_id,
            old_verdict_pass=bool(base_cases[case_id].get("verdict_pass", False)),
            new_verdict_pass=new_cases_map[case_id].verdict_pass,
            old_citation_pass=bool(base_cases[case_id].get("citation_pass", False)),
            new_citation_pass=new_cases_map[case_id].citation_pass,
        )
        for case_id in shared_ids
    )

    new_metric_values: dict[str, float] = {
        "verdict_accuracy": result.verdict_accuracy,
        "citation_accuracy": result.citation_accuracy,
        "combined_accuracy": result.combined_accuracy,
    }
    aggregate_deltas: list[AggregateAccuracyDelta] = []
    for metric in _AGGREGATE_METRICS:
        old = baseline.get(metric)
        if old is None:
            continue
        aggregate_deltas.append(
            AggregateAccuracyDelta(metric=metric, old=float(old), new=new_metric_values[metric])
        )

    return BaselineComparison(
        case_deltas=case_deltas,
        aggregate_deltas=tuple(aggregate_deltas),
        new_cases=tuple(sorted(new_cases_map.keys() - base_cases.keys())),
        dropped_cases=tuple(sorted(base_cases.keys() - new_cases_map.keys())),
    )
