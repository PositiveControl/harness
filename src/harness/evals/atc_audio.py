"""ATC audio eval — score the cite-grounded lint pipeline on real
LiveATC utterances (harness-gy5z).

Reads `character/<name>/atc_audio/utt/*.jsonl` (the human-labelled
output of `scripts/atc_audio_label.py`) and runs each row through
`lint_utterance()` twice:

1. **Clean pass** — `transcript_text` (human-corrected). Isolates the
   verdict pipeline's quality on radio-style language with STT noise
   factored out.
2. **Noisy pass** — `transcript_seed` (whisper original). Measures
   the end-to-end WER → verdict drop on the same row, so we know how
   much the audio-mode pipeline loses to STT vs to the lint tool.

Per-utterance metrics: WER (whisper vs human), verdict_pass + cite_pass
on both passes, verdict_shift (clean was right, noisy was wrong).
Aggregates: mean_wer, clean/noisy verdict + citation accuracy,
verdict_shift_rate, by-event-tag breakdown.

Decoupled from the model adapter via the same `LintFn` callable that
`evals/phraseology.py` uses — the CLI wires in a real adapter, tests
script a deterministic stub.

Empty-fixture handling: if no utt/ rows exist (labelling hasn't started
or `--only` filtered everything out), the eval returns a zero-case
result. The CLI surfaces a hint; the pre-push gate skips cleanly.
Idempotent on (clip_id, utt_index) — labels are the contract, not the
order they were captured in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools.phraseology_lint import PhraseologyVerdict

# (utterance, scenario_hint) → PhraseologyVerdict. Same shape as
# evals/phraseology.LintFn so the CLI can pass the same closure.
LintFn = Callable[[str, str | None], PhraseologyVerdict]

VERDICTS = ("ok", "wrong", "incomplete", "out_of_scope")


# ---------- fixture row ----------


@dataclass(frozen=True)
class AudioFixtureRow:
    """One human-labelled utterance from utt/<clip_id>.jsonl. Mirrors
    the row schema written by `scripts/atc_audio_label.py`. Only the
    fields the eval actually reads are pinned here — `notes`, the
    timestamp, and ingest provenance fields ride through as-is on the
    JSON output but don't drive scoring."""

    clip_id: str
    utt_index: int
    speaker_role: str
    facility: str | None
    position: str | None
    event_tag: str | None
    transcript_text: str  # human-corrected
    transcript_seed: str  # whisper original
    expected_verdict: str
    expected_section: str | None
    expected_phraseology: str | None
    mismatch: str | None

    @property
    def case_id(self) -> str:
        """Stable identifier for baseline diffs. (clip_id, utt_index)
        is unique across the full corpus."""
        return f"{self.clip_id}:{self.utt_index:04d}"


def _utt_dir(target: Path) -> Path:
    return target / "utt"


def iter_utt_rows(target: Path) -> Iterator[dict[str, Any]]:
    """Yield each row from every utt/*.jsonl under `target`. Tolerates
    missing utt/ (returns nothing) and skips malformed lines so a
    partial-write crash doesn't kill the eval."""
    utt = _utt_dir(target)
    if not utt.exists():
        return
    for path in sorted(utt.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    continue


def load_fixture(
    target: Path,
    *,
    only_verified: bool = True,
    only_clip_ids: tuple[str, ...] | None = None,
) -> tuple[AudioFixtureRow, ...]:
    """Load utt/*.jsonl rows into AudioFixtureRow tuples.

    `only_verified=True` (default) drops rows where `human_verified`
    isn't truthy — the eval only scores against ground truth that a
    human signed off on.

    `only_clip_ids` filters by clip; useful for the small-N pre-push
    gate subset."""
    rows: list[AudioFixtureRow] = []
    seen: set[tuple[str, int]] = set()  # de-dup if a row was appended twice
    for raw in iter_utt_rows(target):
        if only_verified and not raw.get("human_verified"):
            continue
        clip_id = str(raw.get("clip_id") or "")
        if only_clip_ids and clip_id not in only_clip_ids:
            continue
        try:
            utt_index = int(raw["utt_index"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (clip_id, utt_index)
        if key in seen:
            continue
        seen.add(key)
        verdict = str(raw.get("expected_verdict") or "")
        if verdict not in VERDICTS:
            continue
        rows.append(
            AudioFixtureRow(
                clip_id=clip_id,
                utt_index=utt_index,
                speaker_role=str(raw.get("speaker_role") or "unknown"),
                facility=_opt_str(raw.get("facility")),
                position=_opt_str(raw.get("position")),
                event_tag=_opt_str(raw.get("event_tag")),
                transcript_text=str(raw.get("transcript_text") or ""),
                transcript_seed=str(raw.get("transcript_seed") or ""),
                expected_verdict=verdict,
                expected_section=_normalize_section(raw.get("expected_section")),
                expected_phraseology=_opt_str(raw.get("expected_phraseology")),
                mismatch=_opt_str(raw.get("mismatch")),
            )
        )
    rows.sort(key=lambda r: (r.clip_id, r.utt_index))
    return tuple(rows)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_section(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip().lstrip("§").replace("−", "-").replace("–", "-")  # noqa: RUF001
    return stripped or None


# ---------- WER ----------

_TOKEN_SPLIT = re.compile(r"\s+")
_NON_ALNUM_EDGE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")


def _normalize_for_wer(text: str) -> list[str]:
    """Lowercase + edge-strip punctuation + whitespace-tokenize. WER is
    a coarse measure on the best of days; we don't want quote-vs-no-
    quote or comma placement to inflate it. Internal punctuation
    (don't, F.A.A.) stays in-token so apostrophe-bearing words still
    line up."""
    lowered = text.strip().lower()
    if not lowered:
        return []
    tokens: list[str] = []
    for raw in _TOKEN_SPLIT.split(lowered):
        cleaned = _NON_ALNUM_EDGE.sub("", raw)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def compute_wer(reference: str, hypothesis: str) -> float:
    """Token-level word error rate.

    `reference` = human-corrected transcript_text; `hypothesis` =
    whisper transcript_seed. Returns a float in [0, ∞) — values > 1
    are valid (a hypothesis longer than the reference can have more
    insertions than the reference has words).

    Empty reference + non-empty hypothesis → ∞ would be technically
    correct, but useless for averaging; we return 0.0 if both are
    empty (nothing to score) and the hypothesis token count if the
    reference is empty (every hypothesis token is an insertion)."""
    ref = _normalize_for_wer(reference)
    hyp = _normalize_for_wer(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return float(len(hyp))
    if not hyp:
        return 1.0
    distance = _levenshtein(ref, hyp)
    return distance / len(ref)


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Standard 2-row dynamic-programming token edit distance.
    O(len(ref) * len(hyp)) time, O(min) space."""
    if len(hyp) < len(ref):
        ref, hyp = hyp, ref
    previous = list(range(len(ref) + 1))
    for j, h_tok in enumerate(hyp, start=1):
        current = [j]
        for i, r_tok in enumerate(ref, start=1):
            cost = 0 if r_tok == h_tok else 1
            current.append(
                min(
                    current[-1] + 1,  # insertion
                    previous[i] + 1,  # deletion
                    previous[i - 1] + cost,  # substitution / match
                )
            )
        previous = current
    return previous[-1]


# ---------- per-case scoring ----------


@dataclass(frozen=True)
class AudioEvalCase:
    """One scored utterance. Both lint passes are recorded so a JSON
    consumer can examine where the verdict shifted."""

    row: AudioFixtureRow
    clean_actual: PhraseologyVerdict
    noisy_actual: PhraseologyVerdict
    wer: float

    @property
    def clean_verdict_pass(self) -> bool:
        return self.clean_actual.verdict == self.row.expected_verdict

    @property
    def noisy_verdict_pass(self) -> bool:
        return self.noisy_actual.verdict == self.row.expected_verdict

    @property
    def clean_citation_pass(self) -> bool:
        return _citation_match(self.row, self.clean_actual)

    @property
    def noisy_citation_pass(self) -> bool:
        return _citation_match(self.row, self.noisy_actual)

    @property
    def clean_passed(self) -> bool:
        return self.clean_verdict_pass and self.clean_citation_pass

    @property
    def noisy_passed(self) -> bool:
        return self.noisy_verdict_pass and self.noisy_citation_pass

    @property
    def verdict_shift(self) -> bool:
        """True when the clean pass was correct but STT noise broke
        the verdict. The headline metric for "how much accuracy do we
        lose to STT" — every shift is a case the lint tool would have
        gotten right with a perfect transcriber."""
        return self.clean_verdict_pass and not self.noisy_verdict_pass


def _citation_match(row: AudioFixtureRow, actual: PhraseologyVerdict) -> bool:
    if row.expected_section is None:
        return actual.expected_section is None
    return actual.expected_section == row.expected_section


# ---------- aggregate ----------


@dataclass(frozen=True)
class AudioEvalResult:
    cases: tuple[AudioEvalCase, ...]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def mean_wer(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.wer for c in self.cases) / len(self.cases)

    def _accuracy(self, predicate: Callable[[AudioEvalCase], bool]) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if predicate(c)) / len(self.cases)

    @property
    def clean_verdict_accuracy(self) -> float:
        return self._accuracy(lambda c: c.clean_verdict_pass)

    @property
    def noisy_verdict_accuracy(self) -> float:
        return self._accuracy(lambda c: c.noisy_verdict_pass)

    @property
    def clean_citation_accuracy(self) -> float:
        return self._accuracy(lambda c: c.clean_citation_pass)

    @property
    def noisy_citation_accuracy(self) -> float:
        return self._accuracy(lambda c: c.noisy_citation_pass)

    @property
    def clean_combined_accuracy(self) -> float:
        return self._accuracy(lambda c: c.clean_passed)

    @property
    def noisy_combined_accuracy(self) -> float:
        return self._accuracy(lambda c: c.noisy_passed)

    @property
    def verdict_shift_rate(self) -> float:
        return self._accuracy(lambda c: c.verdict_shift)

    def by_event_tag(self) -> dict[str, dict[str, float]]:
        """Per-event-tag {clean/noisy verdict + combined acc, count}.
        Spots whether STT failure modes cluster on specific event
        classes (e.g. emergency phraseology cratering while routine
        departure holds)."""
        buckets: dict[str, list[AudioEvalCase]] = {}
        for case in self.cases:
            tag = case.row.event_tag or "(none)"
            buckets.setdefault(tag, []).append(case)
        out: dict[str, dict[str, float]] = {}
        for tag, cases in buckets.items():
            n = len(cases)
            out[tag] = {
                "clean_verdict_accuracy": sum(1 for c in cases if c.clean_verdict_pass) / n,
                "noisy_verdict_accuracy": sum(1 for c in cases if c.noisy_verdict_pass) / n,
                "clean_combined_accuracy": sum(1 for c in cases if c.clean_passed) / n,
                "noisy_combined_accuracy": sum(1 for c in cases if c.noisy_passed) / n,
                "mean_wer": sum(c.wer for c in cases) / n,
                "count": float(n),
            }
        return out


def run_audio_eval(
    fixture: Sequence[AudioFixtureRow],
    lint_fn: LintFn,
    *,
    skip_noisy: bool = False,
) -> AudioEvalResult:
    """Run every fixture row through the lint pipeline twice — once
    on the clean transcript, once on the whisper seed — and aggregate
    the scoring.

    `skip_noisy=True` halves model load when iterating on the lint
    pipeline: the noisy pass is replaced with a stub OOS verdict and
    WER comes back zero. The pre-push gate runs the real two-pass
    flow; only inner-loop iteration uses the skip."""
    cases: list[AudioEvalCase] = []
    for row in fixture:
        scenario_hint = row.event_tag
        clean = lint_fn(row.transcript_text, scenario_hint)
        if skip_noisy:
            noisy = _stub_oos()
            wer = 0.0
        else:
            noisy = lint_fn(row.transcript_seed, scenario_hint)
            wer = compute_wer(row.transcript_text, row.transcript_seed)
        cases.append(AudioEvalCase(row=row, clean_actual=clean, noisy_actual=noisy, wer=wer))
    return AudioEvalResult(cases=tuple(cases))


def _stub_oos() -> PhraseologyVerdict:
    """OOS-verdict stand-in for skipped noisy passes. Same shape as
    `lint_utterance`'s pre-model OOS short-circuit, so JSON consumers
    see a consistent envelope whether the pass ran or was skipped."""
    return PhraseologyVerdict(
        verdict="out_of_scope",
        expected_section=None,
        expected_phraseology=None,
        mismatch=None,
        citation_quote=None,
    )


# ---------- defaults ----------


def default_target_dir(character_path: Path) -> Path:
    """`character/<name>/atc_audio` — the same dir scripts/atc_audio_*.py
    write into."""
    return character_path / "atc_audio"


def default_baseline_path(character_path: Path) -> Path:
    """`character/<name>/atc_audio_baseline.json`."""
    return character_path / "atc_audio_baseline.json"


# ---------- baseline diff ----------


def load_baseline(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


@dataclass(frozen=True)
class CasePassDelta:
    """Per-case pass movement between baseline and current run. The
    pass dimensions split clean/noisy so a STT-only regression (whisper
    got worse on the same audio) is distinguishable from a lint-tool
    regression (clean pass dropped on the same human transcript)."""

    case_id: str
    old_clean_verdict_pass: bool
    new_clean_verdict_pass: bool
    old_noisy_verdict_pass: bool
    new_noisy_verdict_pass: bool
    old_clean_citation_pass: bool
    new_clean_citation_pass: bool
    old_noisy_citation_pass: bool
    new_noisy_citation_pass: bool

    @property
    def is_regression(self) -> bool:
        if self.old_clean_verdict_pass and not self.new_clean_verdict_pass:
            return True
        if self.old_noisy_verdict_pass and not self.new_noisy_verdict_pass:
            return True
        if self.old_clean_citation_pass and not self.new_clean_citation_pass:
            return True
        return self.old_noisy_citation_pass and not self.new_noisy_citation_pass

    @property
    def is_improvement(self) -> bool:
        if not self.old_clean_verdict_pass and self.new_clean_verdict_pass:
            return True
        if not self.old_noisy_verdict_pass and self.new_noisy_verdict_pass:
            return True
        if not self.old_clean_citation_pass and self.new_clean_citation_pass:
            return True
        return not self.old_noisy_citation_pass and self.new_noisy_citation_pass


@dataclass(frozen=True)
class AggregateMetricDelta:
    metric: str
    old: float
    new: float

    @property
    def is_regression(self) -> bool:
        # WER is a where-lower-is-better metric; the others are
        # higher-is-better. Sign rule lives at this layer so the
        # printer + gate use the same definition.
        if self.metric == "mean_wer":
            return self.new > self.old
        return self.new < self.old


@dataclass(frozen=True)
class BaselineComparison:
    case_deltas: tuple[CasePassDelta, ...]
    aggregate_deltas: tuple[AggregateMetricDelta, ...]
    new_cases: tuple[str, ...]
    dropped_cases: tuple[str, ...]

    @property
    def case_regressions(self) -> tuple[CasePassDelta, ...]:
        return tuple(d for d in self.case_deltas if d.is_regression)

    @property
    def case_improvements(self) -> tuple[CasePassDelta, ...]:
        return tuple(d for d in self.case_deltas if d.is_improvement)

    @property
    def aggregate_regressions(self) -> tuple[AggregateMetricDelta, ...]:
        return tuple(d for d in self.aggregate_deltas if d.is_regression)

    def has_regression(self, *, regression_budget: int = 0) -> bool:
        if self.aggregate_regressions:
            return True
        return len(self.case_regressions) > regression_budget


_AGGREGATE_METRICS: tuple[str, ...] = (
    "clean_verdict_accuracy",
    "noisy_verdict_accuracy",
    "clean_citation_accuracy",
    "noisy_citation_accuracy",
    "clean_combined_accuracy",
    "noisy_combined_accuracy",
    "mean_wer",
)


def _bool_field(case: Mapping[str, Any], key: str) -> bool:
    return bool(case.get(key, False))


def compare_baselines(
    baseline: Mapping[str, Any],
    result: AudioEvalResult,
) -> BaselineComparison:
    """Diff a saved baseline JSON against a fresh eval result. Cases
    join on `case_id` (= clip_id:utt_index padded). Aggregate metrics
    join by name; missing metrics in an older baseline are skipped
    rather than failing the gate."""
    base_cases_raw = baseline.get("cases", []) or []
    base_cases: dict[str, Mapping[str, Any]] = {
        str(c["case_id"]): c for c in base_cases_raw if "case_id" in c
    }
    new_cases_map: dict[str, AudioEvalCase] = {c.row.case_id: c for c in result.cases}

    shared_ids = sorted(base_cases.keys() & new_cases_map.keys())
    case_deltas = tuple(
        CasePassDelta(
            case_id=case_id,
            old_clean_verdict_pass=_bool_field(base_cases[case_id], "clean_verdict_pass"),
            new_clean_verdict_pass=new_cases_map[case_id].clean_verdict_pass,
            old_noisy_verdict_pass=_bool_field(base_cases[case_id], "noisy_verdict_pass"),
            new_noisy_verdict_pass=new_cases_map[case_id].noisy_verdict_pass,
            old_clean_citation_pass=_bool_field(base_cases[case_id], "clean_citation_pass"),
            new_clean_citation_pass=new_cases_map[case_id].clean_citation_pass,
            old_noisy_citation_pass=_bool_field(base_cases[case_id], "noisy_citation_pass"),
            new_noisy_citation_pass=new_cases_map[case_id].noisy_citation_pass,
        )
        for case_id in shared_ids
    )

    new_metrics: dict[str, float] = {
        "clean_verdict_accuracy": result.clean_verdict_accuracy,
        "noisy_verdict_accuracy": result.noisy_verdict_accuracy,
        "clean_citation_accuracy": result.clean_citation_accuracy,
        "noisy_citation_accuracy": result.noisy_citation_accuracy,
        "clean_combined_accuracy": result.clean_combined_accuracy,
        "noisy_combined_accuracy": result.noisy_combined_accuracy,
        "mean_wer": result.mean_wer,
    }
    aggregate_deltas: list[AggregateMetricDelta] = []
    for metric in _AGGREGATE_METRICS:
        old = baseline.get(metric)
        if old is None:
            continue
        aggregate_deltas.append(
            AggregateMetricDelta(metric=metric, old=float(old), new=new_metrics[metric])
        )

    return BaselineComparison(
        case_deltas=case_deltas,
        aggregate_deltas=tuple(aggregate_deltas),
        new_cases=tuple(sorted(new_cases_map.keys() - base_cases.keys())),
        dropped_cases=tuple(sorted(base_cases.keys() - new_cases_map.keys())),
    )
