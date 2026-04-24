"""atc domain eval — replays a fixture of (question, expected citations,
expected keywords) cases and scores atc's real replies against them.

Two guardrails this locks in:
- Citation presence: each `expected_citations` string must appear
  somewhere in atc's reply (case-insensitive substring). A reply that
  paraphrases the rule without anchoring to a source fails — atc's
  constitution demands a citation.
- Keyword recall: at least `min_keyword_hits` of `expected_keywords`
  must appear (case-insensitive). Guards against a citation-shaped
  reply whose substance wandered off the topic.

This eval runs the FULL atc stack end-to-end: episodic + semantic
retrieval → system prompt → persona adapter → reply. The
`run_turn_callback` argument lets the CLI wire in a production-grade
stack, tests inject a scripted reply, and bench scripts swap the
model while reusing the scorer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

# A pluggable "run one turn" callable. Takes the user question, returns
# atc's final reply string. Wiring lives in the CLI; tests pass a
# deterministic function. Keeping the signature narrow keeps this
# module pure-Python.
RunTurn = Callable[[str], str]


@dataclass(frozen=True)
class AtcEvalCase:
    """One evaluated case: fixture input + eval scores. Immutable so
    the CLI can feed it straight into the JSON envelope."""

    id: str
    audience: str
    question: str
    # Same alternate-tuple shape as expected_keywords — each entry is
    # a tuple of acceptable citation substrings; the entry passes if
    # ANY alternate is present in the reply. A scalar YAML string
    # loads as a single-alternate tuple (backward compatible). The
    # motivating case is cancel-IFR-in-IMC where §91.155, §91.153,
    # §91.173 are all valid answers depending on angle.
    expected_citations: tuple[tuple[str, ...], ...]
    # Same shape as expected_citations — each entry is a tuple of
    # alternate phrasings. See AtcFixtureRow for rationale.
    expected_keywords: tuple[tuple[str, ...], ...]
    min_keyword_hits: int
    actual_reply: str
    # One entry per MISSING expected_citations element — each element
    # is the first-alternate of an entry that matched no alternate in
    # the reply. Citation entries that DID match are dropped from this
    # tuple, so an empty tuple == all citations present.
    missing_citations: tuple[str, ...]
    # One element per matched entry in expected_keywords — the specific
    # alternate that hit (first-match-wins within the entry) so debug
    # output shows which phrasing the model actually produced.
    matched_keywords: tuple[str, ...]

    @property
    def keyword_hits(self) -> int:
        return len(self.matched_keywords)

    @property
    def citations_pass(self) -> bool:
        return not self.missing_citations

    @property
    def keywords_pass(self) -> bool:
        return self.keyword_hits >= self.min_keyword_hits

    @property
    def passed(self) -> bool:
        return self.citations_pass and self.keywords_pass


@dataclass(frozen=True)
class AtcEvalResult:
    cases: tuple[AtcEvalCase, ...]

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    def pass_rate_by_audience(self) -> dict[str, float]:
        """Per-audience pass rate (ppl / ifr / …). Helps spot whether
        a regression favors one audience over the other."""
        buckets: dict[str, list[bool]] = {}
        for c in self.cases:
            buckets.setdefault(c.audience, []).append(c.passed)
        return {
            audience: (sum(results) / len(results)) if results else 0.0
            for audience, results in buckets.items()
        }

    def failures(self) -> tuple[AtcEvalCase, ...]:
        return tuple(c for c in self.cases if not c.passed)


@dataclass(frozen=True)
class AtcFixtureRow:
    id: str
    audience: str
    question: str
    # Same alternate-tuple shape as expected_keywords (harness-cco):
    # each entry is a tuple of acceptable citation substrings. Entry
    # matches if ANY alternate appears in the reply. Scalar YAML
    # string loads as a single-alternate tuple — backward compatible.
    expected_citations: tuple[tuple[str, ...], ...]
    # Each entry is a tuple of alternate phrasings — the keyword counts
    # as a hit if ANY alternate is present in the reply (case-
    # insensitive). A scalar string in YAML (the common case) loads as
    # a single-element tuple, so pre-lane-A fixtures continue to work
    # unchanged. Lists-of-strings in YAML let a fixture author accept
    # "1,000 feet below" / "1000 feet below" / "1,000 ft below" as the
    # same keyword without over-loosening the rubric to "1,000".
    expected_keywords: tuple[tuple[str, ...], ...]
    min_keyword_hits: int


def default_fixture_path(character_path: Path) -> Path:
    """Conventional location under character/<name>/ for the fixture.
    Kept as a function so tests, CLI, and any future benchmark agree."""
    return character_path / "atc_eval.yaml"


def load_fixture(path: Path) -> tuple[AtcFixtureRow, ...]:
    """Parse an atc-eval YAML file into validated fixture rows.

    Required keys per case: id (str), audience (str, typically 'ppl'
    or 'ifr'), question (str), expected_citations (list of str, may
    be empty). Optional: expected_keywords (list of str, default []),
    min_keyword_hits (int, default 0)."""
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return ()
    # Top-level can be a list of cases or a mapping with a `cases` key
    # (matches the stub shape atc-1 scaffolded in).
    if isinstance(raw, dict):
        raw = raw.get("cases") or []
    if not isinstance(raw, list):
        raise ValueError(f"atc eval fixture {path} is not a list (or {{cases: [...]}})")
    rows: list[AtcFixtureRow] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}] is not a mapping")
        case_id = _str_field(entry, "id", path, idx)
        audience = _str_field(entry, "audience", path, idx)
        question = _str_field(entry, "question", path, idx)
        citations = _load_alternates(
            entry.get("expected_citations") or [], path, idx, field="expected_citations"
        )
        keywords = _load_alternates(
            entry.get("expected_keywords") or [], path, idx, field="expected_keywords"
        )
        min_hits_raw = entry.get("min_keyword_hits", 0) or 0
        try:
            min_hits = int(min_hits_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}[{idx}] 'min_keyword_hits' must be an int") from exc
        if min_hits < 0:
            raise ValueError(f"{path}[{idx}] 'min_keyword_hits' must be >= 0")
        if min_hits > len(keywords):
            raise ValueError(
                f"{path}[{idx}] 'min_keyword_hits' ({min_hits}) exceeds the "
                f"number of expected_keywords ({len(keywords)})"
            )
        rows.append(
            AtcFixtureRow(
                id=case_id,
                audience=audience,
                question=question,
                expected_citations=citations,
                expected_keywords=keywords,
                min_keyword_hits=min_hits,
            )
        )
    return tuple(rows)


def _str_field(entry: dict[str, object], key: str, path: Path, idx: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}[{idx}] missing or empty {key!r}")
    return value.strip()


def _load_alternates(
    raw: object,
    path: Path,
    idx: int,
    *,
    field: str,
) -> tuple[tuple[str, ...], ...]:
    """Normalise the YAML shape for an alternate-list field (either
    `expected_keywords` or `expected_citations`) into a tuple-of-
    alternate-tuples. Each entry can be either a string (single-
    alternate, backward-compatible) or a list of strings (multiple
    alternates — any match counts as one hit). Empty alternate lists
    are rejected; they would always fail to match and indicate a
    fixture bug.

    `field` names the YAML key in error messages so fixture authors
    can find the offending entry quickly."""
    if not isinstance(raw, list):
        raise ValueError(f"{path}[{idx}] {field!r} must be a list")
    out: list[tuple[str, ...]] = []
    for kid, entry in enumerate(raw):
        if isinstance(entry, str):
            alternates: tuple[str, ...] = (entry,)
        elif isinstance(entry, list):
            alternates = tuple(str(a) for a in entry)
            if not alternates:
                raise ValueError(
                    f"{path}[{idx}].{field}[{kid}] is an empty list — "
                    "an empty alternate set can never match."
                )
        else:
            raise ValueError(
                f"{path}[{idx}].{field}[{kid}] must be str or list[str], got {type(entry).__name__}"
            )
        out.append(alternates)
    return tuple(out)


def _score_reply(
    reply: str,
    *,
    expected_citations: Sequence[tuple[str, ...]],
    expected_keywords: Sequence[tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return `(missing_citations, matched_keywords)`. Matching is
    case-insensitive substring so fixture authors don't have to worry
    about exact casing in atc's reply.

    Both `expected_citations` and `expected_keywords` are tuples of
    alternate-tuples: each entry matches if ANY of its alternates
    appears in the reply. `missing_citations` returns the first-
    alternate of each entry that matched no alternate (empty tuple =
    all present). `matched_keywords` returns the specific alternate
    that hit (first-match-wins) so debug output shows which phrasing
    the model used."""
    lower = reply.lower()
    missing: list[str] = []
    for alternates in expected_citations:
        if not any(alt.lower() in lower for alt in alternates):
            # First alternate is the canonical form — surfaces it in
            # error output for the fixture author.
            missing.append(alternates[0])
    matched: list[str] = []
    for alternates in expected_keywords:
        for alt in alternates:
            if alt.lower() in lower:
                matched.append(alt)
                break
    return tuple(missing), tuple(matched)


def run_atc_eval(
    fixture: Iterable[AtcFixtureRow],
    run_turn: RunTurn,
) -> AtcEvalResult:
    """Replay `fixture` through `run_turn` and score each case.

    Pure beyond the `run_turn` call — no filesystem, no network. CLI
    wraps this with the real persona + retrieval stack; tests pass a
    scripted callback."""
    cases: list[AtcEvalCase] = []
    for row in fixture:
        reply = run_turn(row.question)
        missing, matched = _score_reply(
            reply,
            expected_citations=row.expected_citations,
            expected_keywords=row.expected_keywords,
        )
        cases.append(
            AtcEvalCase(
                id=row.id,
                audience=row.audience,
                question=row.question,
                expected_citations=row.expected_citations,
                expected_keywords=row.expected_keywords,
                min_keyword_hits=row.min_keyword_hits,
                actual_reply=reply,
                missing_citations=missing,
                matched_keywords=matched,
            )
        )
    return AtcEvalResult(cases=tuple(cases))
