"""Structural validator for the voice ablation manifest (plan #8 /
harness-wrzz).

The ablation manifest at `character/<name>/voice/ablation.yaml` is an
exclusion list: each entry names a canonical voice sample to hide
from the VoiceRetriever during `eval atc --ablate`. The score delta
vs the all-samples-visible run is the generalization signal
(originally harness-w49p; renamed from "holdout" 2026-04-26).

This module guards three structural drift modes:

1. **ID drift** — an ablation entry names a sample ID that no longer
   exists in canonical.yaml. The eval would silently exclude nothing
   for that ID; the gap measurement would be on a smaller-than-named
   set without anyone noticing.
2. **Gold drift** — the canonical sample's gold body changed, but
   the ablation entry's gold is the old string. The eval still works
   (only the ID is used at retrieval time) but the manifest's
   selection-rationale comments now refer to text that doesn't match
   what's actually deployed. Future reviewers can't trust the
   rationale.
3. **Prompt drift** — same as gold drift but for the prompt field.
   Prompts are what voice retrieval keys on; if the prompt diverges,
   the entry's stated reason for ablation is no longer accurate.

The validator is pure-function — no MLX, no embedder, no eval run.
Loads canonical.yaml + ablation.yaml, emits a list of structural
problems (empty list = clean). Pre-push hooks invoke the CLI wrapper
in `scripts/check_ablation_manifest.py`. Manual measurement work
(stock vs ablate vs round-robin) lives in
`scripts/ablation_check.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class AblationProblemKind(StrEnum):
    """Categories of structural drift the validator detects."""

    MISSING_ID = "missing_id"  # Ablation references an ID not in canonical.
    GOLD_DRIFT = "gold_drift"  # Canonical gold differs from ablation gold.
    PROMPT_DRIFT = "prompt_drift"  # Canonical prompt differs from ablation prompt.
    PARSE_ERROR = "parse_error"  # Manifest YAML failed to parse cleanly.
    EMPTY_MANIFEST = "empty_manifest"  # File exists but no samples listed.


@dataclass(frozen=True)
class AblationProblem:
    """One drift finding. `sample_id` is empty for parse errors that
    happen before any samples are reachable."""

    kind: AblationProblemKind
    sample_id: str
    message: str


def _read_yaml_samples(path: Path) -> tuple[list[dict[str, object]], str | None]:
    """Read a voice YAML and return (samples_list, error_message).
    Returns ([], None) when the file is absent (caller handles
    "no manifest = no problems"). Returns ([], err) on a parse error."""
    if not path.exists():
        return [], None
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        return [], f"YAML parse error in {path.name}: {exc}"
    if not isinstance(doc, dict):
        return [], f"{path.name} must be a YAML mapping at the top level"
    samples_raw = doc.get("samples") or []
    if not isinstance(samples_raw, list):
        return [], f"{path.name}.samples must be a list"
    out: list[dict[str, object]] = []
    for raw in samples_raw:
        if isinstance(raw, dict):
            out.append(raw)
    return out, None


def validate_ablation(
    *,
    canonical_path: Path,
    ablation_path: Path,
) -> list[AblationProblem]:
    """Cross-check the ablation manifest against canonical voice
    samples. Returns a list of structural problems; empty list means
    the manifest is clean.

    No-manifest (file absent) is fine — the eval just runs without
    ablation. Empty-manifest (file present but `samples:` is empty
    or missing) is treated as a problem because the file's mere
    existence implies intent."""
    problems: list[AblationProblem] = []

    if not ablation_path.exists():
        return problems  # No manifest, no problems.

    canonical_samples_raw, canonical_err = _read_yaml_samples(canonical_path)
    if canonical_err is not None:
        return [
            AblationProblem(
                kind=AblationProblemKind.PARSE_ERROR,
                sample_id="",
                message=canonical_err,
            )
        ]

    ablation_samples_raw, ablation_err = _read_yaml_samples(ablation_path)
    if ablation_err is not None:
        return [
            AblationProblem(
                kind=AblationProblemKind.PARSE_ERROR,
                sample_id="",
                message=ablation_err,
            )
        ]

    if not ablation_samples_raw:
        problems.append(
            AblationProblem(
                kind=AblationProblemKind.EMPTY_MANIFEST,
                sample_id="",
                message=(
                    f"{ablation_path.name} parses but lists zero samples. "
                    "Either add an entry or remove the file (no-manifest "
                    "is the correct shape when there's nothing to ablate)."
                ),
            )
        )
        return problems

    canonical_by_id: dict[str, dict[str, object]] = {
        str(s["id"]): s for s in canonical_samples_raw if "id" in s and isinstance(s["id"], str)
    }

    for entry in ablation_samples_raw:
        sample_id = str(entry.get("id", "")).strip()
        if not sample_id:
            problems.append(
                AblationProblem(
                    kind=AblationProblemKind.MISSING_ID,
                    sample_id="",
                    message=(
                        f"{ablation_path.name} entry without an 'id' field — "
                        "every entry must name the canonical sample it ablates."
                    ),
                )
            )
            continue

        canonical = canonical_by_id.get(sample_id)
        if canonical is None:
            problems.append(
                AblationProblem(
                    kind=AblationProblemKind.MISSING_ID,
                    sample_id=sample_id,
                    message=(
                        f"{ablation_path.name} ablates id '{sample_id}' but "
                        f"that ID does not exist in {canonical_path.name}. "
                        "Either rename / restore the canonical entry or "
                        "drop the ablation entry."
                    ),
                )
            )
            continue

        # Gold + prompt drift: surface if the canonical's value differs
        # from the manifest's. Both are stripped before compare since
        # the YAML loader preserves YAML-block trailing newlines we
        # don't care about.
        for field in ("prompt", "gold"):
            canonical_value = str(canonical.get(field, "")).strip()
            ablation_value = str(entry.get(field, "")).strip()
            if not ablation_value:
                # The manifest doesn't have to carry prompt/gold; some
                # entries reference by ID alone. Skip drift check when
                # the manifest field is absent.
                continue
            if canonical_value != ablation_value:
                kind = (
                    AblationProblemKind.GOLD_DRIFT
                    if field == "gold"
                    else AblationProblemKind.PROMPT_DRIFT
                )
                problems.append(
                    AblationProblem(
                        kind=kind,
                        sample_id=sample_id,
                        message=(
                            f"{ablation_path.name} entry '{sample_id}' has a "
                            f"'{field}' that no longer matches "
                            f"{canonical_path.name}. Update the manifest "
                            "entry to match the current canonical text "
                            "(or drop the field — entries can reference by "
                            "ID alone)."
                        ),
                    )
                )

    return problems


def format_problems(problems: Iterable[AblationProblem]) -> str:
    """Human-readable report. Empty input yields the empty string."""
    blocks: list[str] = []
    for p in problems:
        header = f"[{p.kind.value}]"
        if p.sample_id:
            header += f" {p.sample_id}"
        blocks.append(f"{header}: {p.message}")
    return "\n".join(blocks)


# Re-exported for the script's compatibility check — keeps callers
# off the private VoiceSample import path.
def collect_canonical_ids(canonical_path: Path) -> frozenset[str]:
    """Set of every canonical sample ID. Used by the measurement
    script (`scripts/ablation_check.py`) for the round-robin probe
    walk: every canonical ID becomes one ablation run."""
    samples_raw, err = _read_yaml_samples(canonical_path)
    if err is not None:
        return frozenset()
    out: set[str] = set()
    for s in samples_raw:
        sid = s.get("id")
        if isinstance(sid, str) and sid:
            out.add(sid)
    return frozenset(out)


__all__ = [
    "AblationProblem",
    "AblationProblemKind",
    "collect_canonical_ids",
    "format_problems",
    "validate_ablation",
]
