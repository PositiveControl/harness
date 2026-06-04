"""Trust-measurement report for advisory vision-QA (harness-ke4hx.5).

Reads one or more ``.harness/vision_qa.jsonl`` logs (written by the drive
loop's advisory QA pass, harness-ke4hx.4) and reports how the VLM's
PASS/FAIL verdict lines up with the real verify-gate outcome. This is the
data that decides whether the signal is trustworthy enough to GATE on in
Phase 2 — or whether it stays advisory / gets dropped.

HONEST FRAMING — the existing gate is the only label we have, and it is
NOT a perfect oracle: the whole point of vision-QA is to catch the class
the gate is blind to (render-without-interaction, harness-u1il5). So:

  - "agreement" = how often the VLM verdict matches the gate. High
    agreement = the VLM is at least as good as the gate on what the gate
    can see.
  - VLM=FAIL while gate=PASS is NOT necessarily a false alarm — it's the
    target signal (a candidate catch the gate missed). Reported as
    "divergence: VLM flags, gate passed" — eyeball these to judge whether
    the VLM is finding real problems or hallucinating.
  - VLM=PASS while gate=FAIL is a VLM blind spot (it missed a failure the
    gate caught) — these are the disqualifying cases for gating.

precision/recall are computed treating gate-FAIL as ground-truth-positive,
but read them with the caveat above — a "false positive" against the gate
may be a true catch.

Usage:
    uv run python scripts/vision_qa_report.py [PATH ...] [--json]

PATH defaults to ``.harness/vision_qa.jsonl`` under the CWD. Pass explicit
paths (or shell globs) to aggregate across multiple drives:
    uv run python scripts/vision_qa_report.py .harness/loop_runs/*/vision_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Report:
    total: int = 0
    parse_errors: int = 0
    # verdict counts (pass / fail / unknown)
    verdicts: dict[str, int] = field(default_factory=dict)
    gate_pass: int = 0
    gate_fail: int = 0
    # confusion against gate-FAIL = positive, excluding unknown verdicts
    tp: int = 0  # VLM fail, gate fail
    fp: int = 0  # VLM fail, gate pass  (candidate catch OR false alarm)
    fn: int = 0  # VLM pass, gate fail  (VLM blind spot — disqualifying)
    tn: int = 0  # VLM pass, gate pass
    unknown_excluded: int = 0
    # rows the operator should eyeball
    flags_gate_passed: list[dict[str, str]] = field(default_factory=list)
    blind_spots: list[dict[str, str]] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        """Fraction of classifiable (non-unknown) records where the VLM
        verdict matched the gate. 0.0 when nothing classifiable."""
        classifiable = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / classifiable if classifiable else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


def summarize(records: list[dict[str, object]]) -> Report:
    """Aggregate advisory-QA records into a Report. Tolerates missing /
    malformed fields per record (counts them, never raises)."""
    rep = Report()
    verdicts: Counter[str] = Counter()
    for rec in records:
        verdict = rec.get("verdict")
        gate = rec.get("gate_passed")
        if not isinstance(verdict, str) or not isinstance(gate, bool):
            rep.parse_errors += 1
            continue
        rep.total += 1
        verdicts[verdict] += 1
        if gate:
            rep.gate_pass += 1
        else:
            rep.gate_fail += 1
        if verdict == "unknown":
            rep.unknown_excluded += 1
            continue
        vlm_fail = verdict == "fail"
        if vlm_fail and not gate:
            rep.tp += 1
        elif vlm_fail and gate:
            rep.fp += 1
            rep.flags_gate_passed.append(_row(rec))
        elif not vlm_fail and not gate:
            rep.fn += 1
            rep.blind_spots.append(_row(rec))
        else:
            rep.tn += 1
    rep.verdicts = dict(verdicts)
    return rep


def _row(rec: dict[str, object]) -> dict[str, str]:
    return {
        "issue_id": str(rec.get("issue_id", "?")),
        "reason": str(rec.get("reason", "")),
        "shot": str(rec.get("shot", "")),
    }


def load_records(paths: list[Path]) -> tuple[list[dict[str, object]], list[str]]:
    """Read JSONL records from each path. Returns (records, warnings).
    Missing files and unparseable lines warn rather than abort."""
    records: list[dict[str, object]] = []
    warnings: list[str] = []
    for path in paths:
        if not path.is_file():
            warnings.append(f"no such file: {path}")
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"{path}:{i}: unparseable JSON line")
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records, warnings


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_text(rep: Report) -> str:
    lines: list[str] = []
    lines.append(f"records: {rep.total}  (parse errors: {rep.parse_errors})")
    if rep.total == 0:
        lines.append("no advisory-QA data yet — run a drive with HARNESS_VISION_BASE_URL set.")
        return "\n".join(lines)
    v = rep.verdicts
    lines.append(
        f"verdicts: PASS={v.get('pass', 0)}  FAIL={v.get('fail', 0)}  UNKNOWN={v.get('unknown', 0)}"
    )
    lines.append(f"gate:     PASS={rep.gate_pass}  FAIL={rep.gate_fail}")
    lines.append("")
    lines.append("confusion (gate-FAIL = positive; unknown verdicts excluded):")
    lines.append(f"  TP  VLM=fail gate=fail : {rep.tp}")
    lines.append(f"  FP  VLM=fail gate=pass : {rep.fp}  (candidate catches OR false alarms)")
    lines.append(f"  FN  VLM=pass gate=fail : {rep.fn}  (VLM blind spots — disqualifying)")
    lines.append(f"  TN  VLM=pass gate=pass : {rep.tn}")
    lines.append(f"  excluded (unknown)     : {rep.unknown_excluded}")
    lines.append("")
    lines.append(
        f"agreement: {_pct(rep.agreement)}   "
        f"precision(vs gate): {_pct(rep.precision)}   recall(vs gate): {_pct(rep.recall)}"
    )
    lines.append("  (gate is not an oracle — a 'false positive' may be a real catch it missed)")
    if rep.blind_spots:
        lines.append("")
        lines.append(f"VLM BLIND SPOTS (VLM passed, gate failed) — {len(rep.blind_spots)}:")
        for r in rep.blind_spots[:10]:
            lines.append(f"  {r['issue_id']}: {r['reason']}")
    if rep.flags_gate_passed:
        lines.append("")
        lines.append(
            f"DIVERGENCE: VLM flagged, gate passed — {len(rep.flags_gate_passed)} "
            "(eyeball: real catches or hallucinations?):"
        )
        for r in rep.flags_gate_passed[:10]:
            lines.append(f"  {r['issue_id']}: {r['reason']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vision-QA trust report (harness-ke4hx.5)")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[".harness/vision_qa.jsonl"],
        help="JSONL log path(s); default .harness/vision_qa.jsonl",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    records, warnings = load_records([Path(p) for p in args.paths])
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    rep = summarize(records)

    if args.json:
        payload = asdict(rep)
        payload["agreement"] = rep.agreement
        payload["precision"] = rep.precision
        payload["recall"] = rep.recall
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
