"""Voice ablation measurement script (harness-wrzz, plan #8 — slow
counterpart to scripts/check_ablation_manifest.py).

Runs the full atc eval in two or more modes and emits a comparison
table + verdict. Manual-invocation only — too slow for an automatic
gate (each `harness eval atc` run takes minutes per case, tens of
cases, plus 1+N ablation probes).

Modes:
  --measure (default)
      Stock vs single-ablate. Two `harness eval atc` runs:
      one with all voice samples available, one with --ablate
      applied. Produces a per-case pass/fail diff. ~10-20 min.

  --round-robin
      Per-sample memorization probe. One ablate run per canonical
      sample using --ablate-ids <id>. Produces the round-robin
      ablation map (phase1_baseline.md run 6). ~5-10 min PER probe;
      with airton_c1's 8 canonical samples the full map is
      ~40-80 min. Run before / after a non-trivial canonical edit
      to confirm the memorization shape didn't shift.

  --compare-baseline
      Diff the just-run measurement against
      character/<name>/voice/ablation_baseline.json (the
      committed snapshot). Exits non-zero if any case's pass/fail
      flipped past the threshold (default 1 case; tunable via
      --regression-budget).

  --save-baseline
      Write the just-run measurement to
      character/<name>/voice/ablation_baseline.json for future
      --compare-baseline runs to diff against.

Usage:
    uv run python scripts/ablation_check.py --measure --save-baseline
    uv run python scripts/ablation_check.py --measure --compare-baseline
    uv run python scripts/ablation_check.py --round-robin --compare-baseline
    uv run python scripts/ablation_check.py --round-robin --save-baseline

The script reuses the structural validator from
`scripts/check_ablation_manifest.py` as a pre-flight check — if the
manifest is broken structurally, no point running the slow
measurement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.persona.ablation_validate import (  # noqa: E402
    collect_canonical_ids,
    format_problems,
    validate_ablation,
)


@dataclass
class EvalRun:
    """One `harness eval atc` invocation. `passes` is the per-case
    pass/fail mapping {case_id: bool}; `pass_rate` is the headline
    fraction. `mode` is the slot name in the report ('stock',
    'ablate', or 'rr-<sample-id>')."""

    mode: str
    passes: dict[str, bool]
    pass_rate: float
    seconds: float


@dataclass
class AblationReport:
    """Aggregate report from one ablation_check.py run."""

    character: str
    runs: list[EvalRun] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0


def _atc_eval_stock(*, character: str, env_extra: dict[str, str]) -> EvalRun:
    """Run `harness eval atc --json` once with no ablation flags."""
    return _run_eval(mode="stock", character=character, extra_args=[], env_extra=env_extra)


def _atc_eval_ablate(*, character: str, env_extra: dict[str, str]) -> EvalRun:
    """Run `harness eval atc --json --ablate` — applies the on-disk
    ablation manifest at retrieval time."""
    return _run_eval(
        mode="ablate", character=character, extra_args=["--ablate"], env_extra=env_extra
    )


def _atc_eval_round_robin_probe(
    *, character: str, sample_id: str, env_extra: dict[str, str]
) -> EvalRun:
    """Run `harness eval atc --json --ablate-ids <id>` to ablate
    exactly one canonical sample. The round-robin walks every
    canonical ID, building the per-sample memorization map."""
    return _run_eval(
        mode=f"rr-{sample_id}",
        character=character,
        extra_args=["--ablate-ids", sample_id],
        env_extra=env_extra,
    )


def _run_eval(
    *,
    mode: str,
    character: str,
    extra_args: list[str],
    env_extra: dict[str, str],
) -> EvalRun:
    """Common helper: build the env, invoke `harness eval atc`,
    parse the JSON envelope. Skips leading non-JSON noise (model
    load progress, license warnings) the same way bench_embedder.py
    does."""
    env = os.environ.copy()
    env["HARNESS_CHARACTER_NAME"] = character
    env.update(env_extra)
    cmd = [
        "uv",
        "run",
        "harness",
        "eval",
        "atc",
        "--json",
        *extra_args,
    ]
    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 — local benchmark
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"harness eval atc failed (exit {completed.returncode})")

    out = completed.stdout
    try:
        idx = out.index("{\n")
    except ValueError as exc:
        raise RuntimeError("eval atc emitted no JSON envelope") from exc
    envelope = json.loads(out[idx:])

    raw_results = envelope.get("results") or envelope.get("cases") or []
    passes: dict[str, bool] = {}
    for r in raw_results:
        case_id = str(r.get("id", "")).strip()
        if not case_id:
            continue
        # eval atc reports "pass" as a bool; tolerate string variants
        # in case the schema shifts.
        passed = r.get("pass")
        if isinstance(passed, str):
            passed = passed.lower() in ("true", "pass", "yes", "1")
        passes[case_id] = bool(passed)

    pass_rate = sum(passes.values()) / max(len(passes), 1)
    return EvalRun(mode=mode, passes=passes, pass_rate=pass_rate, seconds=elapsed)


def _load_baseline(path: Path) -> AblationReport | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    runs = [
        EvalRun(
            mode=str(r["mode"]),
            passes={str(k): bool(v) for k, v in r.get("passes", {}).items()},
            pass_rate=float(r.get("pass_rate", 0.0)),
            seconds=float(r.get("seconds", 0.0)),
        )
        for r in raw.get("runs", [])
    ]
    return AblationReport(
        character=str(raw.get("character", "")),
        runs=runs,
        started_at=float(raw.get("started_at", 0.0)),
        finished_at=float(raw.get("finished_at", 0.0)),
    )


def _save_baseline(report: AblationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2))


def _compare_runs(before: EvalRun, after: EvalRun) -> tuple[list[str], list[str]]:
    """Return (regressions, improvements) — case IDs that flipped
    pass→fail or fail→pass between the two runs."""
    regressions: list[str] = []
    improvements: list[str] = []
    for case_id in sorted(set(before.passes) | set(after.passes)):
        b = before.passes.get(case_id)
        a = after.passes.get(case_id)
        if b == a:
            continue
        if b is True and a is False:
            regressions.append(case_id)
        elif b is False and a is True:
            improvements.append(case_id)
    return regressions, improvements


def _format_report(report: AblationReport, baseline: AblationReport | None) -> str:
    lines: list[str] = []
    lines.append(f"=== ablation report: {report.character} ===")
    lines.append(f"  total seconds: {report.finished_at - report.started_at:.1f}")
    lines.append("")
    by_mode = {r.mode: r for r in report.runs}
    for mode in by_mode:
        run = by_mode[mode]
        lines.append(f"  {mode:<32} pass {run.pass_rate * 100:5.1f}%  ({run.seconds:.1f}s)")
    lines.append("")
    if "stock" in by_mode and "ablate" in by_mode:
        regressions, improvements = _compare_runs(by_mode["stock"], by_mode["ablate"])
        gap = len(regressions)
        lines.append(f"  ablation gap: {gap} case(s) flipped pass → fail under --ablate")
        if regressions:
            lines.append(f"    regressions: {', '.join(regressions)}")
        if improvements:
            lines.append(f"    improvements: {', '.join(improvements)}")
    if baseline is not None:
        lines.append("")
        lines.append("  vs baseline:")
        baseline_by_mode = {r.mode: r for r in baseline.runs}
        for mode, run in by_mode.items():
            base = baseline_by_mode.get(mode)
            if base is None:
                lines.append(f"    {mode:<32} (new run; no baseline)")
                continue
            base_regressions, _ = _compare_runs(base, run)
            lines.append(f"    {mode:<32} {len(base_regressions)} regression(s)")
            for case_id in base_regressions:
                lines.append(f"      ✗ {case_id}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--character",
        default="airton_c1",
        help="Character whose ablation map to measure.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run stock + single-ablate. Default mode when no other mode is given.",
    )
    parser.add_argument(
        "--round-robin",
        action="store_true",
        help="Per-canonical-sample probe. One ablate run per ID. Slow.",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write the run to character/<name>/voice/ablation_baseline.json.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Diff this run vs the saved baseline; exit non-zero on regression.",
    )
    parser.add_argument(
        "--regression-budget",
        type=int,
        default=0,
        help="Allow up to N flipped cases vs the baseline before failing. "
        "Default 0 (any flip fails).",
    )
    parser.add_argument(
        "--character-path",
        type=Path,
        default=None,
        help="Override character dir. Defaults to character/<character>/.",
    )
    args = parser.parse_args(argv)

    character_path = args.character_path or REPO_ROOT / "character" / args.character
    canonical = character_path / "voice" / "canonical.yaml"
    ablation = character_path / "voice" / "ablation.yaml"

    # Pre-flight: structural validation. No point running a 20-minute
    # measurement against a manifest that won't load cleanly.
    structural = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    if structural:
        sys.stderr.write(format_problems(structural) + "\n")
        sys.stderr.write(
            f"\n{len(structural)} structural problem(s). Fix the manifest "
            "before running the slow measurement.\n"
        )
        return 2

    if args.save_baseline and args.compare_baseline:
        sys.stderr.write(
            "--save-baseline and --compare-baseline are mutually exclusive: "
            "compare first to confirm no regression, then re-run with "
            "--save-baseline to snapshot.\n"
        )
        return 2

    if not args.measure and not args.round_robin:
        args.measure = True

    env_extra = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    report = AblationReport(character=args.character, started_at=time.time())

    if args.measure:
        sys.stderr.write("running stock eval...\n")
        report.runs.append(_atc_eval_stock(character=args.character, env_extra=env_extra))
        sys.stderr.write("running --ablate eval...\n")
        report.runs.append(_atc_eval_ablate(character=args.character, env_extra=env_extra))

    if args.round_robin:
        ids = sorted(collect_canonical_ids(canonical))
        sys.stderr.write(f"round-robin over {len(ids)} canonical sample(s)\n")
        # Stock run is the baseline against which each rr probe diffs.
        if not any(r.mode == "stock" for r in report.runs):
            sys.stderr.write("running stock eval...\n")
            report.runs.append(_atc_eval_stock(character=args.character, env_extra=env_extra))
        for sid in ids:
            sys.stderr.write(f"  --ablate-ids {sid}...\n")
            report.runs.append(
                _atc_eval_round_robin_probe(
                    character=args.character, sample_id=sid, env_extra=env_extra
                )
            )

    report.finished_at = time.time()

    baseline_path = character_path / "voice" / "ablation_baseline.json"
    baseline: AblationReport | None = None
    if args.compare_baseline:
        baseline = _load_baseline(baseline_path)
        if baseline is None:
            sys.stderr.write(f"no baseline at {baseline_path}; --save-baseline first.\n")
            return 2

    sys.stdout.write(_format_report(report, baseline) + "\n")

    if args.save_baseline:
        _save_baseline(report, baseline_path)
        sys.stderr.write(f"baseline written → {baseline_path}\n")

    if args.compare_baseline and baseline is not None:
        # Sum regressions across every shared mode.
        baseline_by_mode = {r.mode: r for r in baseline.runs}
        total_regressions = 0
        for run in report.runs:
            base = baseline_by_mode.get(run.mode)
            if base is None:
                continue
            regressions, _ = _compare_runs(base, run)
            total_regressions += len(regressions)
        if total_regressions > args.regression_budget:
            sys.stderr.write(
                f"\n✗ {total_regressions} regression(s) vs baseline "
                f"(budget={args.regression_budget})\n"
            )
            return 1
        sys.stderr.write(f"\n✓ no regressions ({total_regressions} ≤ {args.regression_budget})\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
