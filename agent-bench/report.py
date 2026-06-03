"""Aggregate the results store into per-framework distributions.

Reports survive any metric set: it discovers metric keys from the rows, so adding
or removing a scorer changes the report with no code change here.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import yaml
from store import ResultsStore

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="agent-bench report")
    ap.add_argument("--config", default=str(ROOT / "bench.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    store = ResultsStore(ROOT / cfg["store_db"])
    rows = store.all_rows()
    if not rows:
        print("no runs recorded yet")
        return 0

    by_fw: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_fw.setdefault(str(r["framework"]), []).append(r)

    for fw, fw_rows in sorted(by_fw.items()):
        n = len(fw_rows)
        durations = [float(r["duration_s"]) for r in fw_rows]
        exit_ok = sum(int(bool(r["exit_ok"])) for r in fw_rows)
        print(f"\n=== {fw}  (n={n}, version={fw_rows[0]['version']}) ===")
        print(f"  exit_ok      : {exit_ok}/{n}")
        print(
            f"  duration_s   : median {statistics.median(durations):.1f}  "
            f"min {min(durations):.1f}  max {max(durations):.1f}"
        )
        _report_metrics(fw_rows)
    return 0


def _report_metrics(fw_rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for r in fw_rows:
        for k in dict(r["scores"]):  # type: ignore[call-overload]
            if k not in keys:
                keys.append(k)
    n = len(fw_rows)
    for key in keys:
        vals = [dict(r["scores"]).get(key) for r in fw_rows]  # type: ignore[call-overload]
        present = [v for v in vals if v is not None]
        if all(isinstance(v, bool) for v in present):
            true_n = sum(1 for v in present if v)
            print(f"  {key:<20}: {true_n}/{n} true")
        elif all(isinstance(v, (int, float)) for v in present):
            nums = [float(v) for v in present]  # type: ignore[arg-type]
            print(
                f"  {key:<20}: median {statistics.median(nums):.2f}  "
                f"min {min(nums):.2f}  max {max(nums):.2f}"
            )
        else:
            print(f"  {key:<20}: {present[:3]}{' ...' if len(present) > 3 else ''}")


if __name__ == "__main__":
    raise SystemExit(main())
