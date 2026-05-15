#!/usr/bin/env python3
"""Fetch the live FAA TFR list (harness-3jz1.2 helper).

Pulls `https://tfr.faa.gov/tfrapi/exportTfrList` — the JSON list backing
the SPA's Download button — and writes the raw response to stdout or
to a file. Each entry is metadata-only (`notam_id`, `type`, `facility`,
`state`, `description`, `creation_date`); the full NOTAM prose with
geometry/altitudes/citations lives behind a separate detail endpoint
that isn't pinned in this script. Use this output as the seed list for
hand-curating `tfr_eval.yaml` rows, or as the input to a future v2
live-feed pipeline (harness-3jz1.8).

Usage:
    python scripts/fetch_tfr_list.py                  # dump to stdout
    python scripts/fetch_tfr_list.py -o tfr_list.json # write to file

Stdlib only — no harness deps so this stays usable from a fresh
checkout without `uv sync`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

TFR_LIST_URL = "https://tfr.faa.gov/tfrapi/exportTfrList"


def fetch_tfr_list(url: str = TFR_LIST_URL, *, timeout_s: float = 15.0) -> list[dict[str, str]]:
    """Return the list of active TFR metadata records.

    Raises `RuntimeError` on HTTP errors or non-JSON responses.
    """
    # HTTPS-only, hardcoded FAA host. urllib's Request + urlopen is fine
    # for a stdlib-only fetcher; S310 fires because urllib *can* open
    # file:// or ftp:// URLs in principle. We accept the default URL and
    # any --url override the operator passes; this is a script, not a
    # service handling untrusted input.
    req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310
            status = response.status
            body = response.read()
    except OSError as exc:
        raise RuntimeError(f"fetch failed: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"unexpected status {status} from {url}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"response was not JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"expected JSON list, got {type(payload).__name__}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the live FAA TFR list.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write JSON to this file (default: stdout).",
    )
    parser.add_argument(
        "--url",
        default=TFR_LIST_URL,
        help=f"Override the list URL (default: {TFR_LIST_URL}).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a per-type count summary instead of the full JSON.",
    )
    args = parser.parse_args(argv)

    try:
        records = fetch_tfr_list(args.url)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.summary:
        counts: dict[str, int] = {}
        for rec in records:
            counts[rec.get("type", "?")] = counts.get(rec.get("type", "?"), 0) + 1
        for type_name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {type_name:<24} {count}")
        print(f"  {'TOTAL':<24} {len(records)}")
        return 0

    out_text = json.dumps(records, indent=2)
    if args.output is not None:
        args.output.write_text(out_text + "\n")
    else:
        sys.stdout.write(out_text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
