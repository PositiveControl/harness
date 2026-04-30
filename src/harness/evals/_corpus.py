"""Shared YAML-fixture scaffolding for corpus-style evals.

The atc / phraseology evals (and any future corpus-grounded eval —
legal-citation, RFC compliance, etc.) load a YAML file shaped as
either a top-level list of cases or a `{cases: [...]}` mapping, then
iterate per-row pulling required + optional string fields with
fixture-author-friendly error messages.

That envelope is identical across evals; only the per-row dataclass
+ scoring logic differs. This module owns just the envelope:

  - `load_yaml_cases(path, eval_name)` — parse + validate the
    top-level shape, return the per-row mapping list. Empty file →
    empty list.
  - `str_field(entry, key, path, idx)` — required string extractor
    with strip, raises ValueError pointing at the offending row.
  - `opt_str_field(entry, key)` — optional-string extractor;
    returns None for missing / null / non-string / blank-after-strip.

The evals/router.py + evals/session_resume.py modules already use a
similar pattern but their case dataclasses are too divergent to share
a row-level loader; they stay independent.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml_cases(path: Path, eval_name: str) -> list[dict[str, Any]]:
    """Parse a corpus-eval YAML file and return its per-case mappings.

    Accepts two top-level shapes:
      - bare list:           `- id: …`
      - mapping with cases:  `cases:\\n  - id: …`

    Empty / missing / null-bodied files return `[]` rather than raise
    — a fixture-in-progress should not crash the eval. The eval's own
    runner decides what to do with an empty case set.

    Per-row entries must be mappings; non-mapping entries raise
    ValueError pointing at the offending index. `eval_name` is woven
    into the top-level error message ('phraseology eval fixture …
    is not a list (or {cases: [...]})') so a fixture author sees
    which eval objected.
    """
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("cases") or []
    if not isinstance(raw, list):
        raise ValueError(f"{eval_name} eval fixture {path} is not a list (or {{cases: [...]}})")
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}] is not a mapping")
        out.append(entry)
    return out


def str_field(entry: Mapping[str, Any], key: str, path: Path, idx: int) -> str:
    """Extract a required string field from a case row.

    Strips surrounding whitespace. Raises ValueError naming the path,
    row index, and field key when the value is missing, null, not a
    string, or blank after strip — same error shape every eval has
    used independently up to now.
    """
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}[{idx}] missing or empty {key!r}")
    return value.strip()


def opt_str_field(entry: Mapping[str, Any], key: str) -> str | None:
    """Extract an optional string field. None when absent, null,
    non-string, or blank-after-strip; the stripped string otherwise.
    Mirrors the shape phraseology.py used independently."""
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
