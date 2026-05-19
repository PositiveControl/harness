"""Persistent-memory ops (remember / memories / forget) for BeadsAdapter.

Extracted from bd_adapter.py (harness-vhoj). Wraps `bd remember`,
`bd memories`, `bd forget` — persistent cross-session knowledge that
bd stores keyed by insight / query.
"""

from __future__ import annotations

import json
import warnings

from harness.store._bd_runner import BeadsAdapterError, BeadsRunner


class BeadsMemoryMixin(BeadsRunner):
    def remember(self, insight: str) -> None:
        self._run(["remember", insight])

    def memories(self, query: str = "") -> str:
        args = ["memories"]
        if query:
            args.append(query)
        result = self._run(args)
        return result.stdout

    def memories_json(self, query: str = "") -> dict[str, str]:
        """Return `bd memories --json` parsed as a `{key: body}` mapping.

        Used by the bd-memory harvester (harness-9yd) to feed episodic
        ingestion with structured input rather than parsing the text
        layout. Empty store is `{}` — bd emits that verbatim. Any
        parse failure surfaces as BeadsAdapterError so harvester
        callers can warn-and-skip rather than crash."""
        args = ["memories", "--json"]
        if query:
            args.append(query)
        result = self._run(args)
        stdout = (result.stdout or "").strip()
        if not stdout:
            return {}
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise BeadsAdapterError(
                f"bd memories --json returned non-JSON stdout: {stdout[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise BeadsAdapterError(
                f"bd memories --json expected a dict, got {type(parsed).__name__}"
            )
        out: dict[str, str] = {}
        for key, value in parsed.items():
            if not isinstance(key, str):
                warnings.warn(
                    f"bd memories --json entry has non-string key "
                    f"({type(key).__name__}); skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if not isinstance(value, str):
                # Metadata sidecar keys (schema_version: int, future
                # last_modified/source_version) ride alongside the
                # memory dict. Skip them so the harvest survives.
                continue
            out[key] = value
        return out

    def forget(self, key: str) -> None:
        """Wrap `bd forget <key>`. Removes a persistent memory by key.
        Empty key raises ValueError so the caller's bad arg surfaces
        before spawning a doomed subprocess."""
        if not key.strip():
            raise ValueError("forget key must be non-empty")
        self._run(["forget", key])
