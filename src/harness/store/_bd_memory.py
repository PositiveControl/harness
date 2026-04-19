"""Persistent-memory ops (remember / memories / forget) for BeadsAdapter.

Extracted from bd_adapter.py (harness-vhoj). Wraps `bd remember`,
`bd memories`, `bd forget` — persistent cross-session knowledge that
bd stores keyed by insight / query.
"""

from __future__ import annotations

from harness.store._bd_runner import BeadsRunner


class BeadsMemoryMixin(BeadsRunner):
    def remember(self, insight: str) -> None:
        self._run(["remember", insight])

    def memories(self, query: str = "") -> str:
        args = ["memories"]
        if query:
            args.append(query)
        result = self._run(args)
        return result.stdout

    def forget(self, key: str) -> None:
        """Wrap `bd forget <key>`. Removes a persistent memory by key.
        Empty key raises ValueError so the caller's bad arg surfaces
        before spawning a doomed subprocess."""
        if not key.strip():
            raise ValueError("forget key must be non-empty")
        self._run(["forget", key])
