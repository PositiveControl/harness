"""Adapter boundary.

Mirrors the harness's model-adapter invariant: framework-specific code (CLIs,
SDKs) lives only inside adapter modules. The runner speaks `Adapter` +
`RunArtifacts` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Endpoint:
    """The shared gx10 vLLM endpoint handed to every framework."""

    base_url: str
    model: str
    api_key: str = "gx10"
    temperature: float = 0.0


@dataclass
class RunArtifacts:
    """Everything one run produces, captured for scoring and replay."""

    exit_ok: bool
    transcript: str
    diff: str
    duration_s: float
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    turns: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Adapter(Protocol):
    """One framework under test."""

    name: str
    version: str

    def prepare(self, workspace: Path) -> None:
        """Set up a fresh, isolated build directory (e.g. git init)."""
        ...

    def invoke(self, spec: str, gx10: Endpoint, workspace: Path, timeout_s: int) -> RunArtifacts:
        """Drive the framework to build the spec; return captured artifacts."""
        ...
