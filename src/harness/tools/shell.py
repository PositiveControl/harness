from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


@dataclass
class ShellTool:
    """Run a shell command, capture stdout + stderr + exit code.
    Write-tier — requires user confirmation. No interactive input
    supported; if the command waits for input it will time out.

    `cwd` pins the working directory. None means inherit the caller's
    cwd (fine for tests); in production chat the CLI pins it to the
    workspace so `pwd` agrees with `read_file` / `write_file`."""

    timeout_seconds: int = 15
    cwd: Path | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description=(
                "Run a shell command in the workspace. Captures stdout + "
                "stderr. Returns combined output with the exit code. "
                "One-shot commands only; no interactive input. User "
                "confirmation is required before every invocation until "
                "the user approves this tool for the session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Command line to run",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (f"Seconds before killing the process. Default {15}."),
                    },
                },
                "required": ["cmd"],
            },
            tier="write",
            display_name="Run shell",
        )

    def call(self, *, cmd: str, timeout: int | None = None) -> str:
        effective_timeout = timeout or self.timeout_seconds
        try:
            # shell=True is deliberate — this tool IS a shell. Gated by the
            # tool-loop's user-confirmation for write-tier tools.
            result = subprocess.run(  # noqa: S602
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
                cwd=str(self.cwd) if self.cwd is not None else None,
            )
        except subprocess.TimeoutExpired:
            return f"timed out after {effective_timeout}s"
        combined = (result.stdout or "") + (result.stderr or "")
        return f"exit={result.returncode}\n{combined}".rstrip()
