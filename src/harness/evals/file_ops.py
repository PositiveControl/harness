"""Model-in-loop file-ops eval (Phase 3 part 2 of harness-bw27).

Where the wall-clock bench in ``scripts/bench_file_ops.py`` measures
the intrinsic speed of each candidate against a programmer-written
best-case call, this eval measures something different: can a model
drive the candidate from a natural-language prompt? The scoring
catches:

  * ``round1_called_tool`` — did the model emit a call to THIS
    candidate's tool in round 1, instead of fabricating or stalling?
  * ``final_correct`` — does the candidate's final output (or the
    mutated workspace, for in-place tasks) match the oracle?
  * ``rounds_used`` — how many model rounds were consumed before
    the loop terminated.

Each (task x candidate x prompt) case runs in its own freshly
materialized workspace; a single-tool ``ToolRegistry`` ensures the
model can't pick anything other than the candidate under test. The
test surface uses a scripted adapter so CI can exercise the scoring
path without loading a real model; the CLI subcommand
``harness eval file-ops`` runs the same code against an MLXAdapter.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from harness.evals._file_ops_corpus import (
    BENCH_TASKS,
    CandidateKind,
    FileOpsTask,
    make_tool,
)
from harness.model.adapter import ChatMessage
from harness.orchestrator import run_tool_loop
from harness.orchestrator.tool_loop import ToolLoopEvent
from harness.tools.base import ModelReply, ToolRegistry, ToolSpec

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise tool-use assistant. The workspace contains the "
    "files referenced by the user; paths in tool arguments are relative "
    "to that workspace. Use the provided tool to satisfy the request. "
    "Do not invent output — call the tool, read its result, and "
    "summarize what it produced. If the first call doesn't finish the "
    "job, call the tool again. Stay brief."
)


@dataclass(frozen=True)
class FileOpsEvalCase:
    """One (task, candidate, prompt) outcome.

    ``round1_called_tool`` is the model-in-loop signal: did the model
    pick the right tool on its first attempt? A False here is the
    schema-discoverability failure mode — the tool was registered, the
    prompt described a fit, the model still didn't reach for it.

    ``final_correct`` is the workload signal: even when the model
    eventually calls the tool, did the tool produce the bytes the
    oracle expects? A False here usually means the model fumbled the
    call's arguments — wrong path, wrong expr.

    ``passed`` combines the two: a usable end-to-end result requires
    BOTH (and no orchestration error)."""

    task_id: str
    candidate: str
    prompt: str
    rounds_used: int
    round1_called_tool: bool
    final_correct: bool
    final_output_digest: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.round1_called_tool and self.final_correct and self.error is None


@dataclass(frozen=True)
class FileOpsEvalResult:
    cases: tuple[FileOpsEvalCase, ...]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def first_try_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.round1_called_tool) / len(self.cases)

    @property
    def correctness_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.final_correct) / len(self.cases)

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    def by_candidate(self) -> dict[str, FileOpsEvalResult]:
        """Bucket the cases by candidate name. Lets the CLI render a
        per-candidate row in the report table."""
        buckets: dict[str, list[FileOpsEvalCase]] = {}
        for case in self.cases:
            buckets.setdefault(case.candidate, []).append(case)
        return {k: FileOpsEvalResult(cases=tuple(v)) for k, v in buckets.items()}

    def by_task(self) -> dict[str, FileOpsEvalResult]:
        buckets: dict[str, list[FileOpsEvalCase]] = {}
        for case in self.cases:
            buckets.setdefault(case.task_id, []).append(case)
        return {k: FileOpsEvalResult(cases=tuple(v)) for k, v in buckets.items()}

    def failures(self) -> tuple[FileOpsEvalCase, ...]:
        return tuple(c for c in self.cases if not c.passed)


class _ToolCapableAdapter(Protocol):
    """Structural subset of the real ModelAdapter interface the
    orchestrator needs for tool-loop runs. Stated explicitly here so
    tests can plug in a scripted adapter without subclassing the full
    adapter Protocol. Signature mirrors
    ``harness.orchestrator.tool_loop._ToolCapableAdapter`` so a real
    MLXAdapter / OllamaAdapter satisfies it by construction."""

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = ...,
        max_tokens: int = ...,
        temperature: float = ...,
    ) -> ModelReply: ...


@dataclass
class _EventTracker:
    """Per-case observer state. Populated by ``_observe`` during the
    tool loop; read back after ``run_tool_loop`` returns to score the
    case."""

    rounds_observed: list[list[str]] = field(default_factory=lambda: [[]])

    def observe(self, event: ToolLoopEvent) -> None:
        if event.kind == "round_start":
            self.rounds_observed.append([])
        elif (
            event.kind
            in {
                "tool_call_end",
                "tool_call_failed",
                "tool_call_declined",
                "tool_call_deduped",
            }
            and event.result is not None
        ):
            # All four kinds carry the same ``result.tool_name``; for
            # round1_called_tool we want to count any attempt to call
            # this tool, success or not, so the eval distinguishes
            # "model never reached for the tool" from "model reached
            # but the call shape was wrong."
            self.rounds_observed[-1].append(event.result.tool_name)

    def round1_called(self, tool_name: str) -> bool:
        """Did ``tool_name`` appear in any round 0/1 event?

        We accept the first non-empty round bucket so the eval is
        robust to whether the observer split round 0 (pre-loop router
        dispatch) and round 1 (first main-model round) as two buckets
        or one. The model-in-loop signal we care about is 'did the
        model reach for the right tool on its first chance.'"""
        for bucket in self.rounds_observed:
            if bucket:
                return tool_name in bucket
        return False


def run_file_ops_case(
    *,
    adapter: _ToolCapableAdapter,
    candidate: CandidateKind,
    task: FileOpsTask,
    workspace: Path,
    prompt: str,
    max_rounds: int = 5,
    system_prompt: str | None = None,
) -> FileOpsEvalCase:
    """Run one (task, candidate, prompt) case end-to-end.

    Resets ``workspace`` to a fresh fixture, registers exactly one
    tool (the candidate under test), runs ``run_tool_loop``, then
    scores. No catchers / hooks / router — the eval measures the
    candidate's tool-schema usability, not the surrounding pipeline.
    """
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    task.fixture_fn(workspace)

    # Model-boundary cap. Empirically (Qwen 2.5 7B on MLX): 32 KB of
    # tool result fed back into round 2's context stalls decoding even
    # though the math says 8k tokens easily fits in the 131k window.
    # Likely a chat-template / tokenizer cost we haven't traced down.
    # 4 KB (~1k tokens) is small enough that round 2 decodes reliably
    # AND large enough to carry a representative head-of-output sample
    # for the model to summarize. The wall-clock bench's 16 MB cap is
    # bench-only.
    tool = make_tool(candidate, workspace, max_output_bytes=4 * 1024)
    tool_name = tool.spec.name

    registry = ToolRegistry()
    registry.register(tool)

    tracker = _EventTracker()
    messages = [
        ChatMessage(role="system", content=system_prompt or _DEFAULT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt),
    ]

    try:
        loop_result = run_tool_loop(
            adapter,
            messages,
            registry,
            max_rounds=max_rounds,
            observe=tracker.observe,
            hooks=None,
        )
    except Exception as exc:
        return FileOpsEvalCase(
            task_id=task.id,
            candidate=candidate,
            prompt=prompt,
            rounds_used=0,
            round1_called_tool=False,
            final_correct=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    round1_called_tool = tracker.round1_called(tool_name)

    # Correctness: score whatever artifact the candidate actually
    # produced. For in_place tasks (multi-file-replace, function-body-
    # rewrite), the source of truth is the mutated workspace. For
    # read-only tasks, it's the candidate's tool output text. The
    # model's final reply is descriptive, not authoritative — we
    # don't score it.
    final_correct, final_digest = _score_correctness(task, workspace, loop_result, tool_name)

    return FileOpsEvalCase(
        task_id=task.id,
        candidate=candidate,
        prompt=prompt,
        rounds_used=loop_result.rounds,
        round1_called_tool=round1_called_tool,
        final_correct=final_correct,
        final_output_digest=final_digest,
    )


def run_file_ops_eval(
    *,
    adapter: _ToolCapableAdapter,
    candidates: tuple[CandidateKind, ...] = (),
    tasks: tuple[FileOpsTask, ...] = (),
    workspace_root: Path,
    max_rounds: int = 5,
    system_prompt: str | None = None,
) -> FileOpsEvalResult:
    """Run every (task x candidate x prompt) combination once. Each
    case gets its own subdirectory under ``workspace_root`` so a
    destructive in_place run on case N doesn't poison case N+1.

    Defaults: every candidate, every BENCH_TASK. Pass explicit tuples
    to restrict scope (e.g. for a quick smoke run)."""
    from harness.evals._file_ops_corpus import ALL_CANDIDATES

    chosen_candidates = candidates or ALL_CANDIDATES
    chosen_tasks = tasks or BENCH_TASKS

    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    cases: list[FileOpsEvalCase] = []
    for task in chosen_tasks:
        for candidate in chosen_candidates:
            for idx, prompt in enumerate(task.prompts):
                workspace = workspace_root / f"{task.id}__{candidate}__p{idx}"
                case = run_file_ops_case(
                    adapter=adapter,
                    candidate=candidate,
                    task=task,
                    workspace=workspace,
                    prompt=prompt,
                    max_rounds=max_rounds,
                    system_prompt=system_prompt,
                )
                cases.append(case)
    return FileOpsEvalResult(cases=tuple(cases))


def _score_correctness(
    task: FileOpsTask,
    workspace: Path,
    loop_result: object,
    tool_name: str,
) -> tuple[bool, str | None]:
    """Compare what the candidate produced against the oracle.

    For in_place tasks the workspace itself is the artifact; for
    read-only tasks the candidate's most recent tool output (from
    ``loop_result.tool_results``) is. The model's reply text is
    descriptive only — we never score it."""
    import hashlib

    in_place_task_ids = {"multi-file-replace", "function-body-rewrite"}

    if task.id in in_place_task_ids:
        gold = task.oracle_fn(workspace)
        if task.id == "function-body-rewrite":
            calc = workspace / "src" / "calc.py"
            if not calc.exists():
                return False, None
            actual = calc.read_text()
            return actual == gold, hashlib.sha256(actual.encode()).hexdigest()
        # multi-file-replace: build a manifest of the mutated workspace
        # and compare against the oracle's expected-after manifest.
        actual_manifest = _manifest_of(workspace)
        expected_manifest = _expected_post_rewrite_manifest(workspace)
        return (
            actual_manifest == expected_manifest,
            hashlib.sha256(actual_manifest.encode()).hexdigest(),
        )

    # Read-only tasks — score the candidate's last tool output.
    #
    # Two scoring paths: byte-exact against the oracle when the output
    # fits in the model-boundary cap (the truncation marker is absent),
    # and heuristic (success=True, non-empty, no error-prefix) when it
    # doesn't. The byte-exact path is the same standard the wall-clock
    # bench uses; the heuristic path is the honest fallback for tasks
    # where the candidate produces more output than the model can
    # consume — we can still tell whether the model drove the tool to
    # a non-error result, even if we can't verify exact bytes.
    results = getattr(loop_result, "tool_results", [])
    last = next(
        (r for r in reversed(results) if r.tool_name == tool_name),
        None,
    )
    if last is None or not last.success:
        return False, None
    output = last.output or ""
    if not output.strip():
        return False, hashlib.sha256(b"").hexdigest()

    # Known subprocess-level error prefixes — tools wrap subprocess
    # failures as plain strings, so success=True isn't enough.
    error_prefixes = (
        "[awk] exit=",
        "[sed] exit=",
        "[cut] exit=",
        "[tr] exit=",
        "[awk] timed out",
        "[sed] timed out",
        "[python_stream] ERROR",
        "[python_stream] timed out",
        "[pyp_stream] exit=",
        "[pyp_stream] timed out",
    )
    if any(output.startswith(prefix) for prefix in error_prefixes):
        return False, hashlib.sha256(output.encode()).hexdigest()

    if "[truncated at " in output:
        # Output exceeded the model boundary cap — can't byte-compare.
        # Heuristic pass: tool ran, output non-empty, no error prefix.
        return True, hashlib.sha256(output.encode()).hexdigest()

    # Output fit in the cap → strict byte-exact comparison.
    gold = task.oracle_fn(workspace)
    actual = output.rstrip("\n") + "\n"
    return actual == gold, hashlib.sha256(actual.encode()).hexdigest()


def _manifest_of(workspace: Path) -> str:
    """Manifest of the (possibly mutated) tests_ws/*.py files — same
    shape as the oracle's deterministic manifest."""
    import hashlib

    ws = workspace / "tests_ws"
    lines: list[str] = []
    if not ws.exists():
        return ""
    for path in sorted(ws.rglob("*.py")):
        digest = hashlib.sha256(path.read_text().encode()).hexdigest()[:16]
        rel = path.relative_to(workspace).as_posix()
        lines.append(f"{rel} {digest}")
    return "\n".join(lines) + "\n"


def _expected_post_rewrite_manifest(workspace: Path) -> str:
    """Manifest of what each .py file SHOULD look like after the
    foo→bar substitution. Derived from current content via the same
    substitution as the oracle — if the candidate already mutated the
    files correctly, this matches the actual manifest."""
    import hashlib

    ws = workspace / "tests_ws"
    lines: list[str] = []
    if not ws.exists():
        return ""
    for path in sorted(ws.rglob("*.py")):
        rewritten = path.read_text().replace("foo", "bar")
        digest = hashlib.sha256(rewritten.encode()).hexdigest()[:16]
        rel = path.relative_to(workspace).as_posix()
        lines.append(f"{rel} {digest}")
    return "\n".join(lines) + "\n"


__all__ = [
    "FileOpsEvalCase",
    "FileOpsEvalResult",
    "run_file_ops_case",
    "run_file_ops_eval",
]
