"""Tests for harness.evals.file_ops — Phase 3 part 2 of harness-bw27.

Exercise the scoring path with a deterministic scripted adapter so CI
doesn't need to load a real model. Three things to pin:

  1. round1_called_tool is True when the scripted reply calls the
     candidate's tool on round 1, False when it doesn't.
  2. final_correct reflects what the candidate actually produced —
     correct call args → True; wrong args → False; in_place tasks
     score against the mutated workspace, read-only tasks against the
     last tool output.
  3. Aggregate helpers (first_try_rate / correctness_rate / pass_rate
     / by_candidate / by_task) compute the expected ratios.

A real-model run is what the CLI subcommand does; that surface is
covered by integration runs, not unit tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from harness.evals._file_ops_corpus import BENCH_TASKS, FileOpsTask
from harness.evals.file_ops import (
    FileOpsEvalCase,
    FileOpsEvalResult,
    run_file_ops_case,
    run_file_ops_eval,
)
from harness.model.adapter import ChatMessage
from harness.tools.base import ModelReply, ToolCall, ToolSpec


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued ``ModelReply`` objects in order. When the queue
    runs dry, returns a final empty content reply to end the loop."""

    replies: list[ModelReply]
    _seen_prompts: list[str] = field(default_factory=list, init=False)

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        # Capture the last user-role message for assertions if tests need it.
        for msg in messages:
            if msg.role == "user":
                self._seen_prompts.append(msg.content)
                break
        if self.replies:
            return self.replies.pop(0)
        return ModelReply(content="done")


def _task(task_id: str) -> FileOpsTask:
    return next(t for t in BENCH_TASKS if t.id == task_id)


# --- round1_called_tool tracking --------------------------------------------


def test_round1_called_tool_true_when_model_calls_tool_first(tmp_path: Path) -> None:
    """Scripted reply 1: call stream_edit. Scripted reply 2: empty
    (loop terminates). Tracker should report round1_called_tool=True."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="stream_edit",
                        arguments={
                            "tool": "awk",
                            "args": ["{print $3}"],
                            "paths": ["logs/sample.log"],
                        },
                    ),
                ),
            ),
            ModelReply(content="done"),
        ],
    )
    case = run_file_ops_case(
        adapter=adapter,
        candidate="stream_edit",
        task=_task("extract-col3"),
        workspace=tmp_path / "ws",
        prompt=_task("extract-col3").prompts[0],
    )
    assert case.round1_called_tool is True
    assert case.final_correct is True
    assert case.passed is True


def test_round1_called_tool_false_when_model_text_only(tmp_path: Path) -> None:
    """If the model fabricates a text answer with no tool call on round
    1, the eval must catch it — round1_called_tool=False, final_correct
    almost certainly False too."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="hosts: a b c")])
    case = run_file_ops_case(
        adapter=adapter,
        candidate="stream_edit",
        task=_task("extract-col3"),
        workspace=tmp_path / "ws",
        prompt=_task("extract-col3").prompts[0],
    )
    assert case.round1_called_tool is False
    assert case.final_correct is False


def test_truncated_output_falls_back_to_heuristic(tmp_path: Path) -> None:
    """When the candidate's output exceeds the 4 KB model-boundary cap,
    the eval can't byte-compare against the oracle. It falls back to a
    heuristic pass: tool returned success=True, output is non-empty, and
    doesn't start with a known subprocess-error prefix. That's the
    contract the wall-clock bench complements with byte-exact scoring."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="stream_edit",
                        arguments={
                            "tool": "awk",
                            "args": ["{print $3}"],
                            "paths": ["logs/sample.log"],
                        },
                    ),
                ),
            ),
            ModelReply(content="done"),
        ],
    )
    case = run_file_ops_case(
        adapter=adapter,
        candidate="stream_edit",
        task=_task("extract-col3"),
        workspace=tmp_path / "ws",
        prompt=_task("extract-col3").prompts[0],
    )
    # 100k lines of host names truncated at 4 KB → heuristic mode →
    # passes because output is non-empty and no error prefix.
    assert case.round1_called_tool is True
    assert case.final_correct is True


def test_round1_call_with_wrong_args_marked_incorrect(tmp_path: Path) -> None:
    """Model called the right tool but used the wrong column. With
    byte-exact scoring (small enough output to fit in the 4 KB model
    boundary cap), final_correct should be False.

    distinct-count is the right task to test this against because its
    output is a single integer — easily within the cap — so the
    eval's byte-exact path activates instead of the truncated-output
    heuristic fallback."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="stream_edit",
                        arguments={
                            "tool": "awk",
                            # Right shape, wrong column — counts distinct
                            # services (5) instead of distinct users (~1000).
                            "args": ["{seen[$2]=1} END{print length(seen)}"],
                            "paths": ["logs/sample.log"],
                        },
                    ),
                ),
            ),
            ModelReply(content="done"),
        ],
    )
    case = run_file_ops_case(
        adapter=adapter,
        candidate="stream_edit",
        task=_task("distinct-count"),
        workspace=tmp_path / "ws",
        prompt=_task("distinct-count").prompts[0],
    )
    assert case.round1_called_tool is True
    assert case.final_correct is False
    assert case.passed is False


# --- in_place scoring -------------------------------------------------------


def test_in_place_correct_scores_against_workspace(tmp_path: Path) -> None:
    """multi-file-replace: scripted reply mutates the workspace
    correctly. Oracle is the post-edit manifest of the mutated tree."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="stream_edit",
                        arguments={
                            "tool": "sed",
                            "args": ["s/foo/bar/g"],
                            # The eval reset-and-materialized fixture into
                            # `workspace`. The scripted call needs the full
                            # path list — but stream_edit's in_place mode
                            # only accepts paths inside the workspace root.
                            # We rely on stream_edit's _all-paths discovery
                            # via the runner that owned this responsibility
                            # in the bench. For this test, supply a glob-
                            # like list that's stable: every .py under
                            # tests_ws.
                            "paths": _all_py_paths(tmp_path / "ws"),
                            "in_place": True,
                        },
                    ),
                ),
            ),
            ModelReply(content="done"),
        ],
    )
    # We need the fixture materialized BEFORE building the path list,
    # but run_file_ops_case resets the workspace. Mirror that flow:
    task = _task("multi-file-replace")
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    task.fixture_fn(workspace)
    paths = _all_py_paths(workspace)
    adapter.replies[0] = ModelReply(
        content="",
        tool_calls=(
            ToolCall(
                name="stream_edit",
                arguments={
                    "tool": "sed",
                    "args": ["s/foo/bar/g"],
                    "paths": paths,
                    "in_place": True,
                },
            ),
        ),
    )
    case = run_file_ops_case(
        adapter=adapter,
        candidate="stream_edit",
        task=task,
        workspace=workspace,
        prompt=task.prompts[0],
    )
    assert case.round1_called_tool is True
    assert case.final_correct is True


def _all_py_paths(workspace: Path) -> list[str]:
    ws = workspace / "tests_ws"
    return sorted(p.relative_to(workspace).as_posix() for p in ws.rglob("*.py"))


# --- error handling --------------------------------------------------------


def test_orchestration_error_recorded(tmp_path: Path) -> None:
    """If run_tool_loop raises, the case records the error and the
    pass/correctness flags stay False — the eval keeps going."""

    class _ExplodingAdapter:
        def complete_with_tools(self, *args: object, **kwargs: object) -> ModelReply:
            raise RuntimeError("simulated adapter failure")

    case = run_file_ops_case(
        adapter=_ExplodingAdapter(),
        candidate="stream_edit",
        task=_task("extract-col3"),
        workspace=tmp_path / "ws",
        prompt=_task("extract-col3").prompts[0],
    )
    assert case.error is not None
    assert "RuntimeError" in case.error
    assert case.round1_called_tool is False
    assert case.final_correct is False
    assert case.passed is False


# --- aggregate helpers -----------------------------------------------------


def _case(
    task_id: str = "t",
    candidate: str = "stream_edit",
    *,
    round1: bool,
    correct: bool,
    error: str | None = None,
) -> FileOpsEvalCase:
    return FileOpsEvalCase(
        task_id=task_id,
        candidate=candidate,
        prompt="p",
        rounds_used=1,
        round1_called_tool=round1,
        final_correct=correct,
        error=error,
    )


def test_first_try_correctness_pass_rates() -> None:
    result = FileOpsEvalResult(
        cases=(
            _case(round1=True, correct=True),
            _case(round1=True, correct=False),
            _case(round1=False, correct=True),
            _case(round1=False, correct=False),
        ),
    )
    assert result.total == 4
    assert result.first_try_rate == 0.5
    assert result.correctness_rate == 0.5
    assert result.pass_rate == 0.25


def test_by_candidate_buckets() -> None:
    result = FileOpsEvalResult(
        cases=(
            _case(candidate="stream_edit", round1=True, correct=True),
            _case(candidate="stream_edit", round1=False, correct=False),
            _case(candidate="pyp_stream", round1=True, correct=True),
        ),
    )
    buckets = result.by_candidate()
    assert set(buckets) == {"stream_edit", "pyp_stream"}
    assert buckets["stream_edit"].pass_rate == 0.5
    assert buckets["pyp_stream"].pass_rate == 1.0


def test_by_task_buckets() -> None:
    result = FileOpsEvalResult(
        cases=(
            _case(task_id="extract-col3", round1=True, correct=True),
            _case(task_id="extract-col3", round1=True, correct=False),
            _case(task_id="distinct-count", round1=True, correct=True),
        ),
    )
    buckets = result.by_task()
    assert set(buckets) == {"extract-col3", "distinct-count"}
    assert buckets["extract-col3"].correctness_rate == 0.5
    assert buckets["distinct-count"].correctness_rate == 1.0


def test_failures_excludes_passes() -> None:
    result = FileOpsEvalResult(
        cases=(
            _case(round1=True, correct=True),
            _case(round1=False, correct=True),
            _case(round1=True, correct=False, error="boom"),
        ),
    )
    failures = result.failures()
    assert len(failures) == 2
    assert all(not f.passed for f in failures)


# --- full sweep ------------------------------------------------------------


def test_run_file_ops_eval_runs_all_combinations(tmp_path: Path) -> None:
    """Each (task, candidate) combination gets each of the task's
    prompts. With one task and one candidate that's 3 cases."""
    task = _task("extract-col3")
    correct_args = {
        "tool": "awk",
        "args": ["{print $3}"],
        "paths": ["logs/sample.log"],
    }
    replies: list[ModelReply] = []
    # Each case loops twice (call + wrap-up), so 6 replies for 3 cases.
    for _ in range(3):
        replies.append(
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="stream_edit", arguments=correct_args),),
            ),
        )
        replies.append(ModelReply(content="done"))
    adapter = _ScriptedAdapter(replies=replies)
    result = run_file_ops_eval(
        adapter=adapter,
        candidates=("stream_edit",),
        tasks=(task,),
        workspace_root=tmp_path / "ws",
    )
    assert result.total == 3
    assert result.pass_rate == 1.0
    # All three prompts should appear in the case set.
    seen_prompts = {c.prompt for c in result.cases}
    assert seen_prompts == set(task.prompts)


# Tighten test_evals_file_ops to surface the bench against the
# scripted adapter — this stays under the noise floor of test_bench_file_ops.
def test_default_system_prompt_is_terse() -> None:
    """Light sanity: the default system prompt sets a tool-use frame
    and asks the model to summarize tool output rather than fabricate.
    Catching prompt drift early; this is THE handshake that decides
    whether round1_called_tool ever fires for a non-trivial fraction
    of cases."""
    from harness.evals.file_ops import _DEFAULT_SYSTEM_PROMPT

    assert "tool" in _DEFAULT_SYSTEM_PROMPT.lower()
    assert "workspace" in _DEFAULT_SYSTEM_PROMPT.lower()
    # Don't lecture the model on what NOT to do — describe the desired behavior.
    assert "do not invent" in _DEFAULT_SYSTEM_PROMPT.lower()


# Silence "unused import" warnings — pytest, Any.
_ = pytest, Any
