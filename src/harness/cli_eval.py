"""`harness eval` subcommands.

Step 5c of docs/cli-extraction-plan.md. Ten eval commands — voice,
router, session-resume, tfr, atc, atc-retrieval, phraseology,
atc-audio, tool-loop, file-ops.

Each handler is option declarations, wiring and Rich rendering over
`harness.evals.*`; the scoring logic lives there and is tested there.
Most import their `run_*` entry point locally, inside the branch that
needs it, so a command that needs MLX doesn't drag the import cost onto
the ones that don't.

Covered by tests/test_cli_eval_commands.py at a smoke bar — --help for
every command, --fixture plumbing, and full offline runs of the two
evals that need neither a model nor a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from harness.character import VoiceSample, load_character
from harness.cli_adapter import _resolve_adapter
from harness.cli_apps import eval_app
from harness.cli_chat_shared import _render_fact_block, _render_memory_block
from harness.cli_store import _maybe_retriever, _open_episodic_store, _open_semantic_store
from harness.cli_tools import (
    _ATC_LINT_SOURCE_FILTER,
    _build_document_tree_store_for_session,
    _resolve_router_tool_specs,
)
from harness.config import settings
from harness.evals.router import (
    RouterEvalResult,
    default_fixture_path,
    load_fixture,
    run_router_eval,
)
from harness.evals.voice import run_voice_eval
from harness.model.adapter import ChatMessage
from harness.router import GrammarRouter, ModelRouter
from harness.store.episodic import EpisodicRecord
from harness.store.semantic import SemanticFact
from harness.tools import resolve_tool_names

console = Console()


@eval_app.command("voice")
def eval_voice(
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama | vllm"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="Override the model id. MLX: HF repo. Ollama: model tag.",
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter directory (from `mlx_lm.lora` training). Requires --model mlx.",
    ),
    draft_repo: str | None = typer.Option(
        None,
        "--draft-repo",
        help="HF repo of a smaller MLX draft model for speculative decoding. "
        "Distribution-preserving throughput boost on 7B/32B targets. "
        "Defaults to HARNESS_MLX_DRAFT_MODEL_REPO.",
    ),
    sample: list[str] | None = typer.Option(
        None, "--sample", help="Limit to a specific sample id (repeatable)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    temperature: float = typer.Option(0.5, help="Sampling temperature"),
    leave_one_out: bool = typer.Option(
        True,
        "--leave-one-out/--no-leave-one-out",
        help="Exclude each sample from its own few-shot examples (default on). "
        "Disable to measure the ceiling with the full example set in view.",
    ),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Run the voice-rewrite post-pass after the substance pass.",
    ),
    top_k: int = typer.Option(
        6,
        help="Retrieve top-K voice samples by similarity to each prompt "
        "(default 6). Set 0 to show every sample (Phase 1a.2 baseline).",
    ),
    chain_rewrites: bool = typer.Option(
        False,
        "--chain-rewrites/--no-chain-rewrites",
        help="Add a second 'concrete substitution' rewrite pass on top of the "
        "style pass. Requires --persona.",
    ),
    use_judge: bool = typer.Option(
        False,
        "--judge/--no-judge",
        help="After heuristic scoring, ask the adapter to rate each response "
        "1-10 against gold. Circular (same model) but catches register drift "
        "the regex scorer misses.",
    ),
) -> None:
    """Run the canonical voice prompts and show model-vs-gold side by side."""
    character = load_character(settings.character_path)
    # eval runs persona inline in run_voice_eval so both passes stay
    # leave-one-out-consistent — do not wrap adapter here.
    adapter = _resolve_adapter(
        model, model_repo=model_repo, lora_path=lora_path, draft_repo=draft_repo
    )
    retriever = _maybe_retriever(character, top_k)

    results = run_voice_eval(
        character,
        adapter,
        temperature=temperature,
        sample_ids=sample if sample else None,
        leave_one_out=leave_one_out,
        persona=persona,
        retriever=retriever,
        top_k=top_k,
        use_judge=use_judge,
        chain_rewrites=chain_rewrites,
    )

    if as_json:
        payload = [
            {
                "sample_id": r.sample_id,
                "prompt": r.prompt,
                "gold": r.gold,
                "actual": r.actual,
                "score": {
                    "aggregate": r.score.aggregate,
                    "length_match": r.score.length_match,
                    "no_banned_openers": r.score.no_banned_openers,
                    "bullet_discipline": r.score.bullet_discipline,
                    "bullet_density": r.score.bullet_density,
                    "filler_discipline": r.score.filler_discipline,
                    "judge_score": r.score.judge_score,
                    "notes": list(r.score.notes),
                },
                **({"draft": r.draft} if r.draft is not None else {}),
            }
            for r in results
        ]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Voice eval — {adapter.id}", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    table.add_column("actual", style="yellow")
    table.add_column("score", style="cyan")
    for r in results:
        judge_line = f"\njudge={r.score.judge_score}/10" if r.score.judge_score is not None else ""
        score_cell = (
            f"{r.score.aggregate:.2f}\n"
            f"len={r.score.length_match:.2f}\n"
            f"open={r.score.no_banned_openers:.0f}\n"
            f"bul={r.score.bullet_discipline:.1f}\n"
            f"den={r.score.bullet_density:.2f}\n"
            f"fil={r.score.filler_discipline:.2f}"
            f"{judge_line}"
        )
        table.add_row(r.sample_id, r.prompt, r.gold.strip(), r.actual.strip(), score_cell)
    aggregate = sum(r.score.aggregate for r in results) / max(len(results), 1)
    console.print(table)
    console.print(
        f"[bold]aggregate voice score:[/bold] {aggregate:.3f} across {len(results)} sample(s)"
    )
    judge_scores = [r.score.judge_score for r in results if r.score.judge_score is not None]
    if judge_scores:
        judge_mean = sum(judge_scores) / len(judge_scores)
        console.print(
            f"[bold]judge mean:[/bold] {judge_mean:.2f}/10 across {len(judge_scores)} sample(s)"
        )


@eval_app.command("router")
def eval_router(
    router_repo: str = typer.Option(
        settings.router_repo,
        "--router-repo",
        help="HF repo for the router model under test (default from "
        "Settings.router_repo / HARNESS_ROUTER_REPO).",
    ),
    router_mode: str = typer.Option(
        "free",
        "--router-mode",
        help="'free' (default) or 'grammar' (JSON-schema-constrained).",
    ),
    tool_set: str = typer.Option(
        "research",
        "--tool-set",
        help="Tool profile whose specs the router sees. Default 'research' "
        "(read/list/grep/glob + search_memory/facts + search_web).",
    ),
    tools_add: str | None = typer.Option(
        None, "--tools-add", help="Comma-separated tool names to add on top of --tool-set."
    ),
    tools_drop: str | None = typer.Option(
        None, "--tools-drop", help="Comma-separated tool names to drop from --tool-set."
    ),
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a router-eval YAML file. Defaults to `character/<name>/router_eval.yaml`.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the router-eval fixture through the configured router and
    score tool-selection accuracy. Lock in quality before swapping
    models or tweaking prompts."""
    character = load_character(settings.character_path)
    path = fixture_path or default_fixture_path(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"router eval fixture not found: {path}")
    fixture = load_fixture(path)

    try:
        wanted_names = resolve_tool_names(
            tool_set,
            add=tuple((tools_add or "").split(",")),
            drop=tuple((tools_drop or "").split(",")),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    tool_specs = _resolve_router_tool_specs(wanted_names, settings.root)

    # Apply profile-scoped description overrides so the router sees the
    # same reframed specs it would see in a live chat session (e.g. atc's
    # search_memory → rulebook framing from character/<name>/
    # tool_descriptions.yaml). Without this the eval scores the router
    # against the wrong descriptions. Layered builtin → character;
    # character wins on key collision.
    from dataclasses import replace as _replace

    from harness.tools.profiles import BUILTIN_PROFILE_DESCRIPTIONS

    _overrides: dict[str, str] = {}
    _overrides.update(BUILTIN_PROFILE_DESCRIPTIONS.get(tool_set, {}))
    _overrides.update(character.tool_descriptions.get(tool_set, {}))
    if _overrides:
        tool_specs = [
            _replace(spec, description=_overrides[spec.name]) if spec.name in _overrides else spec
            for spec in tool_specs
        ]

    if router_mode not in {"free", "grammar"}:
        raise typer.BadParameter(
            f"--router-mode must be 'free' or 'grammar' (got {router_mode!r})."
        )
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter(repo=router_repo)
    router = (
        GrammarRouter(adapter=adapter) if router_mode == "grammar" else ModelRouter(adapter=adapter)
    )
    result = run_router_eval(router, tool_specs, fixture)

    if as_json:
        payload = {
            "router_repo": router_repo,
            "fixture": str(path),
            "accuracy": result.accuracy,
            "tool_accuracy": result.tool_accuracy,
            "scope_accuracy": result.scope_accuracy,
            "scope_case_count": result.scope_case_count,
            "cases": [
                {
                    "prompt": c.prompt,
                    "expected_tool": c.expected_tool,
                    "actual_tool": c.actual_tool,
                    "expected_args": list(c.expected_args),
                    "actual_args": c.actual_args,
                    "expected_scope": c.expected_scope,
                    "actual_scope": c.actual_scope,
                    "passed": c.passed,
                    "tool_correct": c.tool_correct,
                    "args_correct": c.args_correct,
                    "scope_correct": c.scope_correct,
                }
                for c in result.cases
            ],
        }
        console.print_json(json.dumps(payload))
        return

    _print_router_eval_table(result, router_repo, character.name)


def _print_router_eval_table(
    result: RouterEvalResult, router_repo: str, character_name: str
) -> None:
    table = Table(title=f"Router eval — {router_repo} · {character_name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("prompt")
    table.add_column("expected", style="green")
    table.add_column("actual", style="yellow")
    table.add_column("scope", style="cyan")
    table.add_column("args", style="dim")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        exp = c.expected_tool if c.expected_tool is not None else "[dim]null[/dim]"
        act = c.actual_tool if c.actual_tool is not None else "[dim]null[/dim]"
        if c.expected_scope is None:
            scope_note = "[dim]—[/dim]"
        elif c.scope_correct:
            scope_note = c.actual_scope
        else:
            scope_note = f"[red]{c.actual_scope}[/red] (want {c.expected_scope})"
        if c.args_correct:
            args_note = ""
        else:
            missing = sorted(set(c.expected_args) - c.actual_args.keys())
            args_note = f"missing {missing}"
        table.add_row(mark, c.prompt, exp, act, scope_note, args_note)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    summary = (
        f"[bold]{passed}/{len(result.cases)} passed · "
        f"{result.accuracy * 100:.1f}% full · "
        f"{result.tool_accuracy * 100:.1f}% tool-only"
    )
    if result.scope_case_count > 0:
        scope_passed = sum(
            1 for c in result.cases if c.expected_scope is not None and c.scope_correct
        )
        summary += (
            f" · {result.scope_accuracy * 100:.1f}% scope "
            f"({scope_passed}/{result.scope_case_count})"
        )
    summary += "[/bold]"
    console.print(summary)


@eval_app.command("session-resume")
def eval_session_resume(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a session-resume eval YAML file. Defaults to "
        "`character/<name>/session_resume_eval.yaml`.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the session-resume fixture against build_resume_summary
    and score contains / not_contains assertions per scenario. Locks
    in quality so changes to the resume protocol don't silently drop
    a load-bearing section."""
    from harness.evals.session_resume import (
        default_fixture_path as _sr_default_fixture,
    )
    from harness.evals.session_resume import (
        load_fixture as _sr_load,
    )
    from harness.evals.session_resume import (
        run_session_resume_eval as _sr_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _sr_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"session-resume eval fixture not found: {path}")
    fixtures = _sr_load(path)
    result = _sr_run(fixtures)

    if as_json:
        payload = {
            "character": character.name,
            "pass_rate": result.pass_rate,
            "cases": [
                {
                    "id": c.id,
                    "passed": c.passed,
                    "missing_contains": list(c.missing_contains),
                    "unexpected_contains": list(c.unexpected_contains),
                }
                for c in result.cases
            ],
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Session-resume eval — {character.name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("missing contains", style="yellow")
    table.add_column("unexpected", style="red")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        missing = ", ".join(c.missing_contains) if c.missing_contains else ""
        unexpected = ", ".join(c.unexpected_contains) if c.unexpected_contains else ""
        table.add_row(mark, c.id, missing, unexpected)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(result.cases)} passed · {result.pass_rate * 100:.1f}%[/bold]"
    )


@eval_app.command("tfr")
def eval_tfr(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a tfr eval YAML file. Defaults to `character/<name>/tfr_eval.yaml`.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the tfr_eval fixture through the deterministic NOTAM parser
    and score each case on geometry, type, and citations (harness-3jz1.2).

    This is the parser-axis half of the hybrid scoring spec. Model-axis
    scoring (citation grounded in retrieved chunks + LLM verdict judge)
    arrives when the FastAPI gateway (harness-3jz1.4) wires the request-
    scoped read_parsed_notam tool and the airton_c_tfr corpus is ingested.
    """
    from harness.evals.tfr import (
        default_fixture_path as _tfr_default_fixture,
    )
    from harness.evals.tfr import (
        load_fixture as _tfr_load,
    )
    from harness.evals.tfr import (
        run_tfr_eval as _tfr_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _tfr_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"tfr eval fixture not found: {path}")
    rows = _tfr_load(path)
    result = _tfr_run(rows)

    if as_json:
        payload = {
            "character": character.name,
            "pass_rate": result.pass_rate,
            "geometry_accuracy": result.geometry_accuracy,
            "type_accuracy": result.type_accuracy,
            "citation_accuracy": result.citation_accuracy,
            "cases": [
                {
                    "id": c.case_id,
                    "type_correct": c.type_correct,
                    "geometry_correct": c.geometry_correct,
                    "citations_correct": c.citations_correct,
                    "parsed_type": c.parsed.type_guess,
                    "parsed_citations": list(c.parsed.cited_sections),
                }
                for c in result.cases
            ],
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"TFR eval — {character.name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("type")
    table.add_column("geometry")
    table.add_column("citations")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        type_cell = (
            "[green]✓[/green]" if c.type_correct else f"[red]✗[/red] got {c.parsed.type_guess}"
        )
        geom_cell = "[green]✓[/green]" if c.geometry_correct else "[red]✗[/red]"
        cite_cell = (
            "[green]✓[/green]"
            if c.citations_correct
            else f"[red]✗[/red] got {list(c.parsed.cited_sections)}"
        )
        table.add_row(mark, c.case_id, type_cell, geom_cell, cite_cell)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    total = len(result.cases)
    console.print(
        f"[bold]{passed}/{total} passed · {result.pass_rate * 100:.1f}% · "
        f"geometry {result.geometry_accuracy * 100:.1f}% · "
        f"type {result.type_accuracy * 100:.1f}% · "
        f"citations {result.citation_accuracy * 100:.1f}%[/bold]"
    )


@eval_app.command("atc")
def eval_atc(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to an atc eval YAML. Defaults to `character/<name>/atc_eval.yaml`.",
    ),
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama | vllm"),
    model_repo: str | None = typer.Option(None, "--model-repo"),
    lora_path: str | None = typer.Option(None, "--lora-path"),
    draft_repo: str | None = typer.Option(None, "--draft-repo"),
    temperature: float = typer.Option(
        0.0,
        help=(
            "Sampling temperature. Defaults to 0 (greedy) so run-to-run "
            "pass-rate is stable — see harness-ald. Raise only when you "
            "want to sample variance explicitly."
        ),
    ),
    rewriter_temperature: float = typer.Option(
        0.0,
        help=(
            "Temperature for the PersonaAdapter's voice-rewrite pass. "
            "Defaults to 0 for the same reason — pass-2 style variance "
            "can swing whether a citation survives the rewrite."
        ),
    ),
    memories: int = typer.Option(3, help="Top-K episodic memories per turn"),
    facts: int = typer.Option(5, help="Top-K semantic facts per turn"),
    top_k: int = typer.Option(8, help="Top-K voice samples per turn"),
    audience: str | None = typer.Option(
        None,
        "--audience",
        help="Filter fixture to one audience (ppl|ifr|…). Default: all.",
    ),
    persona: bool = typer.Option(
        True,
        "--persona/--no-persona",
        help="Wrap the base adapter in PersonaAdapter (default on).",
    ),
    ablate: bool = typer.Option(
        False,
        "--ablate",
        help=(
            "Exclude voice samples listed in character/<name>/voice/"
            "ablation.yaml from retrieval for this eval run. Score delta "
            "vs. the default (no flag) is the generalization signal "
            "(harness-w49p; renamed from --holdout 2026-04-26 to avoid "
            "ATC phraseology collision)."
        ),
    ),
    ablate_ids: str | None = typer.Option(
        None,
        "--ablate-ids",
        help=(
            "Comma-separated sample IDs to exclude at retrieval time for "
            "this run. Overrides --ablate and the on-disk manifest — "
            "useful for round-robin per-sample memorization probes "
            "without mutating voice/ablation.yaml."
        ),
    ),
    cite_ground: bool = typer.Option(
        False,
        "--cite-ground",
        help=(
            "Run the lane-F-prime cite-grounding catcher (harness-11ha) "
            "on every case's reply. For each citation in the reply, "
            "checks whether the cited section appears in the question's "
            "top-K hybrid retrieval. Detect-only — does NOT mutate the "
            "reply or scoring; results land in the JSON envelope under "
            "`cite_grounding` and surface in the human-readable output. "
            "Use to measure how often the model emits real-but-wrong-"
            "section fabrications + suggest replacement cites."
        ),
    ),
    cite_ground_k: int = typer.Option(
        10,
        "--cite-ground-k",
        help=(
            "Top-K for the cite-grounding catcher's question retrieval. "
            "Smaller K is stricter (more false-positive ungrounded flags); "
            "larger K is looser. Default 10 balances both for atc."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Run atc's domain eval: replay PPL/IFR Q&A cases through the full
    persona + retrieval stack and score citation presence + keyword
    recall. Phase-1 target: ≥80% pass. Becomes the gate for Phase-2
    voice changes and Phase-3 LoRA (harness-xbk.7)."""
    from harness.evals.atc import (
        default_fixture_path as _atc_default_fixture,
    )
    from harness.evals.atc import (
        load_fixture as _atc_load_fixture,
    )
    from harness.evals.atc import (
        run_atc_eval as _atc_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _atc_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"atc eval fixture not found: {path}")
    fixture = _atc_load_fixture(path)
    if audience is not None:
        fixture = tuple(row for row in fixture if row.audience == audience)
    if not fixture:
        console.print("[yellow](no cases in fixture after filter — nothing to score)[/yellow]")
        raise typer.Exit(code=0)

    adapter = _resolve_adapter(
        model,
        persona=persona,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
        rewriter_temperature=rewriter_temperature,
    )

    # Retrieval stack. Off-the-shelf defaults match `harness chat`
    # with --memories 3 --facts 5 — same numbers the eval target
    # calibrates against.
    retriever = _maybe_retriever(character, top_k=top_k)
    memory_store = _open_episodic_store(character, ingest=False)
    semantic_store = _open_semantic_store()
    speaker = "eval"  # shared persona-wide content is user_id=None; a
    # literal speaker keeps the retrieval API consistent while never
    # matching a user-siloed row.

    # Ablation IDs (harness-w49p; renamed from "holdout" 2026-04-26):
    # samples listed in voice/ablation.yaml are excluded from the
    # retriever's returns when `--ablate` is on. The set is empty when
    # the flag is off or the character has no ablation file, so the
    # default retrieval path is unchanged.
    #
    # `--ablate-ids CSV` overrides both `--ablate` and the manifest for
    # ad-hoc per-sample probes (round-robin memorization map).
    if ablate_ids is not None:
        _ablate_ids: frozenset[str] = frozenset(
            part.strip() for part in ablate_ids.split(",") if part.strip()
        )
    elif ablate:
        _ablate_ids = frozenset(s.id for s in character.ablated_voice_samples)
    else:
        _ablate_ids = frozenset()

    def run_turn(question: str) -> str:
        examples: list[VoiceSample] = []
        if retriever is not None and top_k > 0:
            try:
                # Ask for a wider slate when ablation is on so the post-
                # filter doesn't shrink below top_k on characters with
                # many canonical samples (airton has 20+).
                request_k = top_k + len(_ablate_ids)
                voice_hits = retriever.top_k(question, k=request_k)
                examples = [s for s in voice_hits if s.id not in _ablate_ids][:top_k]
            except Exception:  # eval is read-only; surface score only
                examples = []

        recalled: list[EpisodicRecord] = []
        if memory_store is not None and memories > 0:
            try:
                hits = memory_store.search(question, k=memories, user_id=speaker)
                recalled = [rec for rec, _score in hits]
            except Exception:
                recalled = []

        known_facts: list[SemanticFact] = []
        if semantic_store is not None and facts > 0:
            try:
                fact_hits = semantic_store.search(question, k=facts, user_id=speaker)
                known_facts = [f for f, _score in fact_hits]
            except Exception:
                known_facts = []

        sys_prompt = character.system_prompt(include_samples=examples)
        extra: list[str] = []
        if recalled:
            extra.append(_render_memory_block(recalled))
        if known_facts:
            extra.append(_render_fact_block(known_facts))
        if extra:
            sys_prompt = sys_prompt + "\n\n" + "\n\n".join(extra)

        messages = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=question),
        ]
        return adapter.complete(messages, temperature=temperature).strip()

    try:
        result = _atc_run(fixture, run_turn)

        # Lane F-prime — cite-grounding catcher. Detect-only: per case,
        # check whether each cited §X-Y-Z appears in the question's
        # top-K hybrid retrieval. Ungrounded cites are real-but-wrong-
        # section fabs; the catcher suggests the top-1 retrieved
        # section as a candidate replacement (caller decides what to
        # do with it). Memory store must outlive this pass.
        cite_ground_per_case: dict[str, list[dict[str, object]]] = {}
        if cite_ground and memory_store is not None:
            from harness.persona.cite_grounding import check_cite_groundedness

            for c in result.cases:
                cg = check_cite_groundedness(
                    c.question,
                    c.actual_reply,
                    episodic_store=memory_store,
                    grammar=character.citation_grammar,
                    k=cite_ground_k,
                )
                cite_ground_per_case[c.id] = [
                    {
                        "cite": chk.cite,
                        "section": chk.section,
                        "grounded": chk.grounded,
                        "rank": chk.rank,
                        "suggested": chk.suggested,
                    }
                    for chk in cg.checks
                ]
    finally:
        if memory_store is not None:
            memory_store.close()
        if semantic_store is not None:
            semantic_store.close()

    if as_json:
        cases_json: list[dict[str, object]] = []
        for c in result.cases:
            entry: dict[str, object] = {
                "id": c.id,
                "audience": c.audience,
                "passed": c.passed,
                "citations_pass": c.citations_pass,
                "keywords_pass": c.keywords_pass,
                "missing_citations": list(c.missing_citations),
                "matched_keywords": list(c.matched_keywords),
                "keyword_hits": c.keyword_hits,
                "min_keyword_hits": c.min_keyword_hits,
                "reply": c.actual_reply,
            }
            if cite_ground:
                entry["cite_grounding"] = cite_ground_per_case.get(c.id, [])
            cases_json.append(entry)
        payload: dict[str, object] = {
            "character": character.name,
            "adapter": adapter.id,
            "pass_rate": result.pass_rate,
            "pass_rate_by_audience": result.pass_rate_by_audience(),
            "cases": cases_json,
        }
        if cite_ground:
            ungrounded_count = sum(
                1
                for cid, checks in cite_ground_per_case.items()
                if any(not c["grounded"] for c in checks)
            )
            payload["cite_grounding_summary"] = {
                "ungrounded_cases": ungrounded_count,
                "total_cases": len(result.cases),
                "k": cite_ground_k,
            }
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"atc eval — {character.name} · {adapter.id}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("aud.", width=4)
    table.add_column("cite", style="cyan")
    table.add_column("kw hits", style="cyan")
    table.add_column("missing citations", style="yellow")
    if cite_ground:
        table.add_column("cite-ground", style="magenta")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        cite = "[green]✓[/green]" if c.citations_pass else "[red]✗[/red]"
        kw = f"{c.keyword_hits}/{c.min_keyword_hits}"
        missing = ", ".join(c.missing_citations) if c.missing_citations else ""
        if cite_ground:
            checks = cite_ground_per_case.get(c.id, [])
            ungrounded_bits = [
                f"§{chk['section']}→§{chk['suggested']}"
                if chk["suggested"]
                else f"§{chk['section']}?"
                for chk in checks
                if not chk["grounded"]
            ]
            cg_cell = ", ".join(ungrounded_bits) if ungrounded_bits else "[green]✓[/green]"
            table.add_row(mark, c.id, c.audience, cite, kw, missing, cg_cell)
        else:
            table.add_row(mark, c.id, c.audience, cite, kw, missing)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(result.cases)} passed · {result.pass_rate * 100:.1f}%[/bold]"
    )
    rates = result.pass_rate_by_audience()
    if len(rates) > 1:
        detail = " · ".join(f"{aud}: {r * 100:.1f}%" for aud, r in sorted(rates.items()))
        console.print(f"[dim]by audience — {detail}[/dim]")
    if cite_ground:
        ungrounded_cases = sum(
            1
            for cid, checks in cite_ground_per_case.items()
            if any(not c["grounded"] for c in checks)
        )
        console.print(
            f"[dim]cite-grounding (k={cite_ground_k}) — "
            f"{ungrounded_cases}/{len(result.cases)} cases have ≥1 ungrounded cite[/dim]"
        )


@eval_app.command("atc-retrieval")
def eval_atc_retrieval(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to an atc eval YAML. Defaults to `character/<name>/atc_eval.yaml`.",
    ),
    k: int = typer.Option(10, help="Top-K depth ceiling for retrieval"),
    audience: str | None = typer.Option(
        None,
        "--audience",
        help="Filter fixture to one audience. Default: all.",
    ),
    expand_queries: bool = typer.Option(
        True,
        "--expand-queries/--no-expand-queries",
        help=(
            "Apply corpus/synonyms.yaml query-side expansion (harness-ajn) "
            "before running each fixture case through the store. Default on "
            "to match real-chat behaviour (SearchMemoryTool uses the same "
            "expander). Disable for A/B baselines measuring expander lift."
        ),
    ),
    llm_expand: bool = typer.Option(
        False,
        "--llm-expand/--no-llm-expand",
        help=(
            "Add the LLMQueryExpander pre-pass (harness-hvu1) — small "
            "model rewrites the user query into 3-5 doc-style keyword "
            "phrases that get appended before retrieval. Chains in front "
            "of the static synonym expander. Default off; flip on to "
            "measure recall lift vs the static-only baseline. Adds one "
            "small-model call per case (~100-200ms p50)."
        ),
    ),
    llm_expand_repo: str | None = typer.Option(
        None,
        "--llm-expand-repo",
        help=(
            "HF repo for the LLM-expander adapter when --llm-expand is "
            "set. Defaults to HARNESS_ROUTER_MODEL_REPO so the same "
            "small-model footprint serves routing + query expansion. "
            "Shared adapter, separate calls."
        ),
    ),
    save_baseline: bool = typer.Option(
        False,
        "--save-baseline",
        help=(
            "Write the run to character/<name>/atc_retrieval_baseline.json. "
            "Intended for snapshotting post-change so future runs can diff "
            "against the frozen rank-of-first-expected per case."
        ),
    ),
    compare_baseline: bool = typer.Option(
        False,
        "--compare-baseline",
        help=(
            "Diff this run against character/<name>/atc_retrieval_baseline.json "
            "(or --baseline-path). Exits non-zero on regression: aggregate "
            "recall@N drop OR per-case rank worsening past --regression-budget. "
            "The gate that turns the baseline JSON from a snapshot into a "
            "contract (harness-sb6r)."
        ),
    ),
    baseline_path: Path | None = typer.Option(
        None,
        "--baseline-path",
        help=(
            "Override the baseline file location for --save-baseline / "
            "--compare-baseline. Defaults to "
            "character/<name>/atc_retrieval_baseline.json."
        ),
    ),
    regression_budget: int = typer.Option(
        0,
        "--regression-budget",
        min=0,
        help=(
            "Allow up to N per-case rank regressions WHEN aggregate recall "
            "holds. Use sparingly — chunker changes that rebalance top-K "
            "without losing recall are the only legit case."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Retrieval-only atc eval: runs each fixture case through the
    episodic store and reports rank-of-first-expected + aggregate
    recall@1/@3/@5/@K. Skips the model entirely — decouples retrieval
    quality measurement from reply quality (harness-dfa)."""
    import json as _json_mod

    from harness.evals.atc import (
        default_fixture_path as _atc_default_fixture,
    )
    from harness.evals.atc import (
        load_fixture as _atc_load_fixture,
    )
    from harness.evals.atc_retrieval import (
        RetrievalHit,
        compare_baselines,
        default_baseline_path,
        load_baseline,
        make_tree_search_fn,
        run_atc_retrieval,
    )

    if save_baseline and compare_baseline:
        raise typer.BadParameter(
            "--save-baseline and --compare-baseline are mutually exclusive: "
            "compare first to confirm no regression, then re-run with "
            "--save-baseline to snapshot the new known-good state."
        )

    character = load_character(settings.character_path)
    path = fixture_path or _atc_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"atc eval fixture not found: {path}")
    fixture = _atc_load_fixture(path)
    if audience is not None:
        fixture = tuple(row for row in fixture if row.audience == audience)
    if not fixture:
        console.print("[yellow](no cases in fixture after filter — nothing to score)[/yellow]")
        raise typer.Exit(code=0)

    store = _open_episodic_store(character, ingest=False)
    if store is None:
        raise typer.BadParameter(
            "retrieval eval needs the `retrieval` extra — re-run `uv sync --extra all`."
        )

    # Build the query expander the same way SearchMemoryTool does, so
    # eval recall@k numbers reflect the retrieval path a real chat turn
    # would take. --no-expand-queries gives the A/B baseline.
    from harness.retrieval.query_expander import (
        LLMQueryExpander,
        NullQueryExpander,
        QueryExpander,
        _load_llm_expand_prompt,
        default_llm_expand_prompt_path,
        default_query_only_synonyms_path,
        default_synonyms_path,
        load_query_expander,
    )

    base_expander: QueryExpander = (
        load_query_expander(
            default_synonyms_path(settings.character_path),
            query_only_path=default_query_only_synonyms_path(settings.character_path),
        )
        if expand_queries
        else NullQueryExpander()
    )

    expander: QueryExpander
    if llm_expand:
        from harness.model.mlx import MLXAdapter

        repo = llm_expand_repo or settings.router_repo
        llm_adapter = MLXAdapter(repo=repo)
        prompt_template = _load_llm_expand_prompt(
            default_llm_expand_prompt_path(settings.character_path)
        )
        expander = LLMQueryExpander(
            llm_adapter,
            chain_to=base_expander,
            prompt_template=prompt_template,
        )
        console.print(f"[dim]llm-expand: {repo}[/dim]")
    else:
        expander = base_expander

    # harness-c9fc: when the character ships document_trees (airton_c1
    # post-migration), score against the DocumentTreeStore — the
    # episodic store no longer carries the corpus chunks. Other
    # characters (airton_c) keep the flat episodic path until they
    # migrate.
    from harness.evals.atc_retrieval import SearchFn

    _search: SearchFn
    tree_store_obj = _build_document_tree_store_for_session(character, store)
    from harness.store.document_tree import DocumentTreeStore

    if isinstance(tree_store_obj, DocumentTreeStore):
        # harness-rvnb: the eval intentionally does NOT mirror the
        # contract orchestrator's auto_merge behavior. Auto_merge is
        # the right call when the agent reads structural-only parents
        # alongside the model's wrap-up, but the eval's anchor-based
        # scoring rewards leaf-level retrieval. Replacing leaves with
        # parents during the eval would break fixture-expected
        # leaf-section anchors that happen to cluster post-retrieval.
        # Prefix matching in `_matches` covers the parent-expected
        # fixture cases without touching the orchestrator's behavior.
        _search = make_tree_search_fn(tree_store_obj, expand=expander.expand)
        if not as_json:
            console.print("[dim]retrieval source: document_tree (per character spec)[/dim]")
    else:

        def _search_episodic(query: str, depth: int) -> list[RetrievalHit]:
            raw = store.search(expander.expand(query), k=depth, mode="hybrid")
            return [
                RetrievalHit(principle=rec.principle or "", score=float(score))
                for rec, score in raw
            ]

        _search = _search_episodic

    result = run_atc_retrieval(fixture, _search, k=k)

    resolved_baseline_path = baseline_path or default_baseline_path(settings.character_path)

    comparison = None
    if compare_baseline:
        if not resolved_baseline_path.exists():
            raise typer.BadParameter(
                f"no baseline at {resolved_baseline_path}; run with "
                f"--save-baseline first to snapshot a known-good state."
            )
        comparison = compare_baselines(load_baseline(resolved_baseline_path), result)

    if as_json:
        envelope: dict[str, object] = {
            "character": character.name,
            "fixture": str(path),
            "k": result.k,
            "recall_at_1": result.recall_at_1,
            "recall_at_3": result.recall_at_3,
            "recall_at_5": result.recall_at_5,
            "recall_at_k": result.recall_at_k,
            "median_rank": result.median_rank,
            "cases": [
                {
                    "id": c.id,
                    "audience": c.audience,
                    "query": c.query,
                    "expected_anchors": list(c.expected_anchors),
                    "rank_of_first_expected": c.rank_of_first_expected,
                    "score_of_first_expected": c.score_of_first_expected,
                    "found": c.found,
                    "top_hits": [{"principle": h.principle, "score": h.score} for h in c.hits],
                }
                for c in result.cases
            ],
        }
        if comparison is not None:
            envelope["comparison"] = {
                "baseline_path": str(resolved_baseline_path),
                "regression_budget": regression_budget,
                "has_regression": comparison.has_regression(regression_budget=regression_budget),
                "aggregate_deltas": [
                    {"metric": d.metric, "old": d.old, "new": d.new}
                    for d in comparison.aggregate_deltas
                ],
                "case_regressions": [
                    {"id": d.id, "old_rank": d.old_rank, "new_rank": d.new_rank}
                    for d in comparison.case_regressions
                ],
                "case_improvements": [
                    {"id": d.id, "old_rank": d.old_rank, "new_rank": d.new_rank}
                    for d in comparison.case_improvements
                ],
                "new_cases": list(comparison.new_cases),
                "dropped_cases": list(comparison.dropped_cases),
            }
        if save_baseline:
            resolved_baseline_path.write_text(_json_mod.dumps(envelope, indent=2))
        console.print_json(data=envelope)
        if comparison is not None and comparison.has_regression(
            regression_budget=regression_budget
        ):
            raise typer.Exit(code=1)
        return

    from rich.table import Table

    table = Table(title=f"atc retrieval eval (k={result.k})", show_lines=False)
    table.add_column("pass", justify="center")
    table.add_column("id")
    table.add_column("expected")
    table.add_column("rank", justify="right")
    table.add_column("score", justify="right")
    for c in result.cases:
        mark = (
            "[green]✓[/green]"
            if c.recall_at(3)
            else ("[yellow]~[/yellow]" if c.found else "[red]✗[/red]")
        )
        rank = str(c.rank_of_first_expected) if c.rank_of_first_expected is not None else "—"
        score = f"{c.score_of_first_expected:.4f}" if c.score_of_first_expected is not None else "—"
        expected = ", ".join(c.expected_anchors)
        table.add_row(mark, c.id, expected, rank, score)
    console.print(table)
    console.print(
        f"[bold]recall@1: {result.recall_at_1 * 100:.1f}%  · "
        f"recall@3: {result.recall_at_3 * 100:.1f}%  · "
        f"recall@5: {result.recall_at_5 * 100:.1f}%  · "
        f"recall@{result.k}: {result.recall_at_k * 100:.1f}%[/bold]"
    )
    median = result.median_rank
    if median is not None:
        console.print(f"[dim]median rank of first expected (among found): {median:g}[/dim]")
    misses = result.hard_misses()
    if misses:
        console.print(
            f"[dim]hard misses (expected not in top-{result.k}): "
            f"{', '.join(m.id for m in misses)}[/dim]"
        )
    if save_baseline:
        envelope = {
            "character": character.name,
            "fixture": str(path),
            "k": result.k,
            "recall_at_1": result.recall_at_1,
            "recall_at_3": result.recall_at_3,
            "recall_at_5": result.recall_at_5,
            "recall_at_k": result.recall_at_k,
            "median_rank": result.median_rank,
            "cases": [
                {
                    "id": c.id,
                    "audience": c.audience,
                    "query": c.query,
                    "expected_anchors": list(c.expected_anchors),
                    "rank_of_first_expected": c.rank_of_first_expected,
                    "score_of_first_expected": c.score_of_first_expected,
                    "found": c.found,
                }
                for c in result.cases
            ],
        }
        resolved_baseline_path.write_text(_json_mod.dumps(envelope, indent=2))
        console.print(f"[dim]baseline written → {resolved_baseline_path}[/dim]")

    if comparison is not None:
        diff_table = Table(
            title=f"baseline diff (vs {resolved_baseline_path.name})",
            show_lines=False,
        )
        diff_table.add_column("metric")
        diff_table.add_column("old", justify="right")
        diff_table.add_column("new", justify="right")
        diff_table.add_column("Δ", justify="right")
        for agg in comparison.aggregate_deltas:
            delta = agg.new - agg.old
            color = "red" if delta < 0 else ("green" if delta > 0 else "dim")
            diff_table.add_row(
                agg.metric,
                f"{agg.old * 100:.1f}%",
                f"{agg.new * 100:.1f}%",
                f"[{color}]{delta * 100:+.1f} pp[/{color}]",
            )
        console.print(diff_table)

        if comparison.case_regressions:
            console.print(f"[red]case regressions ({len(comparison.case_regressions)}):[/red]")
            for case in comparison.case_regressions:
                old_s = "—" if case.old_rank is None else str(case.old_rank)
                new_s = "—" if case.new_rank is None else str(case.new_rank)
                console.print(f"  [red]✗[/red] {case.id}: rank {old_s} → {new_s}")
        if comparison.case_improvements:
            console.print(
                f"[green]case improvements ({len(comparison.case_improvements)}):[/green]"
            )
            for case in comparison.case_improvements:
                old_s = "—" if case.old_rank is None else str(case.old_rank)
                new_s = "—" if case.new_rank is None else str(case.new_rank)
                console.print(f"  [green]✓[/green] {case.id}: rank {old_s} → {new_s}")
        if comparison.new_cases:
            console.print(
                f"[dim]new cases (no baseline entry): {', '.join(comparison.new_cases)}[/dim]"
            )
        if comparison.dropped_cases:
            console.print(
                f"[dim]dropped cases (in baseline, not in run): "
                f"{', '.join(comparison.dropped_cases)}[/dim]"
            )

        if comparison.has_regression(regression_budget=regression_budget):
            console.print(
                f"[bold red]✗ regression detected[/bold red] (budget={regression_budget})"
            )
            raise typer.Exit(code=1)
        console.print("[bold green]✓ no regressions[/bold green]")


@eval_app.command("phraseology")
def eval_phraseology(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help=(
            "Path to a phraseology eval YAML. Defaults to `character/<name>/phraseology_eval.yaml`."
        ),
    ),
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama | vllm"),
    model_repo: str | None = typer.Option(None, "--model-repo"),
    lora_path: str | None = typer.Option(None, "--lora-path"),
    draft_repo: str | None = typer.Option(None, "--draft-repo"),
    temperature: float = typer.Option(
        0.0,
        help=(
            "Sampling temperature for the lint pipeline. Default 0 "
            "(greedy) so verdict pass-rate is stable run-to-run."
        ),
    ),
    k: int = typer.Option(
        8,
        help=(
            "Top-K hybrid retrieval per case. The candidate-anchor set "
            "the lint pipeline gates against is built from this slate."
        ),
    ),
    scenario: str | None = typer.Option(
        None,
        "--scenario",
        help=(
            "Filter fixture to one scenario class "
            "(departure | arrival | handoff | emergency). "
            "Default: all classes."
        ),
    ),
    save_baseline: bool = typer.Option(
        False,
        "--save-baseline",
        help=(
            "Write the run to character/<name>/phraseology_baseline.json. "
            "Snapshot the current verdict + citation accuracy as the "
            "frozen contract future runs diff against."
        ),
    ),
    compare_baseline: bool = typer.Option(
        False,
        "--compare-baseline",
        help=(
            "Diff this run against character/<name>/phraseology_baseline.json "
            "(or --baseline-path). Exits non-zero on regression: aggregate "
            "accuracy drop OR per-case verdict/citation pass flip past "
            "--regression-budget. The gate that turns the baseline JSON "
            "into a contract for the pre-push hook."
        ),
    ),
    baseline_path: Path | None = typer.Option(
        None,
        "--baseline-path",
        help=(
            "Override the baseline file location for --save-baseline / "
            "--compare-baseline. Defaults to "
            "character/<name>/phraseology_baseline.json."
        ),
    ),
    regression_budget: int = typer.Option(
        0,
        "--regression-budget",
        min=0,
        help=(
            "Per-case regression allowance for --compare-baseline. "
            "Aggregate accuracy regressions ALWAYS fail the gate; "
            "individual case flips up to N are tolerated when "
            "aggregate accuracy holds."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay phraseology_eval.yaml through the lint pipeline and score
    verdict + citation accuracy.

    Phase-1 ship gate: combined_accuracy ≥ 0.80. Below the gate the
    lint tool is not ready for product launch — fixture growth or
    retrieval hardening lands first.
    """
    from harness.evals.phraseology import (
        compare_baselines as _phraseology_compare,
    )
    from harness.evals.phraseology import (
        default_baseline_path as _phraseology_baseline_path,
    )
    from harness.evals.phraseology import (
        default_fixture_path as _phraseology_fixture_path,
    )
    from harness.evals.phraseology import (
        load_baseline as _phraseology_load_baseline,
    )
    from harness.evals.phraseology import (
        load_fixture as _phraseology_load_fixture,
    )
    from harness.evals.phraseology import (
        run_phraseology_eval as _phraseology_run,
    )
    from harness.tools.phraseology_lint import (
        default_verb_anchors_path,
        lint_utterance,
        load_verb_anchors,
    )

    character = load_character(settings.character_path)
    if "phraseology" not in character.tool_descriptions:
        raise typer.BadParameter(
            f"phraseology eval requires a character that declares the "
            f"`phraseology` profile in tool_descriptions.yaml "
            f"(got {character.name!r}). Set HARNESS_CHARACTER_NAME=airton_c1."
        )
    path = fixture_path or _phraseology_fixture_path(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"phraseology eval fixture not found: {path}")
    fixture = _phraseology_load_fixture(path)
    if scenario is not None:
        fixture = tuple(row for row in fixture if row.scenario == scenario)
    if not fixture:
        console.print("[yellow](no cases in fixture after filter — nothing to score)[/yellow]")
        raise typer.Exit(code=0)

    adapter = _resolve_adapter(
        model,
        persona=False,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
    )
    memory_store = _open_episodic_store(character, ingest=False)
    if memory_store is None:
        raise typer.BadParameter(
            "phraseology eval needs the episodic store — install with "
            "`uv sync --extra all` and run `harness memory ingest`."
        )

    verb_anchors = load_verb_anchors(default_verb_anchors_path(settings.character_path))

    try:

        def _lint_one(utterance: str, scenario_hint: str | None) -> object:
            return lint_utterance(
                utterance,
                adapter=adapter,
                episodic_store=memory_store,
                grammar=character.citation_grammar,
                scenario_hint=scenario_hint,
                user_id=None,
                k=k,
                temperature=temperature,
                verb_anchors=verb_anchors or None,
                source_filter=_ATC_LINT_SOURCE_FILTER,
            )

        result = _phraseology_run(fixture, _lint_one)  # type: ignore[arg-type]
    finally:
        memory_store.close()

    # JSON envelope (pre-baseline-side-effects so --save-baseline writes
    # the same payload we render).
    cases_json: list[dict[str, object]] = []
    for c in result.cases:
        cases_json.append(
            {
                "id": c.row.id,
                "scenario": c.row.scenario,
                "utterance": c.row.utterance,
                "expected_verdict": c.row.expected_verdict,
                "expected_section": c.row.expected_section,
                "actual_verdict": c.actual.verdict,
                "actual_section": c.actual.expected_section,
                "actual_phraseology": c.actual.expected_phraseology,
                "actual_mismatch": c.actual.mismatch,
                "verdict_pass": c.verdict_pass,
                "citation_pass": c.citation_pass,
                "passed": c.passed,
            }
        )
    # Render the fixture path relative to the repo root when possible,
    # so a baseline written on one machine compares cleanly on another
    # (the pre-push gate runs on every developer's clone, where the
    # absolute path is different).
    try:
        rel_fixture = str(path.relative_to(settings.root))
    except ValueError:
        rel_fixture = str(path)

    payload: dict[str, object] = {
        "character": character.name,
        "adapter": adapter.id,
        "fixture": rel_fixture,
        "case_count": len(result.cases),
        "verdict_accuracy": result.verdict_accuracy,
        "citation_accuracy": result.citation_accuracy,
        "combined_accuracy": result.combined_accuracy,
        "by_scenario": result.by_scenario(),
        "cases": cases_json,
    }

    if save_baseline:
        out_path = baseline_path or _phraseology_baseline_path(settings.character_path)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        console.print(f"[green]saved baseline[/green] → {out_path}")

    if as_json:
        console.print_json(json.dumps(payload))
    else:
        table = Table(
            title=f"phraseology eval — {character.name} · {adapter.id}",
            show_lines=False,
        )
        table.add_column("✓", style="bold", width=2)
        table.add_column("id")
        table.add_column("scen.", width=9)
        table.add_column("verdict", width=12)
        table.add_column("§ exp.", width=8)
        table.add_column("§ got", width=8)
        for c in result.cases:
            mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
            verdict_cell: str = c.actual.verdict
            if not c.verdict_pass:
                verdict_cell = f"[red]{c.actual.verdict}[/red] (≠{c.row.expected_verdict})"
            sec_exp = c.row.expected_section or "—"
            sec_got = c.actual.expected_section or "—"
            if not c.citation_pass:
                sec_got = f"[red]{sec_got}[/red]"
            table.add_row(mark, c.row.id, c.row.scenario, verdict_cell, sec_exp, sec_got)
        console.print(table)
        passed = sum(1 for c in result.cases if c.passed)
        console.print(
            f"[bold]{passed}/{len(result.cases)} passed · "
            f"combined {result.combined_accuracy * 100:.1f}% · "
            f"verdict {result.verdict_accuracy * 100:.1f}% · "
            f"citation {result.citation_accuracy * 100:.1f}%[/bold]"
        )
        per_scenario = result.by_scenario()
        if len(per_scenario) > 1:
            details = " · ".join(
                f"{scen}: {m['combined_accuracy'] * 100:.0f}% ({int(m['count'])})"
                for scen, m in sorted(per_scenario.items())
            )
            console.print(f"[dim]by scenario — {details}[/dim]")

    if compare_baseline:
        in_path = baseline_path or _phraseology_baseline_path(settings.character_path)
        if not in_path.exists():
            console.print(
                f"[yellow]no baseline at {in_path} — run with --save-baseline first[/yellow]"
            )
            raise typer.Exit(code=2)
        comparison = _phraseology_compare(_phraseology_load_baseline(in_path), result)

        if not as_json:
            console.print(f"\n[bold]baseline diff[/bold] (vs {in_path})")
            for d in comparison.aggregate_deltas:
                arrow = "→" if abs(d.new - d.old) > 1e-9 else "="
                color = "green" if d.new >= d.old else "red"
                console.print(
                    f"  [{color}]{d.metric}[/{color}]: "
                    f"{d.old * 100:.1f}% {arrow} {d.new * 100:.1f}%"
                )
            if comparison.case_regressions:
                bits = [
                    f"{d.id} (verdict {d.old_verdict_pass}→{d.new_verdict_pass}, "
                    f"cite {d.old_citation_pass}→{d.new_citation_pass})"
                    for d in comparison.case_regressions
                ]
                console.print(f"[red]case regressions:[/red] {'; '.join(bits)}")
            if comparison.case_improvements:
                console.print(
                    f"[green]case improvements:[/green] "
                    f"{', '.join(d.id for d in comparison.case_improvements)}"
                )
            if comparison.new_cases:
                console.print(
                    f"[dim]new cases (not in baseline): {', '.join(comparison.new_cases)}[/dim]"
                )
            if comparison.dropped_cases:
                console.print(
                    f"[dim]dropped cases (in baseline, not in run): "
                    f"{', '.join(comparison.dropped_cases)}[/dim]"
                )

        if comparison.has_regression(regression_budget=regression_budget):
            if not as_json:
                console.print(
                    f"[bold red]✗ regression detected[/bold red] (budget={regression_budget})"
                )
            raise typer.Exit(code=1)
        if not as_json:
            console.print("[bold green]✓ no regressions[/bold green]")


@eval_app.command("atc-audio")
def eval_atc_audio(
    target: Path | None = typer.Option(
        None,
        "--target",
        help=(
            "atc_audio dir under the active character. Defaults to "
            "`character/<name>/atc_audio` — the same path the ingest, "
            "transcribe, and label scripts write into."
        ),
    ),
    only_clip: list[str] | None = typer.Option(
        None,
        "--only",
        help=(
            "Restrict the eval to these clip ids. Useful for the small-N "
            "pre-push gate subset; defaults to every utt/*.jsonl row."
        ),
    ),
    include_unverified: bool = typer.Option(
        False,
        "--include-unverified",
        help=(
            "Score rows even when human_verified is false. Off by "
            "default — unsigned labels aren't ground truth."
        ),
    ),
    skip_noisy: bool = typer.Option(
        False,
        "--skip-noisy",
        help=(
            "Run only the clean (human transcript) lint pass. Halves "
            "model load when iterating on the lint pipeline; the noisy "
            "pass + WER columns come back zero. The pre-push gate runs "
            "both passes — only use this for inner-loop iteration."
        ),
    ),
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama | vllm"),
    model_repo: str | None = typer.Option(None, "--model-repo"),
    lora_path: str | None = typer.Option(None, "--lora-path"),
    draft_repo: str | None = typer.Option(None, "--draft-repo"),
    temperature: float = typer.Option(
        0.0,
        help=(
            "Sampling temperature for the lint pipeline. Default 0 "
            "(greedy) so verdict pass-rate is stable run-to-run."
        ),
    ),
    k: int = typer.Option(
        8,
        help=(
            "Top-K hybrid retrieval per case. Same default as the "
            "phraseology eval so audio-mode and text-mode share the "
            "candidate-anchor budget."
        ),
    ),
    save_baseline: bool = typer.Option(
        False,
        "--save-baseline",
        help=(
            "Write the run to character/<name>/atc_audio_baseline.json. "
            "Snapshot the current verdict + citation accuracy + WER as "
            "the frozen contract future runs diff against."
        ),
    ),
    compare_baseline: bool = typer.Option(
        False,
        "--compare-baseline",
        help=(
            "Diff this run against character/<name>/atc_audio_baseline.json "
            "(or --baseline-path). Exits non-zero on regression: aggregate "
            "accuracy drop, WER rise, OR per-case verdict/citation pass "
            "flip past --regression-budget. Pre-push gate's hook target."
        ),
    ),
    baseline_path: Path | None = typer.Option(
        None,
        "--baseline-path",
        help=(
            "Override the baseline file location for --save-baseline / "
            "--compare-baseline. Defaults to "
            "character/<name>/atc_audio_baseline.json."
        ),
    ),
    regression_budget: int = typer.Option(
        0,
        "--regression-budget",
        min=0,
        help=(
            "Per-case regression allowance for --compare-baseline. "
            "Aggregate accuracy / WER regressions ALWAYS fail the gate; "
            "individual case flips up to N are tolerated when "
            "aggregate metrics hold."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Score the cite-grounded ATC lint pipeline on real LiveATC
    utterances. Two passes per row — clean human transcript and noisy
    whisper seed — measure how much accuracy the audio-mode pipeline
    loses to STT vs to the lint tool itself.
    """
    from harness.evals.atc_audio import (
        compare_baselines as _audio_compare,
    )
    from harness.evals.atc_audio import (
        default_baseline_path as _audio_baseline_path,
    )
    from harness.evals.atc_audio import (
        default_target_dir as _audio_default_target,
    )
    from harness.evals.atc_audio import (
        load_baseline as _audio_load_baseline,
    )
    from harness.evals.atc_audio import (
        load_fixture as _audio_load_fixture,
    )
    from harness.evals.atc_audio import (
        run_audio_eval as _audio_run,
    )
    from harness.tools.phraseology_lint import (
        PhraseologyVerdict,
        lint_utterance,
    )

    character = load_character(settings.character_path)
    # Gate on phraseology-profile ownership rather than hard-coded
    # family list (harness-a2sa). atc-audio eval routes its
    # transcribed utterances through the phraseology lint pipeline,
    # so any character that ships a `phraseology` profile in
    # tool_descriptions.yaml is a valid target.
    if "phraseology" not in character.tool_descriptions:
        raise typer.BadParameter(
            f"atc-audio eval requires a character that declares the "
            f"`phraseology` profile in tool_descriptions.yaml "
            f"(got {character.name!r}). Set HARNESS_CHARACTER_NAME=airton_c1."
        )
    target_dir = target or _audio_default_target(settings.character_path)
    only_clip_ids = tuple(only_clip) if only_clip else None
    fixture = _audio_load_fixture(
        target_dir,
        only_verified=not include_unverified,
        only_clip_ids=only_clip_ids,
    )
    if not fixture:
        msg = (
            f"no labelled utterances under {target_dir}/utt — run scripts/atc_audio_label.py first"
        )
        if as_json:
            console.print_json(
                json.dumps({"target": str(target_dir), "case_count": 0, "hint": msg})
            )
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(code=0)

    adapter = _resolve_adapter(
        model,
        persona=False,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
    )
    memory_store = _open_episodic_store(character, ingest=False)
    if memory_store is None:
        raise typer.BadParameter(
            "atc-audio eval needs the episodic store — install with "
            "`uv sync --extra all` and run `harness memory ingest`."
        )

    try:

        def _lint_one(utterance: str, scenario_hint: str | None) -> PhraseologyVerdict:
            return lint_utterance(
                utterance,
                adapter=adapter,
                episodic_store=memory_store,
                grammar=character.citation_grammar,
                scenario_hint=scenario_hint,
                user_id=None,
                k=k,
                temperature=temperature,
                source_filter=_ATC_LINT_SOURCE_FILTER,
            )

        result = _audio_run(fixture, _lint_one, skip_noisy=skip_noisy)
    finally:
        memory_store.close()

    cases_json: list[dict[str, object]] = []
    for c in result.cases:
        cases_json.append(
            {
                "case_id": c.row.case_id,
                "clip_id": c.row.clip_id,
                "utt_index": c.row.utt_index,
                "event_tag": c.row.event_tag,
                "speaker_role": c.row.speaker_role,
                "expected_verdict": c.row.expected_verdict,
                "expected_section": c.row.expected_section,
                "transcript_text": c.row.transcript_text,
                "transcript_seed": c.row.transcript_seed,
                "wer": c.wer,
                "clean_verdict": c.clean_actual.verdict,
                "clean_section": c.clean_actual.expected_section,
                "noisy_verdict": c.noisy_actual.verdict,
                "noisy_section": c.noisy_actual.expected_section,
                "clean_verdict_pass": c.clean_verdict_pass,
                "clean_citation_pass": c.clean_citation_pass,
                "clean_passed": c.clean_passed,
                "noisy_verdict_pass": c.noisy_verdict_pass,
                "noisy_citation_pass": c.noisy_citation_pass,
                "noisy_passed": c.noisy_passed,
                "verdict_shift": c.verdict_shift,
            }
        )

    payload: dict[str, object] = {
        "character": character.name,
        "adapter": adapter.id,
        "target": str(target_dir),
        "case_count": result.case_count,
        "mean_wer": result.mean_wer,
        "clean_verdict_accuracy": result.clean_verdict_accuracy,
        "noisy_verdict_accuracy": result.noisy_verdict_accuracy,
        "clean_citation_accuracy": result.clean_citation_accuracy,
        "noisy_citation_accuracy": result.noisy_citation_accuracy,
        "clean_combined_accuracy": result.clean_combined_accuracy,
        "noisy_combined_accuracy": result.noisy_combined_accuracy,
        "verdict_shift_rate": result.verdict_shift_rate,
        "by_event_tag": result.by_event_tag(),
        "cases": cases_json,
    }

    if save_baseline:
        out_path = baseline_path or _audio_baseline_path(settings.character_path)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        console.print(f"[green]saved baseline[/green] → {out_path}")

    if as_json:
        console.print_json(json.dumps(payload))
    else:
        table = Table(
            title=f"atc-audio eval — {character.name} · {adapter.id} · {result.case_count} utts",
            show_lines=False,
        )
        table.add_column("✓", style="bold", width=2)
        table.add_column("case", width=22)
        table.add_column("event", width=12)
        table.add_column("verdict (clean→noisy)")
        table.add_column("§ exp", width=7)
        table.add_column("§ got (clean)", width=10)
        table.add_column("WER", width=5, justify="right")
        for c in result.cases:
            mark = "[green]✓[/green]" if c.clean_passed and c.noisy_passed else "[red]✗[/red]"
            verdict_cell = (
                f"{c.clean_actual.verdict} → {c.noisy_actual.verdict}"
                if not skip_noisy
                else c.clean_actual.verdict
            )
            if not c.clean_verdict_pass:
                verdict_cell = f"[red]{verdict_cell}[/red] (≠{c.row.expected_verdict})"
            elif c.verdict_shift:
                verdict_cell = f"[yellow]{verdict_cell}[/yellow] (shifted)"
            sec_exp = c.row.expected_section or "—"
            sec_got = c.clean_actual.expected_section or "—"
            if not c.clean_citation_pass:
                sec_got = f"[red]{sec_got}[/red]"
            table.add_row(
                mark,
                c.row.case_id,
                c.row.event_tag or "(none)",
                verdict_cell,
                sec_exp,
                sec_got,
                f"{c.wer:.2f}",
            )
        console.print(table)
        console.print(
            f"[bold]clean[/bold]: verdict {result.clean_verdict_accuracy * 100:.1f}% · "
            f"citation {result.clean_citation_accuracy * 100:.1f}% · "
            f"combined {result.clean_combined_accuracy * 100:.1f}%"
        )
        if not skip_noisy:
            console.print(
                f"[bold]noisy[/bold]: verdict {result.noisy_verdict_accuracy * 100:.1f}% · "
                f"citation {result.noisy_citation_accuracy * 100:.1f}% · "
                f"combined {result.noisy_combined_accuracy * 100:.1f}% · "
                f"WER {result.mean_wer:.2f} · "
                f"shift-rate {result.verdict_shift_rate * 100:.1f}%"
            )

    if compare_baseline:
        in_path = baseline_path or _audio_baseline_path(settings.character_path)
        if not in_path.exists():
            console.print(
                f"[yellow]no baseline at {in_path} — run with --save-baseline first[/yellow]"
            )
            raise typer.Exit(code=2)
        comparison = _audio_compare(_audio_load_baseline(in_path), result)
        if not as_json:
            console.print(f"\n[bold]baseline diff[/bold] (vs {in_path})")
            for d in comparison.aggregate_deltas:
                arrow = "→" if abs(d.new - d.old) > 1e-9 else "="
                color = "red" if d.is_regression else "green"
                console.print(f"  [{color}]{d.metric}[/{color}]: {d.old:.4f} {arrow} {d.new:.4f}")
            if comparison.case_regressions:
                bits = [d.case_id for d in comparison.case_regressions]
                console.print(f"[red]case regressions:[/red] {', '.join(bits)}")
            if comparison.case_improvements:
                console.print(
                    f"[green]case improvements:[/green] "
                    f"{', '.join(d.case_id for d in comparison.case_improvements)}"
                )
            if comparison.new_cases:
                console.print(
                    f"[dim]new cases (not in baseline): {', '.join(comparison.new_cases)}[/dim]"
                )
            if comparison.dropped_cases:
                console.print(
                    f"[dim]dropped cases (in baseline, not in run): "
                    f"{', '.join(comparison.dropped_cases)}[/dim]"
                )
        if comparison.has_regression(regression_budget=regression_budget):
            if not as_json:
                console.print(
                    f"[bold red]✗ regression detected[/bold red] (budget={regression_budget})"
                )
            raise typer.Exit(code=1)
        if not as_json:
            console.print("[bold green]✓ no regressions[/bold green]")


@eval_app.command("tool-loop")
def eval_tool_loop(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a tool-loop eval YAML file. Defaults to "
        "`character/<name>/tool_loop_eval.yaml`.",
    ),
    attribute: bool = typer.Option(
        False,
        "--attribute",
        help="Also run the per-catcher attribution harness (disable each "
        "catcher in turn and report which scenarios it uniquely saves).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the tool-loop failure corpus through scripted adapters and
    score contains / not_contains / events / messages assertions per
    scenario. With --attribute, also measure which orchestrator catcher
    uniquely saves which scenario."""
    from harness.evals.tool_loop import (
        default_fixture_path as _tl_default_fixture,
    )
    from harness.evals.tool_loop import (
        load_fixture as _tl_load,
    )
    from harness.evals.tool_loop import (
        run_attribution as _tl_attribution,
    )
    from harness.evals.tool_loop import (
        run_tool_loop_eval as _tl_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _tl_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"tool-loop eval fixture not found: {path}")
    fixtures = _tl_load(path)

    if attribute:
        attr = _tl_attribution(fixtures)
        baseline = attr.baseline
    else:
        baseline = _tl_run(fixtures)
        attr = None

    if as_json:
        payload: dict[str, object] = {
            "character": character.name,
            "pass_rate": baseline.pass_rate,
            "cases": [
                {
                    "id": c.id,
                    "label": c.label,
                    "passed": c.passed,
                    "rounds": c.rounds,
                    "missing_contains": list(c.missing_contains),
                    "unexpected_contains": list(c.unexpected_contains),
                    "missing_events": list(c.missing_events),
                    "unexpected_events": list(c.unexpected_events),
                    "missing_message_substrings": list(c.missing_message_substrings),
                    "unexpected_message_substrings": list(c.unexpected_message_substrings),
                    "expected_fallback": c.expected_fallback,
                    "fallback_triggered": c.fallback_triggered,
                }
                for c in baseline.cases
            ],
        }
        if attr is not None:
            payload["attributions"] = [
                {
                    "catcher": a.catcher,
                    "unique_saves": list(a.unique_saves),
                    "also_breaks": list(a.also_breaks),
                    "no_effect": a.no_effect,
                }
                for a in attr.attributions
            ]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Tool-loop eval — {character.name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("label", style="cyan")
    table.add_column("rounds", style="dim", justify="right")
    table.add_column("failures", style="red")
    for c in baseline.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        failures: list[str] = []
        if c.missing_contains:
            failures.append(f"missing: {list(c.missing_contains)}")
        if c.unexpected_contains:
            failures.append(f"unexpected: {list(c.unexpected_contains)}")
        if c.missing_events:
            failures.append(f"missing events: {list(c.missing_events)}")
        if c.unexpected_events:
            failures.append(f"unexpected events: {list(c.unexpected_events)}")
        if c.missing_message_substrings:
            failures.append(f"missing msg: {list(c.missing_message_substrings)}")
        if c.unexpected_message_substrings:
            failures.append(f"unexpected msg: {list(c.unexpected_message_substrings)}")
        if c.expected_fallback != c.fallback_triggered:
            failures.append(f"fallback expected={c.expected_fallback} got={c.fallback_triggered}")
        table.add_row(mark, c.id, c.label, str(c.rounds), " · ".join(failures))
    console.print(table)
    passed = sum(1 for c in baseline.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(baseline.cases)} passed · {baseline.pass_rate * 100:.1f}%[/bold]"
    )

    if attr is not None:
        attr_table = Table(title="Per-catcher attribution", show_lines=False, title_style="bold")
        attr_table.add_column("catcher", style="cyan")
        attr_table.add_column("uniquely saves", style="green")
        attr_table.add_column("shares coverage with (also_breaks)", style="dim")
        attr_table.add_column("no effect", style="red")
        for a in attr.attributions:
            attr_table.add_row(
                a.catcher,
                ", ".join(a.unique_saves) or "-",
                ", ".join(a.also_breaks) or "-",
                "yes" if a.no_effect else "",
            )
        console.print(attr_table)


@eval_app.command("file-ops")
def eval_file_ops(
    model: str = typer.Option("mlx", "--model", help="Adapter: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="HF repo (mlx) or Ollama tag for the model under test. "
        "Default: mlx-community/Qwen2.5-7B-Instruct-4bit via the adapter factory.",
    ),
    workspace_root: Path = typer.Option(
        Path("/tmp/bw27_file_ops_eval"),  # noqa: S108 — eval scratch, not security-sensitive
        "--workspace-root",
        help="Parent directory for per-case workspaces. Wiped + repopulated per case.",
    ),
    candidates: str = typer.Option(
        "",
        "--candidates",
        help="Comma-separated candidate names (stream_edit, python_stream). Default: both.",
    ),
    tasks: str = typer.Option(
        "",
        "--tasks",
        help="Comma-separated task ids to restrict to. Default: every BENCH_TASK.",
    ),
    max_rounds: int = typer.Option(
        5,
        "--max-rounds",
        help="Tool-loop round budget per case. Defaults to 5 — the file-ops "
        "tasks are small enough that more is usually noise.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Run the harness-bw27 model-in-loop file-ops eval.

    Drives every (task x candidate x prompt) combination through a
    single-tool registry containing only the candidate under test, then
    scores round1_called_tool + final_correct against the oracle. The
    scripted-adapter unit tests live in tests/test_evals_file_ops.py;
    this subcommand is the surface for real-model runs."""
    from harness.evals._file_ops_corpus import ALL_CANDIDATES, BENCH_TASKS, CandidateKind
    from harness.evals.file_ops import run_file_ops_eval

    selected_candidates: tuple[CandidateKind, ...]
    if candidates:
        raw_names = tuple(c.strip() for c in candidates.split(","))
        for name in raw_names:
            if name not in ALL_CANDIDATES:
                raise typer.BadParameter(
                    f"unknown candidate {name!r}; valid: {ALL_CANDIDATES}",
                )
        selected_candidates = cast("tuple[CandidateKind, ...]", raw_names)
    else:
        selected_candidates = ALL_CANDIDATES

    selected_tasks: tuple[Any, ...]
    if tasks:
        ids = {t.strip() for t in tasks.split(",")}
        selected_tasks = tuple(t for t in BENCH_TASKS if t.id in ids)
        unknown = ids - {t.id for t in BENCH_TASKS}
        if unknown:
            raise typer.BadParameter(f"unknown task ids: {sorted(unknown)}")
    else:
        selected_tasks = BENCH_TASKS

    adapter = _resolve_adapter(model, model_repo=model_repo)
    # `_resolve_adapter` returns a `ModelAdapter` (the minimal Protocol
    # without `complete_with_tools`). The concrete adapters (MLX,
    # Ollama, Echo) all implement complete_with_tools — the eval's
    # tool-loop runner will fail loudly at call time if not. Cast so
    # mypy stops asking for proof we already have.
    result = run_file_ops_eval(
        adapter=cast(Any, adapter),
        candidates=selected_candidates,
        tasks=selected_tasks,
        workspace_root=workspace_root,
        max_rounds=max_rounds,
    )

    if as_json:
        payload = {
            "model": model,
            "model_repo": model_repo,
            "total": result.total,
            "first_try_rate": result.first_try_rate,
            "correctness_rate": result.correctness_rate,
            "pass_rate": result.pass_rate,
            "cases": [
                {
                    "task": c.task_id,
                    "candidate": c.candidate,
                    "prompt": c.prompt,
                    "rounds_used": c.rounds_used,
                    "round1_called_tool": c.round1_called_tool,
                    "final_correct": c.final_correct,
                    "passed": c.passed,
                    "error": c.error,
                }
                for c in result.cases
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    console.print(
        f"\n[bold]harness-bw27 file-ops eval[/bold] — "
        f"{result.total} case(s), max_rounds={max_rounds}, model={model}",
    )
    console.print(
        f"  first-try call rate: {result.first_try_rate:.1%}  "
        f"correctness: {result.correctness_rate:.1%}  "
        f"pass: {result.pass_rate:.1%}",
    )
    console.print("\n[bold]per candidate[/bold]")
    for cand, sub in result.by_candidate().items():
        console.print(
            f"  {cand:15s}  first-try={sub.first_try_rate:.1%}  "
            f"correct={sub.correctness_rate:.1%}  pass={sub.pass_rate:.1%}  "
            f"(n={sub.total})",
        )

    failures = result.failures()
    if failures:
        console.print(f"\n[bold]{len(failures)} failure(s):[/bold]")
        for case in failures:
            tag = "ERR" if case.error else ("✗call" if not case.round1_called_tool else "✗score")
            console.print(
                f"  [{tag:7s}] {case.task_id:25s}  {case.candidate:14s}  "
                f"rounds={case.rounds_used}  prompt={case.prompt[:60]!r}",
            )
            if case.error:
                console.print(f"           error: {case.error.splitlines()[0]}")
