from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from harness.model.adapter import ChatMessage
from harness.orchestrator.hooks import (
    AB_DATE_HEADER_RE,
    AB_TIER_HEADER_RE,
    BARE_CLAIM_RE,
    DUPLICATE_CALL_NUDGE,
    EXHAUSTED_FABRICATION_FALLBACK,
    FABRICATED_AB_CAPTURE_RE,
    FABRICATED_AB_SCOPE_RE,
    FABRICATED_BEAD_ID_RE,
    FABRICATED_REMEMBER_RE,
    FABRICATED_SEARCH_RE,
    FALSE_SUCCESS_RE,
    META_CONFIRM_RE,
    TEASER_RE,
    TOOL_INTENT_RE,
    BailContext,
    BailOutcome,
    Continue,
    FinalizeContext,
    Halt,
    HookPipeline,
    PostModelContext,
    PostToolContext,
    PreToolContext,
    Replace,
    ReplaceResult,
    Skip,
    Truncated,
    default_hook_pipeline,
    looks_like_ab_fabrication,
)
from harness.persona.banter import BanterStreakTracker, is_banter_prompt
from harness.tools.base import (
    ModelReply,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolRegistry,
    ToolResult,
)

# Canonical catcher names. Exposed for the attribution eval
# (harness-cfm7) to disable one catcher at a time and measure which
# scenarios it uniquely saves. Derived from the default hook pipeline
# so adding a new hook automatically updates the surface. Production
# code never mutates `_DISABLED_CATCHERS`; only `harness.evals.tool_loop`
# does (via its `disable_catchers` context manager).
_DEFAULT_PIPELINE: HookPipeline = default_hook_pipeline()
# Full catcher surface includes the opt-in domain catchers
# (harness-qvwq) — runtime characters install only the subset they
# declare in core.yaml, but the attribution eval + fixture validator
# need to see every nameable catcher.
_CATCHER_NAMES: tuple[str, ...] = default_hook_pipeline(
    catchers=(
        "ab_fabrication",
        "ambiguous_context",
        "scope_redirect",
        "reserved_squawk_code",
    ),
).names()
_DISABLED_CATCHERS: set[str] = set()


def _catcher_enabled(name: str) -> bool:
    return name not in _DISABLED_CATCHERS


# Re-exported for historical import stability. The hooks module owns
# these now, but tests + the CLI stream filter + ab_ops eval fixtures
# import them through `harness.orchestrator.tool_loop` and
# `harness.orchestrator`.
_DUPLICATE_CALL_NUDGE = DUPLICATE_CALL_NUDGE
_EXHAUSTED_FABRICATION_FALLBACK = EXHAUSTED_FABRICATION_FALLBACK
_TEASER_RE = TEASER_RE
_FALSE_SUCCESS_RE = FALSE_SUCCESS_RE
_META_CONFIRM_RE = META_CONFIRM_RE
_FABRICATED_SEARCH_RE = FABRICATED_SEARCH_RE
_FABRICATED_AB_CAPTURE_RE = FABRICATED_AB_CAPTURE_RE
_FABRICATED_REMEMBER_RE = FABRICATED_REMEMBER_RE
_FABRICATED_AB_SCOPE_RE = FABRICATED_AB_SCOPE_RE
_FABRICATED_BEAD_ID_RE = FABRICATED_BEAD_ID_RE
_BARE_CLAIM_RE = BARE_CLAIM_RE
_AB_DATE_HEADER_RE = AB_DATE_HEADER_RE
_AB_TIER_HEADER_RE = AB_TIER_HEADER_RE
_TOOL_INTENT_RE = TOOL_INTENT_RE
_looks_like_ab_fabrication = looks_like_ab_fabrication


# Public + historically-imported names. `_CATCHER_NAMES` etc. are
# underscore-prefixed "internal" identifiers that the attribution eval
# and CLI stream filter pull from this module; re-declaring them in
# __all__ documents the compatibility surface.
__all__ = [
    "_AB_DATE_HEADER_RE",
    "_AB_TIER_HEADER_RE",
    "_BARE_CLAIM_RE",
    "_CATCHER_NAMES",
    "_DISABLED_CATCHERS",
    "_DUPLICATE_CALL_NUDGE",
    "_EXHAUSTED_FABRICATION_FALLBACK",
    "_FABRICATED_AB_CAPTURE_RE",
    "_FABRICATED_AB_SCOPE_RE",
    "_FABRICATED_BEAD_ID_RE",
    "_FABRICATED_REMEMBER_RE",
    "_FABRICATED_SEARCH_RE",
    "_FALSE_SUCCESS_RE",
    "_META_CONFIRM_RE",
    "_TEASER_RE",
    "_TOOL_INTENT_RE",
    "ConfirmFn",
    "ObserverFn",
    "ToolLoopEvent",
    "ToolLoopResult",
    "_catcher_enabled",
    "_looks_like_ab_fabrication",
    "format_truncated_retry_suffix",
    "run_tool_loop",
]


if TYPE_CHECKING:
    from collections.abc import Iterator

    from harness.plan import Plan
    from harness.router.intent import Router, RouterIntent
    from harness.tools.base import StreamChunk, ToolSpec


# Upper bound on the auto-widen loop triggered by a truncated bail.
# 32k is deep into safe territory for the 131k-window Qwen 2.5 7B we ship —
# the real UX wall shows up well before: ~40-60 tok/s on an M4 Pro means an
# 8k reply already takes 2-3 minutes. See harness-cs9 for the architectural
# follow-up when wrap-ups consistently want > ~4k tokens (chunked output /
# model-splitting); this ceiling is just an anti-runaway guard, not a design
# target.
_MAX_TOKENS_CEILING = 32768
# Bumped from 2 → 3 after airton_f smoke 2026-05-15: the model
# tripped two distinct catchers in sequence (opinion_no_trigger,
# then list_count_mismatch) on a single search-grounding failure,
# exhausting the budget before post_search_grounding could enforce.
# Three retries gives an opinionated character with multiple bail
# catchers enough room to converge on one clean draft.
_BAIL_RETRIES_PER_TURN = 3

# Per-turn discarded-draft capture size for the PreambleLoopHook
# (harness-jwp3). 200 chars is wide enough to span the opening
# sentence + the start of the second sentence on the runaway-preamble
# shape Mark saw on 2026-05-20 ("I need to implement the full GTA2
# browser clone according to the spec. Let me create a proper
# implementation of the game.js file…"). The hook compares the first
# _PREAMBLE_LOOP_MIN_LENGTH chars of each capture so a slight
# variation past the first sentence still counts as a loop.
_PREAMBLE_OPENING_CHARS = 200


def _has_lexicon_hit(user_message: str, lexicon: tuple[str, ...]) -> bool:
    """Word-boundary, case-insensitive presence check for any token in
    `lexicon`. Multi-word entries match with internal whitespace
    collapsed ('class b' against 'Class  B'). Tokens whose first/last
    character is a non-word character (e.g. '§') skip the corresponding
    `\\b` anchor on that side — `\\b` only fires at word/non-word
    transitions, so a bare `§` would never match otherwise. Returns
    False on an empty lexicon — callers consume that as 'gate disabled.'"""
    if not lexicon:
        return False
    import re

    haystack = re.sub(r"\s+", " ", user_message.lower())
    for token in lexicon:
        needle = token.strip().lower()
        if not needle:
            continue
        if " " in needle:
            pattern = re.escape(needle).replace(r"\ ", r"\s+")
        else:
            left = r"\b" if needle[0].isalnum() or needle[0] == "_" else ""
            right = r"\b" if needle[-1].isalnum() or needle[-1] == "_" else ""
            pattern = f"{left}{re.escape(needle)}{right}"
        if re.search(pattern, haystack):
            return True
    return False


def _disabled_snapshot() -> frozenset[str]:
    """Snapshot the module-level disabled set at hook-invocation time.
    Pipeline methods take a frozenset so they can't mutate it; the
    attribution eval toggles the underlying `_DISABLED_CATCHERS` set
    via its context manager, and each hook call re-reads via this
    snapshot (matches the pre-refactor `_catcher_enabled` semantics)."""
    return frozenset(_DISABLED_CATCHERS)


class _ToolCapableAdapter(Protocol):
    """Structural type for adapters that support tool calls — narrower
    than the plain ModelAdapter Protocol so type-checkers know the
    orchestrator needs `complete_with_tools`.

    Adapters that additionally implement `stream_with_tools` get
    token-level streaming through the loop; those that don't fall back
    to the blocking `complete_with_tools` path."""

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply: ...


@dataclass(frozen=True)
class ToolLoopEvent:
    """Emitted synchronously to the optional observer so the CLI can
    print inline status. Kind is one of: router_intent, round_start,
    model_call_start, token_delta, model_call_end, tool_call_start,
    tool_call_end, tool_call_failed, tool_call_declined,
    tool_call_deduped, truncated_retry, bail_retry, round_complete,
    scope_redirected, meta_round, wrap_up_forced.

    `meta_round` (harness-rlza) fires when an iteration's only tool
    calls were meta-tools (tool_search / load_tool / introspect /
    spawn_subagent). Such iterations are exempt from the max_rounds
    budget — they're bookkeeping, not work. Observers + evals use
    the event to audit which rounds got the exemption.

    `message_injected` (harness-6fr0) fires when the orchestrator
    drained a user-role ChatMessage from the optional `inbox` callable
    and appended it to the working thread mid-turn (without
    restarting the loop). `delta` carries the injected text — the
    UI already echoed it at submit time, so observers can use the
    event for audit / evals without re-rendering. Drains run at
    iteration boundaries only (top of while and after
    _execute_tool_calls); never between an assistant(tool_calls=…)
    and its tool-role results, which would break the chat template.

    `wrap_up_forced` (harness-0gss) fires once when the loop ran out
    of work budget on a round that emitted tool calls that actually
    ran. The orchestrator runs one additional adapter call with a
    synthesis-only nudge so the user gets a coherent final answer
    instead of the previous round's interim text. Tool calls in
    the wrap-up's reply are stripped — wrap-up is synthesis only.

    `scope_redirected` (harness-8dop) fires at most once per turn,
    before round 0, when the Router pre-pass classified the turn
    `scope=out` AND the persona supplied a redirect template — the
    orchestrator returns the template directly with rounds=0; no
    main-model call.

    `truncated_retry` fires when a wrap-up round stopped mid-stream at
    the token cap and the orchestrator is about to re-run it with a
    doubled budget. Renderers should discard their in-flight stream
    buffer and print a dim marker so the user knows the upcoming reply
    replaces the partial one they just saw, not appends to it
    (harness-6rl). The event carries `budget_before` / `budget_after`
    so renderers can annotate the marker with the budget progression
    (`1024 → 2048`) or a ceiling note when doubling clamped at
    `_MAX_TOKENS_CEILING` (harness-738f).

    `bail_retry` fires when a 0-tool-calls reply tripped a fabrication /
    teaser / meta-confirm / intent catcher and the orchestrator is
    appending a nudge + re-running. Renderers MUST drop the in-flight
    stream buffer (same contract as truncated_retry). Otherwise each
    retry's stream stacks below the last and the user sees two or three
    fabricated paragraphs concatenated under a single turn header
    (harness-24xj).

    `router_intent` fires at most once per turn, before round 0, when a
    Router pre-pass routed to a tool. `call` carries the ToolCall the
    router chose; a tool_call_start/end pair follows immediately as the
    orchestrator executes it itself.

    `tool_call_deduped` fires (in place of tool_call_start+end) when
    the model re-emits a (name, arguments) pair that already ran this
    turn. The call is NOT executed again — the orchestrator appends a
    stock nudge as the tool-role message so the next round sees 'stop
    calling, give the final answer'. `result` carries that nudge so
    the CLI can render an inline 'duplicate; skipped' line.

    `model_call_start` fires just before the adapter is invoked;
    `model_call_end` fires once it returns. When the adapter supports
    streaming, `token_delta` events fire between them, each carrying a
    `delta` of visible text (tool-call tag spans are already masked by
    the adapter). The CLI renders deltas into a live region and drops
    the spinner while tokens are flowing."""

    kind: str
    call: ToolCall | None = None
    result: ToolResult | None = None
    round_index: int = 0
    delta: str | None = None
    # Name of the catcher that produced a `bail_retry` event. Empty
    # string for every other event kind. Populated by the emit site
    # from the BailOutcome's catcher field (Nudge.catcher, set by
    # HookPipeline.run_bail). Lets the CLI / TUI annotate
    # '⋯ discarding draft, retrying (catcher_name)…' so a mysterious
    # retry is diagnosable without re-running with trace logging.
    catcher: str = ""
    # Effective per-round token budget on a `truncated_retry` event,
    # captured BEFORE and AFTER _BailController.on_truncated()
    # doubled the cap (harness-738f). Lets the CLI / TUI render
    # '(1024 → 2048)' inline so the user can tell at a glance
    # whether retries are progressing toward a real ceiling or
    # burning the whole budget on preamble each time. None for every
    # other event kind. When `budget_after == budget_before` the
    # doubling clamped at `_MAX_TOKENS_CEILING` and renderers should
    # show a '; ceiling' annotation instead of an arrow.
    budget_before: int | None = None
    budget_after: int | None = None


def format_truncated_retry_suffix(budget_before: int | None, budget_after: int | None) -> str:
    """Render the parenthetical that follows the
    `truncated, retrying with wider budget` marker (harness-738f).
    Shared between cli.py's Rich renderer and tui/chat_app.py's
    Textual renderer so both surfaces stay in sync.

    Returns:
      - `""` if either budget is None (older event from a non-
        truncation source or a tool_loop running before the field
        was added — defensive).
      - `" (before → after)"` on the normal doubling path.
      - `" (after; ceiling)"` when the doubling clamped at
        `_MAX_TOKENS_CEILING` so the values are equal — useful
        signal that the next retry won't get any wider, which
        means a fourth truncation will exhaust the bail budget."""
    if budget_before is None or budget_after is None:
        return ""
    if budget_after > budget_before:
        return f" ({budget_before} → {budget_after})"
    return f" ({budget_after}; ceiling)"


@dataclass
class ToolLoopResult:
    content: str
    messages: list[ChatMessage]  # full thread including tool turns
    rounds: int
    events: list[ToolLoopEvent] = field(default_factory=list)

    @property
    def tool_results(self) -> list[ToolResult]:
        """Every ToolResult produced this turn, in dispatch order.
        Derived from `events` (every tool dispatch site emits a
        *_end / *_failed / *_declined / *_deduped event carrying the
        result). Consumed by the per-turn audit log (harness-ywp.2)
        so cli.py can aggregate retrieval scores + grounded citations
        across multiple tool calls without walking the event list
        itself."""
        return [e.result for e in self.events if e.result is not None]

    @property
    def retrieval_top_score(self) -> float | None:
        """Max hit-score across every tool call this turn. None when
        no tool call surfaced retrieval metadata. Used by the
        low-confidence fallback hook (harness-ywp.3) and the audit
        log to summarise 'how confident was any grounding tool'."""
        scores: list[float] = []
        for r in self.tool_results:
            if r.hits:
                scores.append(max(h.score for h in r.hits))
        return max(scores) if scores else None

    @property
    def citations_grounded(self) -> frozenset[str]:
        """Union of citations_grounded across every tool call this
        turn. Used by the audit log + low-confidence fallback."""
        return frozenset().union(*(r.citations_grounded for r in self.tool_results))

    @property
    def tools_ran(self) -> frozenset[str]:
        """Names of tools that produced a result this turn, including
        failed / declined / deduped calls. Matches the semantics the
        fabrication catchers already use via FinalizeContext.tools_ran."""
        return frozenset(r.tool_name for r in self.tool_results)


ConfirmFn = Callable[[ToolCall], bool]
ObserverFn = Callable[[ToolLoopEvent], None]


def _last_user_message(messages: Iterable[ChatMessage]) -> str | None:
    """Walk the thread backwards and return the content of the most
    recent user-role message, or None if there isn't one. The router
    classifies on this single message — the system prompt, history,
    and any prior assistant turns are the main model's job."""
    for m in reversed(list(messages)):
        if m.role == "user":
            return m.content
    return None


def _router_prelude(
    intent: RouterIntent | None,
    working: list[ChatMessage],
    registry: ToolRegistry,
    confirm: ConfirmFn | None,
    emit: Callable[[ToolLoopEvent], None],
    seen_calls: dict[tuple[str, str], ToolResult],
    hooks: HookPipeline,
    user_message: str | None,
    succeeded_tools: set[str],
    attempted_calls: dict[tuple[str, str], int] | None = None,
) -> tuple[bool, bool]:
    """On a usable router intent, append a synthetic assistant tool-
    call turn + the tool result to `working` in place. Returns
    (routed, succeeded) — `routed` is True if routing produced a tool
    execution (main model enters wrap-up mode directly), `succeeded`
    is True iff that tool returned ToolResult.success=True. The main
    loop needs both signals: routed-but-failed still counts as
    "tool executed" for the wrap-up token cap but NOT for disarming
    fabrication catchers (see harness-a0y). On success, `succeeded_tools`
    gains the router-executed tool name so downstream narrow gates can
    see it.

    Caller pre-classifies (harness-8dop) so the scope verdict on the
    same intent can be inspected before tool dispatch — `intent=None`
    here means classify() returned None or the caller has chosen not
    to run a prelude.

    Conservative guards: read-tier tools only (write-tier needs the
    main model's richer context + its own confirmation UX), intent
    tool must exist in the registry, all required arguments must be
    present. Any failure falls through silently — the router is
    advisory, never blocking.

    The pre_tool hook pipeline runs before the router-chosen call
    executes — so the argument-grounding hook gets a shot at rejecting
    a router misfire (query entities that don't trace back to what the
    user said) just like it does for main-model calls. A Skip outcome
    here falls through (returns `(False, False)`) so the main loop
    handles the turn instead of letting the main model wrap prose
    around a leaked-entity tool result.

    `seen_calls` is mutated: on success, the router's (name, args-json)
    key is recorded so a downstream main-model re-invocation with
    identical arguments gets short-circuited by the main loop's
    duplicate guard."""
    from harness.orchestrator.hooks import _call_key

    if user_message is None:
        return (False, False)
    if intent is None or intent.tool_name is None:
        return (False, False)
    if intent.tool_name not in registry:
        return (False, False)
    spec = registry.get(intent.tool_name).spec
    if spec.tier != "read":
        return (False, False)
    required = spec.parameters.get("required", []) or []
    if any(key not in intent.arguments for key in required):
        return (False, False)

    call = ToolCall(name=intent.tool_name, arguments=dict(intent.arguments))
    # Run pre_tool hooks BEFORE announcing the router_intent — a Skip
    # outcome (grounding rejection) should fall through silently so the
    # user sees the main-model path, not a "router picked X, discarded"
    # trace. The router is advisory and its misfires are noise.
    if attempted_calls is None:
        attempted_calls = {}
    pre_outcome = hooks.run_pre_tool(
        PreToolContext(
            call=call,
            seen_calls=dict(seen_calls),
            attempted_calls=dict(attempted_calls),
            user_message=user_message,
            prior_tool_outputs=tuple(m.content for m in working if m.role == "tool"),
        ),
        disabled=_disabled_snapshot(),
    )
    if isinstance(pre_outcome, Skip):
        return (False, False)
    # Increment AFTER pre_tool dispatch — the catcher saw PRIOR-only
    # state. The router's emission now contributes to the count for
    # any later catcher invocation this turn (harness-delk).
    attempted_calls[_call_key(call)] = attempted_calls.get(_call_key(call), 0) + 1
    emit(ToolLoopEvent(kind="router_intent", call=call, round_index=0))
    emit(ToolLoopEvent(kind="tool_call_start", call=call, round_index=0))
    if confirm is not None and spec.tier == "write" and not confirm(call):
        # Unreachable under the read-tier guard above, but kept for
        # symmetry with the main loop's confirmation path.
        result = ToolResult(
            tool_name=call.name,
            output="user declined to approve this tool call",
            success=False,
            error="user_declined",
        )
        emit(ToolLoopEvent(kind="tool_call_declined", call=call, result=result, round_index=0))
    else:
        result = registry.call(call.name, call.arguments)
        kind = "tool_call_end" if result.success else "tool_call_failed"
        emit(ToolLoopEvent(kind=kind, call=call, result=result, round_index=0))

    seen_calls[_call_key(call)] = result
    working.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
    working.append(ChatMessage(role="tool", content=result.output, name=call.name))
    if result.success:
        succeeded_tools.add(call.name)
    return (True, result.success)


_FORCED_SEARCH_MEMORY = "search_memory"
_FORCED_ASSEMBLE_CONTEXT = "assemble_context"


# Tool-use rules nudge — two clauses, one system message
# (harness-b7yd + harness-111v). Prepended as a single system-role
# message at the top of every tool-using turn. Two failure modes
# share the same shape ("model terminates too early") so they share
# one preventive message; bundling them keeps prompt overhead low
# and avoids cascading subagent-preamble index shifts.
#
# Clause 1 — synthesis verbs (harness-b7yd): model calls a
# data-gathering tool, receives results, and exits the loop with a
# verbatim regurgitation even when the prompt asked for "rank" /
# "prioritize" / "compare" / "summarize". Defensive layer is the
# RawResultsDump catcher (harness-s451).
#
# Clause 2 — multi-part prompts (harness-111v): user asks for X AND
# Y, model answers X and gives up on Y instead of issuing another
# tool call. Defensive layer is the IncompleteMultipart catcher.
#
# Injected only when the registry exposes at least one tool;
# tool-less turns can't trip either failure mode.
_TOOL_USE_RULES_NUDGE = (
    "Tool-use rules. "
    "(1) Synthesis verbs — when the user's request contains a synthesis "
    "verb (rank, prioritize, compare, summarize, group, score, categorize, "
    "order, sort, filter), do not terminate after the data-gathering tool "
    "call. The tool result is input, not output — continue the turn and "
    "produce the requested synthesis (ranked / grouped / scored / etc.) "
    "before ending. "
    "(2) Multi-part prompts — when the user's request contains multiple "
    "distinct asks (two questions in one turn, 'X. then Y?', 'find X and "
    "what is Y?'), each sub-ask gets its own tool budget. If a source "
    "covered X but not Y, issue ANOTHER tool call targeting Y (different "
    "query, different URL, different tool) before terminating. Only "
    "return a partial answer after multiple genuine attempts. "
    "(3) Named-constraint verification — when the user's request carries "
    "a constraint (a named region like 'South American', a time window "
    "like 'since 2020', a category like 'Republican senators') AND asks "
    "for a top/most/largest/highest pick, verify BEFORE answering that "
    "your pick satisfies the constraint (a named country is actually IN "
    "the named region) AND is genuinely #1 within that constraint, not "
    "#1 globally. If the source returns a broader ranking, FILTER it by "
    "the constraint first; do not pick the global leader by default."
)


# Wrap-up nudge (harness-0gss). Appended as a user-role message just
# before the forced wrap-up model call when the loop ran out of work
# budget mid-investigation. Tells the model: stop calling tools,
# synthesize what you have, be honest about what you couldn't find.
# Without this nudge — and the wrap-up round it gates — the user would
# see the previous round's interim text ("Let me try another source")
# as the turn's final reply because the last round ended on a tool
# call that never got synthesized.
_WRAP_UP_NUDGE = (
    "You have used your tool-call budget for this turn. This is your "
    "final round — do NOT call another tool. Synthesize the data you "
    "have gathered into a final reply, citing what each source said. "
    "If your request had multiple parts and you found some but not "
    "all, say plainly which parts you could not find. Do not promise "
    "more searches or say you will continue investigating — there is "
    "no more budget."
)


def _forced_assemble_context_prelude(
    working: list[ChatMessage],
    registry: ToolRegistry,
    emit: Callable[[ToolLoopEvent], None],
    seen_calls: dict[tuple[str, str], ToolResult],
    user_message: str | None,
    succeeded_tools: set[str],
    role: str,
) -> bool:
    """Inject a mandatory `assemble_context` call at turn start
    (harness-jkmk). Sibling of `_forced_search_memory_prelude` for the
    contract-shaped retrieval path.

    Arguments passed:
      - `role` — the character's `default_contract_role`. Names which
        contract YAML to resolve.
      - `variables` — `{request_summary: <user_message>}`. The contract
        slots' query templates reference `{request_summary}` by
        convention. Other variables a contract needs (customer_id,
        flight_id, …) aren't known at forced-call time — those contracts
        are not candidates for `require_assemble_context` today.

    Graceful degradation: skipped on empty user_message or when
    `assemble_context` isn't in the registry (e.g. `--tool-set minimal`).
    The character flag is advisory, not load-bearing.

    `succeeded_tools` mutated with `'assemble_context'` regardless of the
    tool's own success signal — same rationale as the search_memory
    prelude (UngroundedCitationHook gates on 'did a grounding tool
    run', not 'did it return hits'). Returns True if the forced call
    actually ran.
    """
    from harness.orchestrator.hooks import _call_key

    if user_message is None or not user_message.strip():
        return False
    if _FORCED_ASSEMBLE_CONTEXT not in registry:
        return False
    call = ToolCall(
        name=_FORCED_ASSEMBLE_CONTEXT,
        arguments={"role": role, "variables": {"request_summary": user_message}},
    )
    emit(ToolLoopEvent(kind="tool_call_start", call=call, round_index=0))
    result = registry.call(call.name, call.arguments)
    kind = "tool_call_end" if result.success else "tool_call_failed"
    emit(ToolLoopEvent(kind=kind, call=call, result=result, round_index=0))
    seen_calls[_call_key(call)] = result
    working.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
    working.append(ChatMessage(role="tool", content=result.output, name=call.name))
    succeeded_tools.add(_FORCED_ASSEMBLE_CONTEXT)
    return True


def _forced_search_memory_prelude(
    working: list[ChatMessage],
    registry: ToolRegistry,
    emit: Callable[[ToolLoopEvent], None],
    seen_calls: dict[tuple[str, str], ToolResult],
    user_message: str | None,
    succeeded_tools: set[str],
) -> bool:
    """Inject a mandatory `search_memory` tool call at the start of the
    turn (harness-3uh). Runs before the router prelude and before the
    first model round; the synthesized assistant+tool messages are
    appended to `working` so the model's first
    `complete_with_tools` call sees the retrieval result as if the
    model had asked for it.

    Motivation (airton_c1): passive retrieval with a 0.5 cosine floor
    misses all lay-language phrasings of JO 7110.65 queries, so no
    memory block attaches and the model fabricates from parametric
    weights. The forced call guarantees grounding regardless of score
    and lights up UngroundedCitationHook's `tools_ran` signal so the
    finalize phase can refuse fabricated citations.

    Graceful degradation: if `search_memory` isn't in the active
    registry (e.g. `--tool-set minimal`), we log nothing and skip
    injection — the character flag is advisory, not load-bearing, and a
    tool-set without memory tools should still answer. Empty
    user_message also skips (rare: system-only bootstrap).

    `succeeded_tools` is mutated unconditionally with
    `'search_memory'` so the downstream grounding signal
    (UngroundedCitationHook) sees this call even if the tool returned a
    no-matches sentinel. The registry call-path still reports success
    accurately for telemetry via the `tool_call_end` /
    `tool_call_failed` event kind.

    Returns True if the forced call actually ran, False otherwise.
    The return value isn't load-bearing today (the main loop gates on
    `succeeded_tools` / working-history tool messages), but callers
    may want it for debugging."""
    from harness.orchestrator.hooks import _call_key

    if user_message is None or not user_message.strip():
        return False
    if _FORCED_SEARCH_MEMORY not in registry:
        return False
    call = ToolCall(name=_FORCED_SEARCH_MEMORY, arguments={"query": user_message})
    emit(ToolLoopEvent(kind="tool_call_start", call=call, round_index=0))
    result = registry.call(call.name, call.arguments)
    kind = "tool_call_end" if result.success else "tool_call_failed"
    emit(ToolLoopEvent(kind=kind, call=call, result=result, round_index=0))
    seen_calls[_call_key(call)] = result
    working.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
    working.append(ChatMessage(role="tool", content=result.output, name=call.name))
    # Always register the forced call as a grounding signal, even if the
    # tool returned a no-matches sentinel (success=True with empty
    # body) or an error. Rationale: UngroundedCitationHook gates on
    # "did a grounding tool run this turn" — we ran one, and refusing
    # this signal because the store was empty would defeat the point
    # of forcing it in the first place.
    succeeded_tools.add(_FORCED_SEARCH_MEMORY)
    return True


class _BailController:
    """Owns the retry budget + current token caps for a tool-loop
    turn. Pulled out of run_tool_loop (harness-z4ev) so the main
    orchestration isn't juggling three mutable ints inline.

    `on_truncated()` widens both caps (main + wrap-up) because
    wrap-up rounds clamp to `min(main, wrap_up)` — doubling only
    the main cap still re-truncates at the old wrap-up cap
    (harness-jly).

    `consume_retry()` decrements the retry budget. Returns True
    when a retry is still allowed; False once exhausted."""

    def __init__(self, max_tokens: int, wrap_up_max_tokens: int) -> None:
        self.current_max_tokens = max_tokens
        self.current_wrap_up_max_tokens = wrap_up_max_tokens
        self._retries = _BAIL_RETRIES_PER_TURN
        # Per-turn ledger of discarded-draft openings (harness-jwp3).
        # Each entry is `reply.content[:_PREAMBLE_OPENING_CHARS]` for a
        # reply that just got a bail Nudge or Truncated outcome. The
        # PreambleLoopHook compares the current reply's opening against
        # this list to detect 'same intent restatement, retried with a
        # wider budget' loops where doubling the budget buys longer
        # preamble, not progress.
        self.discarded_openings: list[str] = []

    def round_max_tokens(self, tools_already_ran: bool) -> int:
        """Post-tool rounds are wrap-up rounds — tighter cap."""
        if tools_already_ran:
            return min(self.current_max_tokens, self.current_wrap_up_max_tokens)
        return self.current_max_tokens

    def on_truncated(self) -> None:
        self.current_max_tokens = min(self.current_max_tokens * 2, _MAX_TOKENS_CEILING)
        self.current_wrap_up_max_tokens = min(
            self.current_wrap_up_max_tokens * 2, _MAX_TOKENS_CEILING
        )

    def consume_retry(self) -> bool:
        """Decrement the retry budget. Returns True if a retry can
        still fire after this consumption, False if exhausted."""
        if self._retries <= 0:
            return False
        self._retries -= 1
        return True

    @property
    def retries_left(self) -> int:
        return self._retries


def _run_model_round(
    adapter: _ToolCapableAdapter,
    working: list[ChatMessage],
    registry: ToolRegistry,
    *,
    round_max_tokens: int,
    temperature: float,
    round_idx: int,
    emit: Callable[[ToolLoopEvent], None],
    hooks: HookPipeline,
) -> ModelReply:
    """Call the adapter for one round. Streams via `stream_with_tools`
    when available (emitting token_delta events); falls back to
    blocking `complete_with_tools` otherwise. Emits model_call_start
    / model_call_end around the call. Runs the post-model hook pass
    (strips paired-meta-confirm narrative when the reply also carries
    a valid tool call — harness-fup / harness-q27)."""
    emit(ToolLoopEvent(kind="model_call_start", round_index=round_idx))
    stream_fn = getattr(adapter, "stream_with_tools", None)
    try:
        if callable(stream_fn):
            stream_iter: Iterator[StreamChunk] = stream_fn(
                working,
                tools=registry.specs(),
                max_tokens=round_max_tokens,
                temperature=temperature,
            )
            reply: ModelReply | None = None
            for chunk in stream_iter:
                if isinstance(chunk, StreamText):
                    emit(
                        ToolLoopEvent(
                            kind="token_delta",
                            delta=chunk.text,
                            round_index=round_idx,
                        )
                    )
                elif isinstance(chunk, StreamComplete):
                    reply = chunk.reply
            if reply is None:
                raise RuntimeError("stream_with_tools exhausted without StreamComplete")
            last_reply = reply
        else:
            last_reply = adapter.complete_with_tools(
                working,
                tools=registry.specs(),
                max_tokens=round_max_tokens,
                temperature=temperature,
            )
    finally:
        emit(ToolLoopEvent(kind="model_call_end", round_index=round_idx))

    outcome = hooks.run_post_model(
        PostModelContext(reply=last_reply), disabled=_disabled_snapshot()
    )
    if isinstance(outcome, Replace):
        return outcome.reply
    return last_reply


def _execute_tool_calls(
    calls: Iterable[ToolCall],
    registry: ToolRegistry,
    working: list[ChatMessage],
    seen_calls: dict[tuple[str, str], ToolResult],
    *,
    confirm: ConfirmFn | None,
    emit: Callable[[ToolLoopEvent], None],
    round_idx: int,
    hooks: HookPipeline,
    user_message: str | None,
    succeeded_tools: set[str],
    attempted_calls: dict[tuple[str, str], int] | None = None,
) -> bool:
    """Execute the round's tool calls: in-round dedup, duplicate-call
    hook (cross-round), write-tier confirm, dispatch, append tool-role
    messages. Returns True if any call returned success=True (used to
    gate fabrication catchers on later rounds). Mutates `succeeded_tools`
    with the names of tools whose calls succeeded — lets narrow gates
    (e.g. fabricated_search only disarming on web-fetch tools) inspect
    which specific tools ran this turn.

    In-round dedup (harness-69f2): when a single model reply emits the
    same (name, args) twice, run it once. The dropped duplicates never
    reach DuplicateCallHook, so the model doesn't see a confusing
    'duplicate call rejected' nudge on its first visible attempt. A
    `tool_call_deduped` event fires for telemetry but no tool-role
    message is appended for the dropped twin.

    Cross-round dedup stays — the hook fires on a real prior call and
    re-issues that prior result preserving success/error (harness-v5w),
    not a fixed success=True nudge."""
    from harness.orchestrator.hooks import _call_key

    # In-round dedup: keep the first call of each duplicate set.
    # Iterating once is fine — typical tool-call batches are short.
    deduped: list[ToolCall] = []
    in_round_keys: set[tuple[str, str]] = set()
    for call in calls:
        key = _call_key(call)
        if key in in_round_keys:
            emit(
                ToolLoopEvent(
                    kind="tool_call_deduped",
                    call=call,
                    result=ToolResult(
                        tool_name=call.name,
                        output="(in-round duplicate — collapsed silently)",
                        success=True,
                    ),
                    round_index=round_idx,
                )
            )
            continue
        in_round_keys.add(key)
        deduped.append(call)

    any_success = False
    # attempted_calls counts PRIOR emissions (not including the current
    # call) — incremented at the END of each iteration so the catcher's
    # mental model is 'how many previous attempts' (harness-delk).
    # In-round duplicates were collapsed above; deduped here is the
    # unique-per-round set.
    if attempted_calls is None:
        attempted_calls = {}
    for call in deduped:
        key = _call_key(call)
        pre_outcome = hooks.run_pre_tool(
            PreToolContext(
                call=call,
                seen_calls=dict(seen_calls),
                attempted_calls=dict(attempted_calls),
                user_message=user_message,
                prior_tool_outputs=tuple(m.content for m in working if m.role == "tool"),
            ),
            disabled=_disabled_snapshot(),
        )
        # Increment AFTER pre_tool dispatch — the catcher saw the
        # PRIOR-only state. This emission now contributes to the count
        # the NEXT call's catcher will see.
        attempted_calls[key] = attempted_calls.get(key, 0) + 1
        if isinstance(pre_outcome, Skip):
            # Cross-round duplicate (or grounding rejection). For
            # duplicate_call this carries the prior ToolResult re-issued
            # with a prefix; for other Skip-emitting hooks (grounding)
            # it carries the hook's own ToolResult.
            result = pre_outcome.result
            emit(
                ToolLoopEvent(
                    kind="tool_call_deduped",
                    call=call,
                    result=result,
                    round_index=round_idx,
                )
            )
            working.append(ChatMessage(role="tool", content=result.output, name=call.name))
            continue

        emit(ToolLoopEvent(kind="tool_call_start", call=call, round_index=round_idx))

        spec = registry.get(call.name).spec if call.name in registry else None
        needs_confirm = confirm is not None and spec is not None and spec.tier == "write"
        if needs_confirm and not confirm(call):  # type: ignore[misc]  # confirm is not None when needs_confirm is True
            result = ToolResult(
                tool_name=call.name,
                output="user declined to approve this tool call",
                success=False,
                error="user_declined",
            )
            kind = "tool_call_declined"
        else:
            result = registry.call(call.name, call.arguments)
            kind = "tool_call_end" if result.success else "tool_call_failed"
        emit(ToolLoopEvent(kind=kind, call=call, result=result, round_index=round_idx))

        # post_tool hooks (sota punch #3) may replace the result's
        # visible text before it lands in the thread — e.g. the
        # summarizer compressing a 12 KB grep dump to 200 tokens.
        # Runs only when we have a spec (we own this tool) and after
        # the tool-success/tool-failed event has fired so observers
        # see the original (non-summarized) outcome for telemetry.
        if spec is not None:
            post_outcome = hooks.run_post_tool(
                PostToolContext(call=call, result=result, spec=spec),
                disabled=_disabled_snapshot(),
            )
            if isinstance(post_outcome, ReplaceResult):
                result = post_outcome.result

        seen_calls[key] = result
        if result.success:
            any_success = True
            succeeded_tools.add(call.name)
        working.append(ChatMessage(role="tool", content=result.output, name=call.name))
    return any_success


def run_tool_loop(
    adapter: _ToolCapableAdapter,
    messages: Iterable[ChatMessage],
    registry: ToolRegistry,
    *,
    max_rounds: int = 8,
    confirm: ConfirmFn | None = None,
    observe: ObserverFn | None = None,
    # Defaults bumped 1024 → 2048 (harness-gt0m). Tool-use turns
    # routinely emit a tool call AND a paragraph of reasoning;
    # 1024 was clipping legitimate work and tripping
    # truncated_retry on routine rounds. 2048 costs ~13s extra
    # wall-clock at 150 tok/s on M4 Pro (still interactive) and
    # stays well under the 4-6K coherence ceiling for Qwen 2.5
    # 7B / Qwen3-Coder-30B. Ceiling at _MAX_TOKENS_CEILING
    # (32768) and the bail-retry budget are unchanged — those
    # are the safety rail, not the operating point.
    max_tokens: int = 2048,
    wrap_up_max_tokens: int = 2048,
    temperature: float = 0.5,
    router: Router | None = None,
    hooks: HookPipeline | None = None,
    memory_block_attached: bool = False,
    force_search_memory: bool = False,
    force_assemble_context: str | None = None,
    banter_tracker: BanterStreakTracker | None = None,
    scope_redirect_template: str | None = None,
    scope_lexicon: tuple[str, ...] = (),
    plan: Plan | None = None,
    inbox: Callable[[], list[ChatMessage]] | None = None,
) -> ToolLoopResult:
    """Drive a model + tool registry until the model emits a text-only
    reply or `max_rounds` rounds are spent.

    `confirm(call)` is called for every write-tier tool call. Return
    True to execute, False to refuse — the refusal is fed back to the
    model as a tool-role message so it can adjust.

    `observe(event)` is called synchronously on every state transition
    so a CLI can print inline status.

    `wrap_up_max_tokens` caps generation in rounds that come AFTER a
    tool has already executed this turn. Those rounds are just the
    model restating what changed — they don't need the full 1024-token
    budget. Capping prevents small models from burning 10s+ on a
    'thinking…' spinner generating filler after the work is done.

    `router`, when set, classifies the last user turn once before the
    first model round. On a usable intent (known read-tier tool, all
    required args present), the orchestrator synthesizes the tool call
    itself, executes it, and the main model only sees a wrap-up round.
    This skips the fabrication-and-nudge loop that small adapters fall
    into on factual queries (see harness-j1d / harness-q27).

    `hooks` lets callers supply a custom HookPipeline — useful for
    subagent loops (1.A) that want to share or layer on the parent's
    catchers. Defaults to the module-level pipeline.

    `memory_block_attached` records whether the caller assembled a
    retrieval memory block into the system prompt for this turn. The
    `ungrounded_citation` finalize hook consumes it: a reply with a
    JO-style section citation + no memory block + no grounding tool
    call is a fabricated citation, so the hook replaces the reply with
    a scope-aware refusal. Defaults to False — callers that don't
    plumb the signal leave the hook permissive (only the other two
    signals can disarm it).

    `force_search_memory`, when True, injects a mandatory
    `search_memory` tool call (query = latest user message) before the
    router prelude and before the first model round. Used by
    characters with `require_search_memory: true` in core.yaml (e.g.
    airton_c1, harness-3uh) where passive retrieval routinely misses
    lay-language paraphrases of in-scope queries. Degrades gracefully
    to a no-op if `search_memory` isn't in the registry.

    `force_assemble_context`, when set to a role name, injects a
    mandatory `assemble_context` tool call before the first model round
    (harness-jkmk). The role is the character's `default_contract_role`;
    the variables dict is `{request_summary: <latest user message>}`.
    Used by characters whose primary retrieval path is the contract
    orchestrator (rather than flat search_memory). Degrades gracefully
    to a no-op if `assemble_context` isn't in the registry. Runs after
    `force_search_memory` so a character can opt into both (the contract
    package then sees the search_memory result as part of an episodic
    slot if its YAML wires one).

    `inbox`, when set, is consulted between iterations for new
    user-role messages that the UI accepted mid-turn (harness-6fr0).
    Each call returns a (possibly empty) list of ChatMessages; the
    orchestrator appends them to the working thread as separate
    user turns (no concatenation) and emits one `message_injected`
    event per drained message so observers can audit. Drains run at
    two safe positions: top of the main while loop (before the next
    model round) and immediately after `_execute_tool_calls`
    returns. Never between an assistant(tool_calls=…) and its
    tool-role results — that breaks the Qwen 2.5 chat template.
    `turn_user_message` is captured from `working[:initial_count]`
    and stays frozen to the original prompt, so grounding hooks do
    NOT treat injected text as the turn's subject.

    `banter_tracker`, when set, intercepts banter-shaped user messages
    (epic harness-jjm9) BEFORE any forced grounding, router pass, or
    model round. On a hit, the tracker composes a joke or redirect
    (1-joke-then-3-redirects cycle) and the loop returns immediately
    with a 0-round result — no tokens spent on a hallucinated rule
    chunk for 'this page intentionally left blank'. On a real (non-
    banter) prompt the tracker is reset so the next banter prompt
    starts fresh with a joke."""
    pipeline = hooks if hooks is not None else _DEFAULT_PIPELINE
    working: list[ChatMessage] = list(messages)
    # Inject the active-plan block as a system-role message before
    # the loop opens (harness-uzan). The plan is read-only here —
    # actual mutations happen in the heartbeat plan-revision task
    # (rbj9). Block is prepended so it appears alongside the
    # character's main system prompt rather than after the user turn.
    if plan is not None:
        from harness.plan import render_plan_block

        block = render_plan_block(plan)
        if block.strip():
            working.insert(0, ChatMessage(role="system", content=block))
    # Tool-use rules nudge (harness-b7yd + harness-111v). See
    # _TOOL_USE_RULES_NUDGE for rationale. Gated on registry.active_names()
    # so tool-less turns (e.g. an orchestrator entered with an empty
    # registry as a no-op) don't carry the rule.
    if registry.active_names():
        working.insert(0, ChatMessage(role="system", content=_TOOL_USE_RULES_NUDGE))
    initial_count = len(working)
    events: list[ToolLoopEvent] = []
    last_reply: ModelReply = ModelReply(content="", tool_calls=())
    bail = _BailController(max_tokens=max_tokens, wrap_up_max_tokens=wrap_up_max_tokens)
    # Duplicate-call guard. Small models sometimes wrap a real answer
    # around a redundant re-call ("here's the summary" + same list_dir
    # with same args as a prior round). Tracking (name, args-json) per
    # turn lets us skip execution on repeats (harness-pun). The value
    # is the ToolResult the first call produced — DuplicateCallHook
    # re-issues it on duplicates so a prior failure can't get
    # paraphrased as success (harness-v5w).
    seen_calls: dict[tuple[str, str], ToolResult] = {}
    # Attempt counter (harness-delk). Incremented on EVERY tool_call the
    # model emits, regardless of whether duplicate_call Skips it. Loop-
    # detection catchers (tool_search_loop) read this instead of
    # seen_calls so a dedup-masked loop still trips the threshold.
    attempted_calls: dict[tuple[str, str], int] = {}

    def emit(event: ToolLoopEvent) -> None:
        events.append(event)
        if observe is not None:
            observe(event)

    def drain_inbox(round_idx: int) -> None:
        """Pull any pending mid-turn injections (harness-6fr0) into
        the working thread as separate user-role messages. Each
        drained message becomes its own ChatMessage — the UI
        echoed it at submit time, so we just emit a
        `message_injected` event per item for audit/evals and
        skip empties. Safe only at iteration boundaries; callers
        must not invoke between an assistant(tool_calls=…) and
        its tool-role results."""
        if inbox is None:
            return
        for msg in inbox():
            if not msg.content:
                continue
            working.append(msg)
            emit(
                ToolLoopEvent(
                    kind="message_injected",
                    round_index=round_idx,
                    delta=msg.content,
                )
            )

    # Tracks whether any tool call THIS TURN returned success=True. The
    # fabrication catchers gate on this: if every tool this turn errored,
    # the model has no real data to wrap up, so a completion-style reply
    # is still hallucination (harness-a0y). `succeeded_tools` carries the
    # finer signal — the set of tool names that succeeded — so narrow
    # gates (e.g. fabricated_search disarming only on search_web /
    # fetch_url, not on search_memory) can inspect it (harness-78z).
    any_tool_succeeded = False
    succeeded_tools: set[str] = set()

    # Captured once per turn — the last user message at loop entry is
    # the question we're answering. The grounding hook uses it to reject
    # tool calls whose argument entities trace back to memory / prior
    # turns instead of what the user just asked. Does NOT change when
    # the orchestrator appends its own user-role nudges during
    # bail-retries (those aren't the real user question).
    turn_user_message = _last_user_message(working[:initial_count])

    # Banter intercept (epic harness-jjm9). Fires BEFORE forced search
    # and the router pass — both would otherwise route a "this page
    # intentionally left blank" prompt to search_memory and the model
    # would fabricate a rule chunk from cosine ~0.02 retrieval. The
    # tracker owns the 1-joke-then-3-redirects cycle and the no-repeat
    # joke set; here we just consume one tier and bail with a synthetic
    # ToolLoopResult. On a real prompt we reset the cycle so the next
    # banter prompt starts fresh with a joke.
    if banter_tracker is not None and turn_user_message is not None:
        if is_banter_prompt(turn_user_message):
            return ToolLoopResult(
                content=banter_tracker.consume(),
                messages=working,
                rounds=0,
                events=events,
            )
        banter_tracker.note_real_prompt()

    # Pre-prelude lexical scope gate (harness-8dop option 4). When the
    # character ships a `scope_lexicon` AND the user message contains
    # zero domain-vocabulary hits AND a redirect template is set,
    # short-circuit BEFORE the forced search/assemble preludes or the
    # router call. The motivating failures ("how many fruit bats fit
    # in a cave", "wedding ring on which finger") contain no aviation
    # vocabulary at all — running retrieval on them just burns latency
    # and gives the model fodder for a fabricated cite. The smart
    # router still gets a turn on prompts with mixed signal; this only
    # fires on the obvious-out case.
    if (
        scope_lexicon
        and scope_redirect_template is not None
        and scope_redirect_template.strip()
        and turn_user_message is not None
        and not _has_lexicon_hit(turn_user_message, scope_lexicon)
    ):
        emit(ToolLoopEvent(kind="scope_redirected", round_index=0))
        return ToolLoopResult(
            content=scope_redirect_template.strip(),
            messages=working,
            rounds=0,
            events=events,
        )

    # Forced search_memory injection (harness-3uh). Runs BEFORE the
    # router prelude so the grounding result is already in-thread when
    # the router (if any) classifies the turn. Mutates working /
    # seen_calls / succeeded_tools like any other tool call; the rest
    # of the loop can't tell the difference.
    if force_search_memory:
        forced_ran = _forced_search_memory_prelude(
            working,
            registry,
            emit,
            seen_calls,
            turn_user_message,
            succeeded_tools,
        )
        any_tool_succeeded = any_tool_succeeded or forced_ran

    # Forced assemble_context injection (harness-jkmk). Same shape as
    # the search_memory prelude; runs after it so a character that opts
    # into both gets the search hit AND the contract package staged
    # before the first model round. Skipped when no role is specified
    # (the common case — only contract-first characters opt in).
    if force_assemble_context is not None:
        forced_ran = _forced_assemble_context_prelude(
            working,
            registry,
            emit,
            seen_calls,
            turn_user_message,
            succeeded_tools,
            role=force_assemble_context,
        )
        any_tool_succeeded = any_tool_succeeded or forced_ran

    # Router pre-pass: classify ONCE so both the scope short-circuit
    # (harness-8dop) and the tool-routing prelude consume the same
    # verdict — no duplicate Hermes call. classify() returns None on
    # parse / adapter failure; downstream guards already tolerate it.
    router_intent: RouterIntent | None = None
    if router is not None and turn_user_message is not None:
        router_intent = router.classify(turn_user_message, registry.specs())

    # Scope short-circuit. When the router confidently classifies the
    # turn as `out` AND the persona supplies a redirect template,
    # bail with the canned reply — no main-model call, no fabricated
    # citation. `unsure` and `in` both fall through to the normal
    # flow; an absent template is treated as "scope gate disabled."
    # Note: the pre-prelude lexical gate above already handled the
    # zero-lexicon-hit prompts, so a router `out` here is the smart-
    # router's harder call (some aviation vocabulary but actually
    # out-of-scope, e.g. pilot-side wake-turbulence material).
    if (
        router_intent is not None
        and router_intent.scope == "out"
        and scope_redirect_template is not None
        and scope_redirect_template.strip()
    ):
        emit(ToolLoopEvent(kind="scope_redirected", round_index=0))
        return ToolLoopResult(
            content=scope_redirect_template.strip(),
            messages=working,
            rounds=0,
            events=events,
        )

    if router is not None:
        _, router_success = _router_prelude(
            router_intent,
            working,
            registry,
            confirm,
            emit,
            seen_calls,
            pipeline,
            turn_user_message,
            succeeded_tools,
            attempted_calls=attempted_calls,
        )
        any_tool_succeeded = any_tool_succeeded or router_success

    # Meta-tool exemption (harness-rlza). Meta-tools (tool_search,
    # load_tool, introspect, spawn_subagent) are discovery /
    # bookkeeping primitives — they help the agent figure out WHAT
    # to do, not DO it. Charging them against the work budget
    # leaves multi-step / multi-part queries starved (the 2026-05-19
    # Nairobi multi-part repro burned 3 of 8 rounds on discovery
    # before the actual data-gathering started). The exemption
    # tracks work_rounds separately from total_iterations: only
    # rounds that emitted a content tool count against max_rounds;
    # meta-only rounds are free. A hard ceiling (2 * max_rounds)
    # prevents a malformed agent from looping forever on meta calls.
    from harness.orchestrator.hooks import _META_TOOLS

    work_rounds = 0
    total_iterations = 0
    hard_ceiling = max(2 * max_rounds, max_rounds + 1)
    while work_rounds < max_rounds and total_iterations < hard_ceiling:
        round_idx = total_iterations
        total_iterations += 1
        # Drain mid-turn injections (harness-6fr0) BEFORE the model
        # round so the next model call sees the new user messages
        # in its context. Safe here — any prior round's tool-role
        # results were appended contiguously after their
        # assistant(tool_calls=…) on the previous iteration.
        drain_inbox(round_idx)
        tools_already_ran = any(m.role == "tool" for m in working[initial_count:])
        emit(ToolLoopEvent(kind="round_start", round_index=round_idx))
        last_reply = _run_model_round(
            adapter,
            working,
            registry,
            round_max_tokens=bail.round_max_tokens(tools_already_ran),
            temperature=temperature,
            round_idx=round_idx,
            emit=emit,
            hooks=pipeline,
        )

        if not last_reply.tool_calls:
            # Gate on successful tool execution, not mere execution: an
            # all-errored turn (e.g. plan with invalid scope) must still
            # trip fabrication catchers (harness-a0y).
            bail_outcome: BailOutcome = pipeline.run_bail(
                BailContext(
                    reply=last_reply,
                    tools_ran_this_turn=any_tool_succeeded,
                    tools_ran=frozenset(succeeded_tools),
                    user_message=turn_user_message,
                    prior_tool_outputs=tuple(m.content for m in working if m.role == "tool"),
                    discarded_openings=tuple(bail.discarded_openings),
                ),
                disabled=_disabled_snapshot(),
            )
            # Bail retries are an inner-loop concern bounded by
            # `_BAIL_RETRIES_PER_TURN` (3 by default). We additionally
            # gate on `total_iterations < hard_ceiling` so a pathological
            # bail-retry storm can't outrun the safety ceiling.
            can_retry = bail.retries_left > 0 and total_iterations < hard_ceiling
            if not isinstance(bail_outcome, Continue) and can_retry:
                bail.consume_retry()
                # Record the discarded opening BEFORE the retry fires so
                # PreambleLoopHook on the next round can compare against
                # it (harness-jwp3). Bounds the capture to
                # _PREAMBLE_OPENING_CHARS — the rest of the discarded
                # draft is gone with the retry anyway.
                bail.discarded_openings.append(last_reply.content[:_PREAMBLE_OPENING_CHARS])
                if isinstance(bail_outcome, Truncated):
                    # Capture the effective per-round budget before
                    # and after the doubling so the renderer can show
                    # `(1024 → 2048)` inline (harness-738f). Equal
                    # values mean we clamped at _MAX_TOKENS_CEILING —
                    # renderers signal that distinctly.
                    budget_before = bail.round_max_tokens(tools_already_ran)
                    bail.on_truncated()
                    budget_after = bail.round_max_tokens(tools_already_ran)
                    emit(
                        ToolLoopEvent(
                            kind="truncated_retry",
                            round_index=round_idx,
                            budget_before=budget_before,
                            budget_after=budget_after,
                        )
                    )
                else:
                    # Must fire BEFORE the nudge is queued so the CLI /
                    # TUI can drop the in-flight stream buffer — each
                    # retry re-streams from scratch (harness-24xj).
                    emit(
                        ToolLoopEvent(
                            kind="bail_retry",
                            round_index=round_idx,
                            catcher=bail_outcome.catcher,
                        )
                    )
                    working.append(ChatMessage(role="user", content=bail_outcome.text))
                continue
            # Retries exhausted (or none needed). Let finalize hooks
            # decide whether to substitute a canned fallback —
            # fabrication_fallback fires only when THIS round's reply
            # still trips a fabrication-shaped Nudge. A Continue outcome
            # (legitimate final reply) disarms the fallback; a Truncated
            # outcome keeps the partial reply.
            # Tool outputs for this turn, in execution order. Pulled
            # from the working thread's role="tool" messages — every
            # tool execution (and dedup-skip) appends one. Feeds
            # table_fabrication's verbatim-row check (harness-5uq).
            turn_tool_outputs = tuple(m.content for m in working if m.role == "tool" and m.content)
            # Aggregate retrieval-side telemetry for the finalize phase
            # (harness-ywp.3 consumer). Walks the event log — every
            # tool dispatch site emits an event with the ToolResult, so
            # this covers router prelude + forced search_memory +
            # main-loop calls uniformly.
            turn_tool_results = [e.result for e in events if e.result is not None]
            turn_top_score: float | None = None
            for _r in turn_tool_results:
                if _r.hits:
                    _local_top = max(h.score for h in _r.hits)
                    if turn_top_score is None or _local_top > turn_top_score:
                        turn_top_score = _local_top
            turn_citations_grounded = frozenset().union(
                *(_r.citations_grounded for _r in turn_tool_results)
            )
            finalize_outcome = pipeline.run_finalize(
                FinalizeContext(
                    reply=last_reply,
                    last_outcome=bail_outcome,
                    tools_ran=frozenset(succeeded_tools),
                    memory_block_attached=memory_block_attached,
                    tool_outputs=turn_tool_outputs,
                    retrieval_top_score=turn_top_score,
                    citations_grounded=turn_citations_grounded,
                ),
                disabled=_disabled_snapshot(),
            )
            if isinstance(finalize_outcome, Halt):
                last_reply = finalize_outcome.reply
            emit(ToolLoopEvent(kind="round_complete", round_index=round_idx))
            return ToolLoopResult(
                content=last_reply.content,
                messages=working,
                rounds=total_iterations,
                events=events,
            )

        # Record the assistant's tool-call turn so the model can re-read it
        # on the next round via the chat template.
        working.append(
            ChatMessage(
                role="assistant",
                content=last_reply.content,
                tool_calls=last_reply.tool_calls,
            )
        )
        round_success = _execute_tool_calls(
            last_reply.tool_calls,
            registry,
            working,
            seen_calls,
            confirm=confirm,
            emit=emit,
            round_idx=round_idx,
            hooks=pipeline,
            user_message=turn_user_message,
            succeeded_tools=succeeded_tools,
            attempted_calls=attempted_calls,
        )
        any_tool_succeeded = any_tool_succeeded or round_success
        # Second drain (harness-6fr0). All tool-role results for the
        # current round are now appended; injecting a user message
        # here is template-safe (assistant + N tool messages are
        # contiguous above). The next iteration's drain at the top
        # of the loop would catch this too — draining here just
        # means the user's message is in the working thread before
        # work-round accounting decides whether to continue.
        drain_inbox(round_idx)

        # Work-round accounting (harness-rlza). A round counts as "work"
        # only if at least one of the model's tool calls named a content
        # tool. Meta-only rounds (tool_search → load_tool sequences,
        # introspect lookups, depth-1 subagent dispatch) don't burn the
        # max_rounds budget — they're free passes. Emit a meta_round
        # event for observers + evals so the exemption is auditable.
        is_work_round = any(call.name not in _META_TOOLS for call in last_reply.tool_calls)
        if is_work_round:
            work_rounds += 1
        else:
            emit(ToolLoopEvent(kind="meta_round", round_index=round_idx))

    # Wrap-up round (harness-0gss). If the loop exited mid-investigation
    # — work budget hit AND the last round emitted a tool call that
    # actually ran — force ONE more model round (no new tools allowed)
    # so the model can synthesize the data it gathered. Without this,
    # callers see the previous round's interim text ('Let me try
    # another source') as 'final' because the last round ended on a
    # tool call that never got synthesized.
    #
    # Hard ceiling still applies — we must leave room for one more
    # adapter call, and only an honest mid-investigation exit (some
    # content tool ran) deserves the wrap-up.
    wrap_up_eligible = (
        bool(last_reply.tool_calls)
        and any_tool_succeeded
        and work_rounds >= max_rounds
        and total_iterations < hard_ceiling
    )
    if wrap_up_eligible:
        emit(ToolLoopEvent(kind="wrap_up_forced", round_index=total_iterations))
        working.append(ChatMessage(role="user", content=_WRAP_UP_NUDGE))
        wrap_up_reply = _run_model_round(
            adapter,
            working,
            registry,
            round_max_tokens=wrap_up_max_tokens,
            temperature=temperature,
            round_idx=total_iterations,
            emit=emit,
            hooks=pipeline,
        )
        wrap_up_round_idx = total_iterations
        total_iterations += 1
        # Strip any tool_calls — wrap-up is synthesis only. If the model
        # emitted text + tool_call, keep the text; if tool_call only,
        # content stays empty and the canned fallback below covers it.
        last_reply = ModelReply(
            content=wrap_up_reply.content,
            tool_calls=(),
            was_truncated=wrap_up_reply.was_truncated,
            had_unparseable_call=wrap_up_reply.had_unparseable_call,
        )
        # Run bail + finalize on the wrap-up reply. No retry budget at
        # this point — bail outcome only feeds fabrication_fallback so
        # a fabricated wrap-up still gets the canned refusal substitution
        # instead of leaking through.
        wrap_bail_outcome: BailOutcome = pipeline.run_bail(
            BailContext(
                reply=last_reply,
                tools_ran_this_turn=any_tool_succeeded,
                tools_ran=frozenset(succeeded_tools),
                user_message=turn_user_message,
                prior_tool_outputs=tuple(m.content for m in working if m.role == "tool"),
                discarded_openings=tuple(bail.discarded_openings),
            ),
            disabled=_disabled_snapshot(),
        )
        wrap_tool_outputs = tuple(m.content for m in working if m.role == "tool" and m.content)
        wrap_tool_results = [e.result for e in events if e.result is not None]
        wrap_top_score: float | None = None
        for _wr in wrap_tool_results:
            if _wr.hits:
                _wlocal_top = max(h.score for h in _wr.hits)
                if wrap_top_score is None or _wlocal_top > wrap_top_score:
                    wrap_top_score = _wlocal_top
        wrap_citations_grounded = frozenset().union(
            *(_wr.citations_grounded for _wr in wrap_tool_results)
        )
        wrap_finalize_outcome = pipeline.run_finalize(
            FinalizeContext(
                reply=last_reply,
                last_outcome=wrap_bail_outcome,
                tools_ran=frozenset(succeeded_tools),
                memory_block_attached=memory_block_attached,
                tool_outputs=wrap_tool_outputs,
                retrieval_top_score=wrap_top_score,
                citations_grounded=wrap_citations_grounded,
            ),
            disabled=_disabled_snapshot(),
        )
        if isinstance(wrap_finalize_outcome, Halt):
            last_reply = wrap_finalize_outcome.reply
        emit(ToolLoopEvent(kind="round_complete", round_index=wrap_up_round_idx))

    # Loop exhausted — return what we have. `rounds` reports the
    # total iterations spent (work + meta + optional wrap-up) so
    # callers and tests see the real model-invocation count, not the
    # work-round projection.
    return ToolLoopResult(
        content=last_reply.content or "[tool loop exhausted without final reply]",
        messages=working,
        rounds=total_iterations,
        events=events,
    )
