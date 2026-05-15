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
    "run_tool_loop",
]


if TYPE_CHECKING:
    from collections.abc import Iterator

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
    scope_redirected.

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
    (harness-6rl).

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
    seen_calls: set[tuple[str, str]],
    hooks: HookPipeline,
    user_message: str | None,
    succeeded_tools: set[str],
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
    pre_outcome = hooks.run_pre_tool(
        PreToolContext(
            call=call,
            seen_calls=frozenset(seen_calls),
            user_message=user_message,
        ),
        disabled=_disabled_snapshot(),
    )
    if isinstance(pre_outcome, Skip):
        return (False, False)
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

    seen_calls.add(_call_key(call))
    working.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
    working.append(ChatMessage(role="tool", content=result.output, name=call.name))
    if result.success:
        succeeded_tools.add(call.name)
    return (True, result.success)


_FORCED_SEARCH_MEMORY = "search_memory"
_FORCED_ASSEMBLE_CONTEXT = "assemble_context"


def _forced_assemble_context_prelude(
    working: list[ChatMessage],
    registry: ToolRegistry,
    emit: Callable[[ToolLoopEvent], None],
    seen_calls: set[tuple[str, str]],
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
    seen_calls.add(_call_key(call))
    working.append(ChatMessage(role="assistant", content="", tool_calls=(call,)))
    working.append(ChatMessage(role="tool", content=result.output, name=call.name))
    succeeded_tools.add(_FORCED_ASSEMBLE_CONTEXT)
    return True


def _forced_search_memory_prelude(
    working: list[ChatMessage],
    registry: ToolRegistry,
    emit: Callable[[ToolLoopEvent], None],
    seen_calls: set[tuple[str, str]],
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
    seen_calls.add(_call_key(call))
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
    seen_calls: set[tuple[str, str]],
    *,
    confirm: ConfirmFn | None,
    emit: Callable[[ToolLoopEvent], None],
    round_idx: int,
    hooks: HookPipeline,
    user_message: str | None,
    succeeded_tools: set[str],
) -> bool:
    """Execute the round's tool calls: duplicate-call hook, write-tier
    confirm, dispatch, append tool-role messages. Returns True if any
    call returned success=True (used to gate fabrication catchers on
    later rounds). Mutates `succeeded_tools` with the names of tools
    whose calls succeeded — lets narrow gates (e.g. fabricated_search
    only disarming on web-fetch tools) inspect which specific tools
    ran this turn."""
    from harness.orchestrator.hooks import _call_key

    any_success = False
    for call in calls:
        key = _call_key(call)
        pre_outcome = hooks.run_pre_tool(
            PreToolContext(
                call=call,
                seen_calls=frozenset(seen_calls),
                user_message=user_message,
            ),
            disabled=_disabled_snapshot(),
        )
        if isinstance(pre_outcome, Skip):
            # Duplicate of an earlier call this turn — skip execution.
            # Feed the nudge back as the tool-role message so the next
            # round sees 'finalize, don't re-call'.
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

        seen_calls.add(key)
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
    max_tokens: int = 1024,
    wrap_up_max_tokens: int = 1024,
    temperature: float = 0.5,
    router: Router | None = None,
    hooks: HookPipeline | None = None,
    memory_block_attached: bool = False,
    force_search_memory: bool = False,
    force_assemble_context: str | None = None,
    banter_tracker: BanterStreakTracker | None = None,
    scope_redirect_template: str | None = None,
    scope_lexicon: tuple[str, ...] = (),
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
    initial_count = len(working)
    events: list[ToolLoopEvent] = []
    last_reply: ModelReply = ModelReply(content="", tool_calls=())
    bail = _BailController(max_tokens=max_tokens, wrap_up_max_tokens=wrap_up_max_tokens)
    # Duplicate-call guard. Small models sometimes wrap a real answer
    # around a redundant re-call ("here's the summary" + same list_dir
    # with same args as a prior round). Tracking (name, args-json) per
    # turn lets us skip execution on repeats (harness-pun).
    seen_calls: set[tuple[str, str]] = set()

    def emit(event: ToolLoopEvent) -> None:
        events.append(event)
        if observe is not None:
            observe(event)

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
        )
        any_tool_succeeded = any_tool_succeeded or router_success

    for round_idx in range(max_rounds):
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
                ),
                disabled=_disabled_snapshot(),
            )
            can_retry = bail.retries_left > 0 and round_idx + 1 < max_rounds
            if not isinstance(bail_outcome, Continue) and can_retry:
                bail.consume_retry()
                if isinstance(bail_outcome, Truncated):
                    bail.on_truncated()
                    emit(ToolLoopEvent(kind="truncated_retry", round_index=round_idx))
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
                rounds=round_idx + 1,
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
        )
        any_tool_succeeded = any_tool_succeeded or round_success

    # Loop exhausted — return what we have.
    return ToolLoopResult(
        content=last_reply.content or "[tool loop exhausted without final reply]",
        messages=working,
        rounds=max_rounds,
        events=events,
    )
