from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from harness.model.adapter import ChatMessage
from harness.tools.base import (
    ModelReply,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolRegistry,
    ToolResult,
)

_DUPLICATE_CALL_NUDGE = (
    "[duplicate call — identical arguments to an earlier call this turn. "
    "Result is unchanged from the earlier tool message. Give the user your "
    "final answer now; do NOT emit any more tool calls.]"
)

# Final fallback when bail-retries are exhausted but the reply still trips
# a fabrication catcher. Before this, we surfaced the last fabricated reply
# verbatim (harness-24xj: "what is the date?" produced three stacked
# fabricated plans because the final retry still fabricated and we returned
# it). Swap to a canned refusal so the user never sees hallucinated tool
# output as an answer.
_EXHAUSTED_FABRICATION_FALLBACK = (
    "I couldn't answer that without calling a tool, and my attempts to "
    "call one didn't land cleanly. Try rephrasing, or ask me something I "
    "can answer without live data."
)


def _call_key(call: ToolCall) -> tuple[str, str]:
    """Canonical (name, arguments-json) key for duplicate detection.
    Sorting keys means argument order doesn't create false-positive
    uniqueness ({'a':1,'b':2} == {'b':2,'a':1}); default=str keeps the
    key stable if a model emits exotic-but-JSON-stringifiable types
    (dates, Paths). We never decode the key back — only equality
    matters — so lossy coercion is fine."""
    return (call.name, json.dumps(call.arguments, sort_keys=True, default=str))


if TYPE_CHECKING:
    from collections.abc import Iterator

    from harness.router.intent import Router
    from harness.tools.base import StreamChunk, ToolSpec


# Matches "Let me check…", "I'll now read…", "Next, I'll…" etc. — the
# model announcing more work without actually emitting tool calls. Anchored
# to end of content so a teaser mid-paragraph (followed by real prose) doesn't
# trigger.
_TEASER_RE = re.compile(
    r"\b(let me|i'?ll|now i'?ll|now let me|next,?\s+i'?ll?)\b[^.\n]*[:.]\s*$",
    re.IGNORECASE,
)

# Matches past-tense / present-perfect claims that an action was completed —
# "has been added", "is now included", "I've created", "successfully updated",
# etc. When the reply contains one of these AND no tool was executed in the
# turn, the model is hallucinating success (harness-3fn).
_ACTION_VERBS = (
    r"(?:added|included|updated|created|written|modified|"
    r"replaced|removed|set|appended|saved|deleted)"
)
_FALSE_SUCCESS_RE = re.compile(
    r"\b(?:"
    rf"has been\s+{_ACTION_VERBS}"
    r"|"
    r"(?:is|are)\s+now\s+(?:in|included|added|excluded|"
    r"set|present|updated|available|saved)"
    r"|"
    rf"(?:i(?:'ve|\shave))(?:\s+(?:just|now|successfully))?\s+{_ACTION_VERBS}"
    r"|"
    rf"successfully\s+{_ACTION_VERBS}"
    r"|"
    r"the\s+\S+\s+(?:has\s+been|is\s+now|will\s+be)\s+"
    r"(?:added|included|updated|created|excluded|modified|replaced)"
    r")\b",
    re.IGNORECASE,
)

# Matches chat-level meta-confirm prompts — "Would you like me to…?",
# "Should I…?", "Please confirm…", "Shall I…?" etc. Small models default
# to this pattern when they misunderstand that the user's request IS
# the instruction and the tool layer handles confirmation. Broad on
# purpose: 7B Qwen hit several variants in a single reply
# ("let's confirm", "would you like to", "please confirm your approval")
# so we catch all of them.
_META_CONFIRM_RE = re.compile(
    r"(?:"
    r"would you like (?:(?:me|us|you)\s+)?to\s+"
    r"(?:add|proceed|continue|update|create|edit|write|change|append|"
    r"remove|modify|delete|run|install|make|do|confirm|go\s+ahead)"
    r"|"
    r"shall i\b"
    r"|"
    r"should i (?:proceed|go ahead|continue|update|add|edit|change|"
    r"write|do|run)"
    r"|"
    r"do you want me to"
    r"|"
    r"please confirm"
    r"|"
    r"confirm (?:your |the |my )?(?:approval|request|intent|instruction)"
    r"|"
    r"let(?:'s|\s+us)\s+confirm"
    r"|"
    r"we\s+need\s+to\s+make\s+sure\s+(?:the\s+user|you)\s+confirms?"
    r"|"
    r"(?:please\s+)?approve\s+(?:the\s+|this\s+)?action"
    r")",
    re.IGNORECASE,
)

# Matches replies that look like fabricated search-tool output — a
# numbered results intro or URLs hosted on classic placeholder domains
# (example.com/org/net, your-site, localhost, etc.). Small models
# sometimes respond to 'search the web for X' by inventing a result
# list with a made-up URL and snippet rather than calling search_web.
# Paired with 'no tool has run this turn' this is a strong fabrication
# tell (see harness-q27, harness-j1d).
_FABRICATED_SEARCH_RE = re.compile(
    r"(?:"
    r"here\s+are\s+the\s+results"
    r"|"
    r"here\s+(?:is|are)\s+(?:what\s+)?i\s+found"
    r"|"
    r"(?:top|first|search)\s+results?\s*:"
    r"|"
    r"https?://(?:www\.)?(?:example|your-?site|your-?domain|"
    r"placeholder|localhost|test|dummy|fake)\.(?:com|org|net|io)\b"
    r"|"
    # Numbered-list entry whose content ends with terminal punctuation
    # plus a closing double-quote (`."`, `!"`, `?"`) and contains no
    # URL. Classic fabricated-snippet shape: 7B-class models imitate
    # search-tool output in an HTML-excerpt style instead of emitting
    # a <tool_call>. Tempered match rules out real `1. TITLE —
    # https://...` search_web entries. See harness-j1d.
    r"(?:\A|\n)\s*\d+\.\s+(?:(?!https?://).)*?[.!?]\"(?=\s|$)"
    r")",
    re.IGNORECASE,
)

# Matches ab_ops-specific fabrication shapes — ab imitates its own
# tool receipts (capture / plan) without emitting a <tool_call>.
# Distinct tells:
#   - `^Captured[.:]` — real CaptureTool output is `Captured <id> — [scope]…`
#     (id, not punctuation, follows `Captured`). A colon OR period after
#     `Captured` at line start is fabrication (harness-lbh colon form;
#     harness-ce2x period form, "Captured. Scope: personal. Outcome: …").
#   - `[prof/…]` / `[pers/…]` — real scopes are `professional` / `personal`,
#     never abbreviated. The abbreviation is a telltale of imitation.
#   - Two+ tier-label line headers (Shall/Should/Shmaybe/Watching) in one
#     reply — the `_render_plan` shape. A single mention is prose; two
#     together is plan imitation.
_FABRICATED_AB_CAPTURE_RE = re.compile(r"(?:\A|\n)\s*Captured\s*[.:]", re.IGNORECASE)
_FABRICATED_AB_SCOPE_RE = re.compile(r"\[(?:prof|pers)/[^\]]+\]", re.IGNORECASE)
# `Bead id: harness-xxx` in free text. Real CaptureTool output never
# uses this phrasing — the id appears bare as the second token after
# `Captured`. Any "Bead id:" / "Bead:" label with an id is the model
# imitating an imagined schema and is always a fabrication tell
# (harness-ce2x: user saw `Bead id: harness-abc123..`).
_FABRICATED_BEAD_ID_RE = re.compile(
    r"\bbead\s*(?:id)?\s*[:=]\s*(?:harness|bd|ab)-[a-z0-9]+",
    re.IGNORECASE,
)
# Bare past-tense success claim as a standalone sentence — "Updated.",
# "Captured.", "Created." — word flanked by whitespace/quote and
# followed by end-of-sentence. Gated on `tools_ran_this_turn=False`
# by the caller, so legitimate wrap-ups after a real tool ran never
# trip. Observed fabrication (harness-ce2x):
#   `Missed "…". Updated. Rerun /plan.` — no tool call, pure claim.
_BARE_CLAIM_RE = re.compile(
    r"(?:\A|[\s\"'])"
    r"(?:Updated|Captured|Created|Deleted|Removed|Added|Saved|Noted|Done)"
    r"\.(?:\s|$)",
    re.IGNORECASE,
)
# 'Today — YYYY-MM-DD' is the exact _render_plan header shape; natural
# prose basically never emits this prefix. Accept em-dash or hyphen
# (harness-jj9: the observed fabrication reproduces the em-dash verbatim).
_AB_DATE_HEADER_RE = re.compile(
    r"\bToday\s*[\u2014\-]\s*\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
# Tier labels anywhere in the reply (not just line-start): the user's
# observed fabrication emitted them inline after the date header, so the
# earlier line-start anchor missed the case entirely. Two+ labels with a
# colon/dash suffix is a plan-imitation signal regardless of layout
# (harness-jj9).
_AB_TIER_HEADER_RE = re.compile(
    r"\b(?:Shall|Should|Shmaybe|Watching)\b[-:]",
    re.IGNORECASE,
)


def _looks_like_ab_fabrication(content: str) -> bool:
    if _FABRICATED_AB_CAPTURE_RE.search(content):
        return True
    if _FABRICATED_BEAD_ID_RE.search(content):
        return True
    if _FABRICATED_AB_SCOPE_RE.search(content):
        return True
    if _AB_DATE_HEADER_RE.search(content):
        return True
    if len(_AB_TIER_HEADER_RE.findall(content)) >= 2:
        return True
    return bool(_BARE_CLAIM_RE.search(content))


# Matches tool-intent statements that should be accompanied by an
# actual <tool_call>. Broader than the trailing-teaser regex: doesn't
# require end-of-content anchoring, and includes 'I will <verb>' not
# just "I'll <verb>". When a reply contains one of these AND no tool
# was called this turn, the model has announced intent without acting
# on it — classic 7B failure mode (harness-q27 follow-up).
_TOOL_INTENT_RE = re.compile(
    r"\b(?:"
    r"i['\u2019]?ll|i\s+will|i\s+need\s+to|i\s+should|i'?m\s+going\s+to|"
    r"let\s+me|let['\u2019]?s|"
    r"now\s+i['\u2019]?ll|now\s+let\s+me|next,?\s+i['\u2019]?ll"
    r")\s+"
    r"(?:search|look\s+(?:up|for|at)|find|check|read|run|fetch|call|"
    r"invoke|execute|list|grep|edit|write|open|browse|query|"
    r"retrieve|download|inspect|examine)\b",
    re.IGNORECASE,
)

# Upper bound on the auto-widen loop triggered by _diagnose_bail=="truncated".
# 32k is deep into safe territory for the 131k-window Qwen 2.5 7B we ship —
# the real UX wall shows up well before: ~40-60 tok/s on an M4 Pro means an
# 8k reply already takes 2-3 minutes. See harness-cs9 for the architectural
# follow-up when wrap-ups consistently want > ~4k tokens (chunked output /
# model-splitting); this ceiling is just an anti-runaway guard, not a design
# target.
_MAX_TOKENS_CEILING = 32768
_BAIL_RETRIES_PER_TURN = 2


def _diagnose_bail(reply: ModelReply, *, tools_ran_this_turn: bool) -> str | None:
    """Classify a 0-tool-calls reply. Returns:
    - "truncated" — caller should retry with a larger token budget
    - a nudge string — caller should append it as a user message and retry
    - None — genuine final reply, terminate normally

    `tools_ran_this_turn` distinguishes a first-round bail (no tools
    have executed yet) from a wrap-up round (tools ran in a prior
    round; this round is summarizing). Completion claims are legitimate
    in wrap-ups but hallucinations in first-round bails."""
    if reply.was_truncated:
        return "truncated"
    if reply.had_unparseable_call:
        return (
            "Your last <tool_call> block was malformed and could not be parsed. "
            "Re-emit it as a single line of valid JSON inside <tool_call>…</tool_call>: "
            '<tool_call>{"name": "...", "arguments": {...}}</tool_call>'
        )
    if _TEASER_RE.search(reply.content.strip()):
        return (
            "Your reply announced more work but didn't include any tool calls. "
            "Either call the tool now, or give the user your final answer."
        )
    if not tools_ran_this_turn and _FALSE_SUCCESS_RE.search(reply.content):
        return (
            "Your reply claims that a file was changed / created / updated, "
            "but you did not call any tool this turn. You CANNOT modify the "
            "workspace without calling a write-tier tool (edit_file, "
            "write_file, shell). Either call the appropriate tool now, or "
            "tell the user you cannot make that change."
        )
    if not tools_ran_this_turn and _META_CONFIRM_RE.search(reply.content):
        return (
            "Do NOT ask the user to confirm in chat. The user's previous "
            "message IS the instruction — call the tool right now. Write-tier "
            "tools have their own approve/decline UX at the tool layer; "
            "re-asking in chat just wastes a round."
        )
    if not tools_ran_this_turn and _FABRICATED_SEARCH_RE.search(reply.content):
        return (
            "Your reply looks like fabricated tool output (search results / "
            "placeholder URLs / 'here are the results'). You did NOT call any "
            "tool this turn — you cannot know results without actually calling "
            "search_web / fetch_url / read_file. Call the appropriate tool now, "
            "or tell the user you cannot answer without live data."
        )
    if not tools_ran_this_turn and _looks_like_ab_fabrication(reply.content):
        return (
            "Your reply looks like fabricated tool output (ab_ops capture "
            "receipt / tiered plan / fake scope abbreviation). You did NOT "
            "call any tool this turn — you cannot produce a capture receipt "
            "or plan without actually calling `capture` / `plan`. Call the "
            "appropriate tool now, or tell the user plainly that you cannot."
        )
    if not tools_ran_this_turn and _TOOL_INTENT_RE.search(reply.content):
        return (
            "Your reply said you would do something ('I will search…', "
            "'let me check…', etc.) but you did NOT emit a tool_call. "
            "Stated intent is not action. To call a tool, emit EXACTLY "
            "this block (no wrapping text, no commentary) as part of "
            "your next reply:\n"
            '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>\n'
            "Example for search_web:\n"
            '<tool_call>{"name": "search_web", "arguments": '
            '{"query": "ahwatukee bbq"}}</tool_call>\n'
            "If you cannot figure out the right tool/args, say so plainly."
        )
    return None


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
    tool_call_deduped, truncated_retry, bail_retry, round_complete.

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


@dataclass
class ToolLoopResult:
    content: str
    messages: list[ChatMessage]  # full thread including tool turns
    rounds: int
    events: list[ToolLoopEvent] = field(default_factory=list)


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
    router: Router,
    working: list[ChatMessage],
    registry: ToolRegistry,
    confirm: ConfirmFn | None,
    emit: Callable[[ToolLoopEvent], None],
    seen_calls: set[tuple[str, str]],
) -> tuple[bool, bool]:
    """Classify the last user turn and, on a usable intent, append a
    synthetic assistant tool-call turn + the tool result to `working`
    in place. Returns (routed, succeeded) — `routed` is True if routing
    produced a tool execution (main model enters wrap-up mode directly),
    `succeeded` is True iff that tool returned ToolResult.success=True.
    The main loop needs both signals: routed-but-failed still counts as
    "tool executed" for the wrap-up token cap but NOT for disarming
    fabrication catchers (see harness-a0y).

    Conservative guards: read-tier tools only (write-tier needs the
    main model's richer context + its own confirmation UX), intent
    tool must exist in the registry, all required arguments must be
    present. Any failure falls through silently — the router is
    advisory, never blocking.

    `seen_calls` is mutated: on success, the router's (name, args-json)
    key is recorded so a downstream main-model re-invocation with
    identical arguments gets short-circuited by the main loop's
    duplicate guard."""
    user_message = _last_user_message(working)
    if user_message is None:
        return (False, False)
    intent = router.classify(user_message, registry.specs())
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
    return (True, result.success)


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
    into on factual queries (see harness-j1d / harness-q27)."""
    working: list[ChatMessage] = list(messages)
    initial_count = len(working)
    events: list[ToolLoopEvent] = []
    last_reply: ModelReply = ModelReply(content="", tool_calls=())
    bail_retries = _BAIL_RETRIES_PER_TURN
    current_max_tokens = max_tokens
    # Wrap-up cap lives alongside the main cap so the truncated-recovery
    # branch can widen it too. Earlier bug (harness-jly): only
    # current_max_tokens was doubled on retry — round_max_tokens was
    # still min(current_max_tokens, wrap_up_max_tokens) in wrap-up
    # rounds, so every retry re-truncated at the original cap and the
    # user saw the same partial summary streamed 3x.
    current_wrap_up_max_tokens = wrap_up_max_tokens
    # Duplicate-call guard. Small models sometimes wrap a real answer
    # around a redundant re-call ("here's the summary" + same list_dir
    # with same args as a prior round). Each round the call runs,
    # returns identical output, and the model rewrites the summary —
    # wasting rounds and making the TUI look like it's stuck. Tracking
    # (name, args-json) per turn lets us skip execution on repeats and
    # feed a 'stop, finalize' nudge to the model instead. See
    # harness-pun.
    seen_calls: set[tuple[str, str]] = set()

    def emit(event: ToolLoopEvent) -> None:
        events.append(event)
        if observe is not None:
            observe(event)

    # Tracks whether any tool call THIS TURN returned success=True. The
    # fabrication catchers (_diagnose_bail) gate on this: if every tool
    # this turn errored, the model has no real data to wrap up, so a
    # completion-style reply is still hallucination (harness-a0y).
    any_tool_succeeded = False

    if router is not None:
        _, router_success = _router_prelude(router, working, registry, confirm, emit, seen_calls)
        any_tool_succeeded = any_tool_succeeded or router_success

    stream_fn = getattr(adapter, "stream_with_tools", None)

    for round_idx in range(max_rounds):
        # Post-tool rounds are wrap-up rounds — tighter cap.
        tools_already_ran = any(m.role == "tool" for m in working[initial_count:])
        round_max_tokens = (
            min(current_max_tokens, current_wrap_up_max_tokens)
            if tools_already_ran
            else current_max_tokens
        )
        emit(ToolLoopEvent(kind="round_start", round_index=round_idx))
        emit(ToolLoopEvent(kind="model_call_start", round_index=round_idx))
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

        # Small models sometimes emit meta-confirm narrative AND a tool call
        # in the same reply ("Would you like me to …? <tool_call>…"). The
        # tool call is valid but the narrative is noise — strip it from the
        # assistant turn's content so the wrap-up round doesn't see the model
        # hallucinating a confirmation dialog in its own history. The tool
        # still runs; the user just doesn't get a bizarre 'did you want me
        # to?' before an action they already asked for.
        if last_reply.tool_calls and _META_CONFIRM_RE.search(last_reply.content):
            last_reply = ModelReply(
                content="",
                tool_calls=last_reply.tool_calls,
                was_truncated=last_reply.was_truncated,
                had_unparseable_call=last_reply.had_unparseable_call,
            )

        if not last_reply.tool_calls:
            # If any tool has already executed in this turn (prior round's
            # tool_calls produced tool-role messages appended to working),
            # a completion claim is legitimate — the model is wrapping up.
            # Only treat a claim as hallucination when no tool has run yet.
            # Gate on successful tool execution, not mere execution: an
            # all-errored turn (e.g. plan with invalid scope) must still
            # trip fabrication catchers because the model has no real
            # data to wrap up (harness-a0y).
            diag = _diagnose_bail(last_reply, tools_ran_this_turn=any_tool_succeeded)
            can_retry = bail_retries > 0 and round_idx + 1 < max_rounds
            if diag is not None and can_retry:
                bail_retries -= 1
                if diag == "truncated":
                    current_max_tokens = min(current_max_tokens * 2, _MAX_TOKENS_CEILING)
                    # Wrap-up rounds need the widened budget too, else
                    # the round_max_tokens clamp still truncates at the
                    # old wrap_up cap and the retry re-truncates at the
                    # same spot (harness-jly).
                    current_wrap_up_max_tokens = min(
                        current_wrap_up_max_tokens * 2, _MAX_TOKENS_CEILING
                    )
                    emit(ToolLoopEvent(kind="truncated_retry", round_index=round_idx))
                else:
                    # Must fire BEFORE the nudge is queued so the CLI /
                    # TUI can drop the in-flight stream buffer — each
                    # retry re-streams from scratch and we don't want
                    # the fabricated draft to stay on screen
                    # (harness-24xj).
                    emit(ToolLoopEvent(kind="bail_retry", round_index=round_idx))
                    working.append(ChatMessage(role="user", content=diag))
                continue
            # Retries / rounds exhausted. If the reply is still
            # fabrication-shaped, swap in the canned fallback rather
            # than surfacing the hallucination as the final answer
            # (harness-24xj). Truncation isn't fabrication — we'd
            # rather show the partial than a refusal.
            if diag is not None and diag != "truncated":
                last_reply = ModelReply(
                    content=_EXHAUSTED_FABRICATION_FALLBACK,
                    tool_calls=(),
                    was_truncated=last_reply.was_truncated,
                    had_unparseable_call=last_reply.had_unparseable_call,
                )
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

        for call in last_reply.tool_calls:
            key = _call_key(call)
            if key in seen_calls:
                # Duplicate of an earlier call this turn — skip execution.
                # Emit the deduped event for CLI visibility and feed the
                # nudge back as the tool-role message so the next round
                # sees 'finalize, don't re-call'.
                result = ToolResult(
                    tool_name=call.name,
                    output=_DUPLICATE_CALL_NUDGE,
                    success=True,
                )
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

            seen_calls.add(key)
            if result.success:
                any_tool_succeeded = True
            working.append(ChatMessage(role="tool", content=result.output, name=call.name))

    # Loop exhausted — return what we have.
    return ToolLoopResult(
        content=last_reply.content or "[tool loop exhausted without final reply]",
        messages=working,
        rounds=max_rounds,
        events=events,
    )
