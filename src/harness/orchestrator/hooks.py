"""Hook pipeline for orchestrator catchers.

The tool loop has ~11 named catchers — regex guards + flag checks that
intercept fabrication-shaped model replies. They used to live as inline
`if` branches in `tool_loop.py`; this module hosts them as typed hooks
organized into four phases:

- `post_model` — runs on every model reply, before tool dispatch.
  Today: `paired_meta_confirm_strip` (strip narrative from a reply that
  also carries a valid tool call).
- `bail` — runs on 0-tool-calls replies. First matching hook wins.
  Today: `truncated` · `unparseable` · `teaser` · `false_success` ·
  `meta_confirm` · `fabricated_search` · `ab_fabrication` · `tool_intent`.
- `pre_tool` — runs before each individual tool call.
  Today: `duplicate_call` (short-circuit identical-args re-invocation).
- `finalize` — runs after bail retries are exhausted.
  Today: `fabrication_fallback` (replace lingering fabricated reply with
  canned refusal).

Behavior preservation is the non-negotiable success criterion: every
catcher below is a direct port of its inline predecessor, and the
attribution eval (`harness.evals.tool_loop`) keeps running via the
`disabled: frozenset[str]` argument threaded through each pipeline
method — toggle a name, watch which scenarios it uniquely saves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from harness.model.adapter import ChatMessage
from harness.tools.base import ModelReply, ToolCall, ToolResult, ToolSpec

# ---------- canned strings ----------


DUPLICATE_CALL_NUDGE = (
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
EXHAUSTED_FABRICATION_FALLBACK = (
    "I couldn't answer that without calling a tool, and my attempts to "
    "call one didn't land cleanly. Try rephrasing, or ask me something I "
    "can answer without live data."
)


# ---------- regex library (shared with cli stream filter) ----------


# Matches "Let me check…", "I'll now read…", "Next, I'll…" etc. — the
# model announcing more work without actually emitting tool calls. Anchored
# to end of content so a teaser mid-paragraph (followed by real prose) doesn't
# trigger.
TEASER_RE = re.compile(
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
FALSE_SUCCESS_RE = re.compile(
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
META_CONFIRM_RE = re.compile(
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
FABRICATED_SEARCH_RE = re.compile(
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

# Past-tense claim of having searched the web / online / the internet
# when no web-fetch tool actually ran. Harness-78z shape: search_memory
# returned empty, model narrates "I've searched the web for X and
# found..." without ever calling search_web. Split from
# FABRICATED_SEARCH_RE because its gate is narrower: any tool running
# disarms the result-list patterns (legit wrap-up), but NO amount of
# non-web tool activity legitimizes a "I've searched the web" claim —
# only search_web / fetch_url actually running does.
_APOS_CLASS = "['’]"  # noqa: RUF001 — straight + curly apostrophe in a char class
FABRICATED_WEB_CLAIM_RE = re.compile(
    rf"(?:"
    rf"\bi(?:{_APOS_CLASS}ve|\s+have|\s+just)?\s+searched\s+"
    rf"(?:the\s+(?:web|internet)|online)"
    rf"|"
    rf"\b(?:after|upon)\s+searching\s+(?:the\s+(?:web|internet)|online)"
    rf")",
    re.IGNORECASE,
)

# Matches replies that regenerate a "here are the articles/stories/
# headlines…" summary list without calling any tool this turn — the
# classic follow-up fabrication shape (harness-f5x). User asks "more
# details on 8" from an earlier fetch result; 3B model skips the
# index→URL lookup and fakes a new summary. Distinct from
# FABRICATED_SEARCH_RE because the intro phrasing names articles /
# stories / headlines rather than "results" / "what I found".
# Alternatives listed first for regex efficiency (Python re backs off).
FABRICATED_ITEMIZATION_RE = re.compile(
    r"(?:"
    r"here\s+are\s+(?:some|the|a\s+few)\s+"
    r"(?:of\s+the\s+)?(?:first|top|latest|popular|recent|main)?\s*"
    r"(?:\d+\s+)?"
    r"(?:articles|stories|headlines|items|posts|entries|news|"
    r"top\s+stories)"
    r"|"
    r"let'?s\s+focus\s+on\s+(?:one|a\s+few)\s+of\s+the"
    r"|"
    r"here'?s\s+(?:a\s+)?(?:summary|rundown|overview|recap)\s+of\s+"
    r"(?:the\s+)?(?:articles|stories|headlines|items|posts)"
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
FABRICATED_AB_CAPTURE_RE = re.compile(r"(?:\A|\n)\s*Captured\s*[.:]", re.IGNORECASE)
# `Remembered:` receipt from ab_ops RememberTool. Real shape is
# `Remembered: <insight>` (colon form), which is exactly what the model
# imitates (harness-z734: user asked ab to remember a birthday; round 0
# bail-retry fired on 'Updated.' bare-claim, round 1 fabricated
# `Remembered: dad Steve's birthday is Oct 8th` and — because no
# 'Remembered'-shaped catcher existed — slipped through as the final
# answer). Safe because bail runs gated on `tools_ran_this_turn=False`;
# real RememberTool output disarms this.
FABRICATED_REMEMBER_RE = re.compile(r"(?:\A|\n)\s*Remembered\s*[.:]", re.IGNORECASE)
FABRICATED_AB_SCOPE_RE = re.compile(r"\[(?:prof|pers)/[^\]]+\]", re.IGNORECASE)
# `Bead id: harness-xxx` in free text. Real CaptureTool output never
# uses this phrasing — the id appears bare as the second token after
# `Captured`. Any "Bead id:" / "Bead:" label with an id is the model
# imitating an imagined schema and is always a fabrication tell
# (harness-ce2x: user saw `Bead id: harness-abc123..`).
FABRICATED_BEAD_ID_RE = re.compile(
    r"\bbead\s*(?:id)?\s*[:=]\s*(?:harness|bd|ab)-[a-z0-9]+",
    re.IGNORECASE,
)
# Domain-like token inside a tool argument's string value: matches
# `example.com`, `dailydrop.fm`, `sub.example.co.uk`, etc. Captures the
# leftmost label ("example", "dailydrop") — that's the "root" the
# grounding hook matches against the user message. TLD whitelist is
# intentionally narrow: catches the public-web TLDs that show up in
# real fetch_url / search_web args without false-positiving on
# filenames like `new.txt`, `config.yaml`, `data.json` (file extensions
# are NOT in the list). A missing TLD is cheap — legitimate calls pass;
# only "leaked fabricated domain" cases get caught. Extend as needed
# when a legitimate domain appears that isn't covered.
_KNOWN_TLDS = (
    "com|org|net|io|ai|dev|me|co|edu|gov|fm|tv|app|info|biz|xyz|"
    "uk|us|de|fr|jp|ca|au|nz|nl|es|it|ru|in|cn|br|mx|ly|to|cc|so|"
    "gg|pro|tech|site|online|store|cloud|news|blog|wiki"
)
ARG_DOMAIN_RE = re.compile(
    rf"\b([a-z0-9][a-z0-9-]*)\.(?:[a-z]{{2,}}\.)*(?:{_KNOWN_TLDS})\b",
    re.IGNORECASE,
)


def _arg_string_values(arguments: dict[str, object]) -> list[str]:
    """Flatten a tool-call's argument dict into its string values (one
    level deep). Non-string values are stringified via `str()` so a
    `max_results=1` doesn't count as a URL but also doesn't silently
    drop any string-like content. Nested dicts/lists get their leaf
    strings pulled too — covers `arguments={"filters": ["python"]}`
    shapes without recursion blow-ups on cycles (tool args are
    JSON-serializable, so no cycles by construction)."""

    out: list[str] = []

    def _walk(value: object) -> None:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list | tuple):
            for v in value:
                _walk(v)

    _walk(arguments)
    return out


def _arg_domain_roots(arg_values: list[str]) -> set[str]:
    """Collect the lowercased leftmost labels of every domain-like
    token embedded in `arg_values`. Duplicates are collapsed; order
    doesn't matter."""
    roots: set[str] = set()
    for value in arg_values:
        for match in ARG_DOMAIN_RE.finditer(value):
            roots.add(match.group(1).lower())
    return roots


# Bare past-tense success claim as a standalone sentence — "Updated.",
# "Captured.", "Created." — word flanked by whitespace/quote and
# followed by end-of-sentence. Gated on `tools_ran_this_turn=False`
# by the caller, so legitimate wrap-ups after a real tool ran never
# trip. Observed fabrication (harness-ce2x):
#   `Missed "…". Updated. Rerun /plan.` — no tool call, pure claim.
BARE_CLAIM_RE = re.compile(
    r"(?:\A|[\s\"'])"
    r"(?:Updated|Captured|Created|Deleted|Removed|Added|Saved|Noted|Done|Remembered)"
    r"\.(?:\s|$)",
    re.IGNORECASE,
)
# 'Today — YYYY-MM-DD' is the exact _render_plan header shape; natural
# prose basically never emits this prefix. Accept em-dash or hyphen
# (harness-jj9: the observed fabrication reproduces the em-dash verbatim).
AB_DATE_HEADER_RE = re.compile(
    r"\bToday\s*[—\-]\s*\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
# Tier labels anywhere in the reply (not just line-start): the user's
# observed fabrication emitted them inline after the date header, so the
# earlier line-start anchor missed the case entirely. Two+ labels with a
# colon/dash suffix is a plan-imitation signal regardless of layout
# (harness-jj9).
AB_TIER_HEADER_RE = re.compile(
    r"\b(?:Shall|Should|Shmaybe|Watching)\b[-:]",
    re.IGNORECASE,
)


def looks_like_ab_fabrication(content: str) -> bool:
    if FABRICATED_AB_CAPTURE_RE.search(content):
        return True
    if FABRICATED_REMEMBER_RE.search(content):
        return True
    if FABRICATED_BEAD_ID_RE.search(content):
        return True
    if FABRICATED_AB_SCOPE_RE.search(content):
        return True
    if AB_DATE_HEADER_RE.search(content):
        return True
    if len(AB_TIER_HEADER_RE.findall(content)) >= 2:
        return True
    return bool(BARE_CLAIM_RE.search(content))


# Matches tool-intent statements that should be accompanied by an
# actual <tool_call>. Broader than the trailing-teaser regex: doesn't
# require end-of-content anchoring, and includes 'I will <verb>' not
# just "I'll <verb>". When a reply contains one of these AND no tool
# was called this turn, the model has announced intent without acting
# on it — classic 7B failure mode (harness-q27 follow-up).
TOOL_INTENT_RE = re.compile(
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


# ---------- outcomes ----------


@dataclass(frozen=True)
class Continue:
    """Default outcome — this hook had nothing to say."""


@dataclass(frozen=True)
class Nudge:
    """0-tool-call reply tripped a catcher. Caller appends `text` as a
    user-role message and re-runs the round.

    `catcher` names the hook that produced this Nudge. Populated by
    `HookPipeline.run_bail` at return time (individual hooks construct
    Nudge with text only — the pipeline attaches the name). Surfaced
    in `ToolLoopEvent(kind="bail_retry")` so the CLI / TUI can show
    which rule triggered the retry ('⋯ discarding draft, retrying
    (ambiguous_context)…')."""

    text: str
    catcher: str = ""


@dataclass(frozen=True)
class Truncated:
    """Reply stopped mid-stream at the token cap. Caller widens caps
    and re-runs without appending anything."""


@dataclass(frozen=True)
class Replace:
    """Mutate the round's reply in place (used by
    `paired_meta_confirm_strip` to drop narrative when a tool call is
    already present)."""

    reply: ModelReply


@dataclass(frozen=True)
class Halt:
    """Return this reply as the final turn output; no more retries."""

    reply: ModelReply


@dataclass(frozen=True)
class Skip:
    """Skip executing this tool call; feed `result` back as the tool-
    role message instead."""

    result: ToolResult


@dataclass(frozen=True)
class ReplaceResult:
    """Replace a just-executed tool's result before it's appended to
    the model-visible message thread. Used by the tool-result
    summarizer (sota punch #3, harness-zoz) to compress high-noise
    outputs before they eat through context. The tool DID run; only
    the string the model sees is rewritten."""

    result: ToolResult


BailOutcome = Continue | Nudge | Truncated
PostModelOutcome = Continue | Replace
PreToolOutcome = Continue | Skip
PostToolOutcome = Continue | ReplaceResult
FinalizeOutcome = Continue | Halt


# ---------- hook contexts + protocols ----------


@dataclass(frozen=True)
class BailContext:
    """Inputs to a bail hook. `tools_ran_this_turn` distinguishes
    first-round bails (no tools executed yet → completion claims are
    hallucinations) from wrap-up rounds (tools ran earlier → claims
    are legitimate). `tools_ran` carries the finer signal — the set
    of tool names that succeeded this turn — so hooks can gate on
    specific tools (e.g. fabricated_search only cares whether a
    web-fetch tool ran; search_memory running doesn't legitimize
    a fabricated web-search narration).

    `user_message` is the verbatim content of the most recent
    user-role turn (same value that feeds `PreToolContext.user_message`).
    Hooks use it to tell 'model echoed the user's input' apart from
    'model is making a new claim' — e.g. reserved_squawk_code
    disarms when the reserved-code match in the reply is a verbatim
    echo of the user's quoted question. Defaults to None so callers
    that don't plumb it through fall through untouched."""

    reply: ModelReply
    tools_ran_this_turn: bool
    tools_ran: frozenset[str] = frozenset()
    user_message: str | None = None


@dataclass(frozen=True)
class PostModelContext:
    reply: ModelReply


@dataclass(frozen=True)
class PreToolContext:
    """Inputs to a pre-tool hook. `user_message` is the verbatim content
    of the most recent user-role turn in the loop's working thread — the
    grounding hook uses it to verify that entity-specific arguments
    (URLs, domains) trace back to something the user actually named. Nil
    when no user turn exists yet (system-only bootstrap)."""

    call: ToolCall
    seen_calls: frozenset[tuple[str, str]]
    user_message: str | None = None


@dataclass(frozen=True)
class PostToolContext:
    """Inputs to a post-tool hook. `spec` is the ToolSpec of the
    tool that just executed — the summarizer hook uses `spec.high_noise`
    to decide whether to compress, so we pass it in rather than have
    the hook hold a registry reference."""

    call: ToolCall
    result: ToolResult
    spec: ToolSpec


@dataclass(frozen=True)
class FinalizeContext:
    """Inputs to a finalize hook. `last_outcome` is the last non-Continue
    bail outcome the pipeline produced this turn, so `fabrication_fallback`
    can fire only when a fabrication-shaped bail outcome survived
    retries (truncated outcomes are exempt).

    `tools_ran` carries the set of tool names that succeeded this turn —
    used by `ungrounded_citation` to check whether any grounding tool
    (search_memory / fact_search) ran before a section-citation reply.
    `memory_block_attached` reports whether the CLI (or equivalent
    call-site) attached a retrieval memory block to the system prompt
    for this turn — the other grounding signal the citation catcher
    needs.

    `tool_outputs` carries the concatenated text of every successful
    tool-role message this turn, in execution order — the grounding
    corpus against which `table_fabrication` verifies that every
    pipe-table data row in the reply appears verbatim (harness-5uq).
    Defaults to the empty tuple so callers that don't plumb tool
    outputs through fall through untouched."""

    reply: ModelReply
    last_outcome: BailOutcome
    tools_ran: frozenset[str] = frozenset()
    memory_block_attached: bool = False
    tool_outputs: tuple[str, ...] = ()
    # Max retrieval score across every tool call this turn, or None
    # when no tool surfaced retrieval metadata. Populated by the
    # tool loop from ToolResult.hits (harness-ywp.4). Consumed by
    # LowConfidenceFallbackHook (harness-ywp.3) to refuse citations
    # that ride on weak retrieval.
    retrieval_top_score: float | None = None
    # Union of citations_grounded across every tool call this turn —
    # canonicalised (harness-ywp.5). Consumed by the low-confidence
    # fallback to distinguish citations the tool actually grounded
    # from ones the model made up.
    citations_grounded: frozenset[str] = frozenset()


class BailHook(Protocol):
    # Declared as a read-only property so frozen-dataclass implementations
    # (every concrete hook below) structurally match. A bare `name: str`
    # in the Protocol would require a writable attribute.
    @property
    def name(self) -> str: ...

    def check(self, ctx: BailContext) -> BailOutcome: ...


class PostModelHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: PostModelContext) -> PostModelOutcome: ...


class PreToolHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: PreToolContext) -> PreToolOutcome: ...


class PostToolHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: PostToolContext) -> PostToolOutcome: ...


class FinalizeHook(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome: ...


# ---------- bail hooks ----------


@dataclass(frozen=True)
class TruncatedHook:
    name: str = "truncated"

    def check(self, ctx: BailContext) -> BailOutcome:
        return Truncated() if ctx.reply.was_truncated else Continue()


_UNPARSEABLE_NUDGE = (
    "Your last <tool_call> block was malformed and could not be parsed. "
    "Re-emit it as a single line of valid JSON inside <tool_call>…</tool_call>: "
    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>'
)


@dataclass(frozen=True)
class UnparseableHook:
    name: str = "unparseable"

    def check(self, ctx: BailContext) -> BailOutcome:
        return Nudge(_UNPARSEABLE_NUDGE) if ctx.reply.had_unparseable_call else Continue()


_TEASER_NUDGE = (
    "Your reply announced more work but didn't include any tool calls. "
    "Either call the tool now, or give the user your final answer."
)


@dataclass(frozen=True)
class TeaserHook:
    name: str = "teaser"

    def check(self, ctx: BailContext) -> BailOutcome:
        if TEASER_RE.search(ctx.reply.content.strip()):
            return Nudge(_TEASER_NUDGE)
        return Continue()


_FALSE_SUCCESS_NUDGE = (
    "Your reply claims that a file was changed / created / updated, "
    "but you did not call any tool this turn. You CANNOT modify the "
    "workspace without calling a write-tier tool (edit_file, "
    "write_file, shell). Either call the appropriate tool now, or "
    "tell the user you cannot make that change."
)


@dataclass(frozen=True)
class FalseSuccessHook:
    name: str = "false_success"

    def check(self, ctx: BailContext) -> BailOutcome:
        if ctx.tools_ran_this_turn:
            return Continue()
        if FALSE_SUCCESS_RE.search(ctx.reply.content):
            return Nudge(_FALSE_SUCCESS_NUDGE)
        return Continue()


_META_CONFIRM_NUDGE = (
    "Do NOT ask the user to confirm in chat. The user's previous "
    "message IS the instruction — call the tool right now. Write-tier "
    "tools have their own approve/decline UX at the tool layer; "
    "re-asking in chat just wastes a round."
)


@dataclass(frozen=True)
class MetaConfirmHook:
    name: str = "meta_confirm"

    def check(self, ctx: BailContext) -> BailOutcome:
        if ctx.tools_ran_this_turn:
            return Continue()
        if META_CONFIRM_RE.search(ctx.reply.content):
            return Nudge(_META_CONFIRM_NUDGE)
        return Continue()


_FABRICATED_SEARCH_NUDGE = (
    "Your reply looks like fabricated tool output (search results / "
    "placeholder URLs / 'here are the results'). You did NOT call any "
    "tool this turn — you cannot know results without actually calling "
    "search_web / fetch_url / read_file. Call the appropriate tool now, "
    "or tell the user you cannot answer without live data."
)


_WEB_FETCH_TOOLS: frozenset[str] = frozenset({"search_web", "fetch_url"})


@dataclass(frozen=True)
class FabricatedSearchHook:
    """Catches two shapes of web-search fabrication.

    1. Result-list patterns (FABRICATED_SEARCH_RE): "here are the
       results", placeholder-domain URLs, quoted-snippet numbered
       lists. Gated on `tools_ran_this_turn` — ANY tool running
       disarms the catcher because the wrap-up may legitimately
       summarize that tool's output.
    2. Past-tense web-claim (FABRICATED_WEB_CLAIM_RE): "I've
       searched the web", "after searching online". Gated more
       narrowly on `tools_ran` — only a real search_web or
       fetch_url disarms it. Rationale (harness-78z): the model
       claims to have done a web search; only an actual web call
       legitimizes that claim. A search_memory miss followed by
       "I've searched the web" narration is still fabrication."""

    name: str = "fabricated_search"

    def check(self, ctx: BailContext) -> BailOutcome:
        web_ran = bool(ctx.tools_ran & _WEB_FETCH_TOOLS)
        if not web_ran and FABRICATED_WEB_CLAIM_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_SEARCH_NUDGE)
        if ctx.tools_ran_this_turn:
            return Continue()
        if FABRICATED_SEARCH_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_SEARCH_NUDGE)
        return Continue()


_FABRICATED_ITEMIZATION_NUDGE = (
    "Your reply looks like a regenerated summary list ('here are the "
    "articles/stories/headlines…') but you did NOT call any tool this "
    "turn. When the user asks for more detail on a specific item from "
    "an earlier tool result, the ONLY correct response is to call "
    "`fetch_url` on THAT item's URL from the prior tool output. Do NOT "
    "paraphrase, re-list, or summarize from context. If you can't find "
    "the URL for the item the user named, tell them you need it pasted."
)


@dataclass(frozen=True)
class FabricatedItemizationHook:
    """Catches the follow-up-fabrication shape: user asks for more
    detail on item N from an earlier fetch, small model skips the
    index→URL lookup and regenerates a fake summary list. Gated on
    `tools_ran_this_turn=False` so legitimate wrap-up lists after a
    real tool ran never trip (harness-f5x)."""

    name: str = "fabricated_itemization"

    def check(self, ctx: BailContext) -> BailOutcome:
        if ctx.tools_ran_this_turn:
            return Continue()
        if FABRICATED_ITEMIZATION_RE.search(ctx.reply.content):
            return Nudge(_FABRICATED_ITEMIZATION_NUDGE)
        return Continue()


_AB_FABRICATION_NUDGE = (
    "Your reply looks like fabricated tool output (ab_ops capture "
    "receipt / tiered plan / fake scope abbreviation). You did NOT "
    "call any tool this turn — you cannot produce a capture receipt "
    "or plan without actually calling `capture` / `plan`. Call the "
    "appropriate tool now, or tell the user plainly that you cannot."
)


@dataclass(frozen=True)
class AbFabricationHook:
    name: str = "ab_fabrication"

    def check(self, ctx: BailContext) -> BailOutcome:
        if ctx.tools_ran_this_turn:
            return Continue()
        if looks_like_ab_fabrication(ctx.reply.content):
            return Nudge(_AB_FABRICATION_NUDGE)
        return Continue()


_TOOL_INTENT_NUDGE = (
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


@dataclass(frozen=True)
class ToolIntentHook:
    name: str = "tool_intent"

    def check(self, ctx: BailContext) -> BailOutcome:
        if ctx.tools_ran_this_turn:
            return Continue()
        if TOOL_INTENT_RE.search(ctx.reply.content):
            return Nudge(_TOOL_INTENT_NUDGE)
        return Continue()


# Reply-side regex for "this is in-scope for the cited document" signal.
# airton_c1 scope is JO 7110.65 only; matches the most common ways the
# model refers to it ('JO 7110.65', 'FAA Order JO 7110.65'). Narrow on
# purpose — we don't want this hook to fire on generic prose that
# happens to contain the word 'order'. Airton / airton_b / other
# non-citation-scoped characters simply never produce replies that
# match this pattern, so the hook is self-gating for them.
_ORDER_REFERENCE_RE = re.compile(
    r"\b(?:FAA\s+Order\s+)?JO\s*7110\.65\b",
    re.IGNORECASE,
)


# Secondary in-scope signal: JO-normative phraseology. The order
# formats controller phraseology blocks in ALL CAPS (e.g.
# 'RADAR SERVICE TERMINATED', 'SQUAWK VFR'), but airton_c1's replies
# routinely narrate these in mixed case ('the correct phraseology is
# "Radar service terminated, squawk VFR"'). The regex is
# case-insensitive so both forms trigger — the phrase tokens are
# specific enough to ATC that a casual narrative use outside an
# airton_c1 reply is vanishingly unlikely. Plus meta-phraseology
# triggers ('correct phraseology', 'proper phraseology',
# 'phraseology for terminating/clearing/...') so a reply that DOESN'T
# quote the exact JO line but still teaches normative phraseology
# trips the in-scope check.
_JO_PHRASEOLOGY_MARKERS_RE = re.compile(
    r"\b(?:"
    r"SQUAWK\s+(?:VFR|IDENT|STANDBY|STOP|MAYDAY|\d{4}|"
    r"(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|ZERO)"
    r"(?:\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|ZERO))*)"
    r"|RADAR\s+(?:SERVICE\s+TERMINATED|CONTACT(?:\s+LOST)?)"
    r"|CLEARED\s+(?:TO|FOR)\s+[A-Z]"
    r"|CLEARED\s+FOR\s+(?:TAKEOFF|LANDING|THE\s+APPROACH)"
    r"|CONTACT\s+(?:TOWER|GROUND|DEPARTURE|APPROACH|CENTER|CLEARANCE)"
    r"|FREQUENCY\s+CHANGE\s+APPROVED"
    r"|CHANGE\s+TO\s+ADVISORY"
    r"|MAINTAIN\s+(?:VFR|ALTITUDE|FL|FLIGHT\s+LEVEL)"
    r"|HOLD\s+SHORT(?:\s+OF)?"
    r"|(?:DESCEND|CLIMB)\s+(?:AND\s+MAINTAIN|TO)"
    r"|TRAFFIC(?:\s+ALERT)?"
    r"|PROCEED\s+DIRECT"
    r"|CROSS\s+[A-Z]+\s+AT"
    r"|TURN\s+(?:LEFT|RIGHT)\s+HEADING"
    # Meta-phraseology signals (narrative references to normative
    # phraseology, not direct quotes):
    r"|correct\s+phraseology"
    r"|proper\s+phraseology"
    r"|phraseology\s+(?:for|is|should\s+be|would\s+be)"
    r"|the\s+phrase(?:ology)?\s+\""
    r")\b",
    re.IGNORECASE,
)


# Citation shapes we accept as 'the reply did cite a section':
#   §N-N-N                (primary anchor — JO chapter-section-paragraph)
#   TBL N-N-N / Table N-N-N (tables are always section-scoped; accept
#                            as a standalone cite when no § is present)
#   FIG N-N-N / Figure N-N-N (same rationale as tables)
#   Chapter N §N-N-N      (constitution's preferred form — §-anchor is
#                            what matches here, Chapter N is prose)
# Tolerant of the hyphen / en-dash / unicode-minus variants in the
# corpus. Reuses UNGROUNDED_SECTION_CITATION_RE's shape for the §-form
# so the two hooks can't disagree about what counts as a citation.
_CITATION_PRESENT_RE = re.compile(
    # §N-N-N (primary — chapter-section-paragraph) or §N-N (broader
    # chapter-section reference, e.g. '§9-6' for the entire Unmanned
    # Free Balloons section). Both count as a citation for the
    # missing_citation check even though UngroundedCitationHook
    # stays strict at 3-segment to avoid matching prices / anchors.
    r"§\s*\d+[-–−]\d+(?:[-–−]\d+)?"  # noqa: RUF001 — dash variants load-bearing
    r"|\b(?:TBL|Table|FIG|Figure)\s+\d+[-–−]\d+[-–−]\d+\b",  # noqa: RUF001
    re.IGNORECASE,
)


# A reply that's asking the user to clarify between variants isn't
# making a substantive claim — it's a question back, not an answer.
# No citation expected. MissingCitationHook and (later) any other
# 'you must cite' gate should exempt replies matching this shape so
# a valid clarifying question doesn't get nudged into a pointless
# retry loop (session 2026-04-24 repro: ambiguous_context fires;
# model asks 'manned or unmanned?'; missing_citation then fires on
# the clarifying question because it has 'JO 7110.65' but no §).
_CLARIFYING_QUESTION_RE = re.compile(
    r"(?:"
    r"could\s+you\s+(?:please\s+)?(?:clarify|specify)"
    r"|can\s+you\s+(?:please\s+)?(?:clarify|specify)"
    r"|do\s+you\s+mean"
    r"|are\s+you\s+(?:asking|referring)"
    r"|which\s+(?:one|variant|type|kind)\s+(?:of\b|do\s+you|are\s+you)"
    r"|which\s+type\s+of"
    r"|please\s+(?:specify|clarify)"
    r"|before\s+I\s+answer"
    r"|to\s+clarify[,:]"
    r")",
    re.IGNORECASE,
)


_MISSING_CITATION_NUDGE = (
    "Your reply references JO 7110.65 but does not include a specific "
    "section citation (e.g. `§1-1-1`, `§13-1-2(a)`). The airton_c1 "
    "constitution requires a citation on every substantive answer, "
    "and the tool output you just read contains explicit section "
    "anchors. Re-answer with the citation inline — 'per JO 7110.65 "
    "§N-N-N' — or, if the question is out of scope for JO 7110.65, "
    "say so plainly without the reference."
)


# Number-word -> integer lookup for count-claim parsing. Only cover
# 2..12 — higher numbers are rare in ATC prose and mostly appear as
# digits when they do. 'one' is excluded on purpose: 'one of the ...'
# is a different shape than a count claim and would false-positive
# on phrases like 'only one reason'.
_NUMBER_WORDS: dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


# Matches a count claim followed by a list intro. Intended to fire on
# reply patterns like 'The four specific primary purposes of ATC are:',
# 'There are three reasons:', 'four main options include'. The span
# between the count word and the list-intro verb is permissive
# (non-newline chars up to 120 chars) so adjective runs with
# punctuation ('of Air Traffic Control (ATC)') match cleanly. Trailing
# 'are' / 'include' / 'comprise' pins it to a list intro — avoids
# matching narrative prose that happens to contain a number elsewhere.
_COUNT_CLAIM_RE = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    # Require whitespace then a letter after the count word. Excludes
    # section numbers ('7110.65', '2-1-1'), list item markers ('1.'),
    # dates, phone numbers — anything where the digit is immediately
    # followed by punctuation or another digit.
    r"\s+[A-Za-z]"
    r"[^\n]{0,120}?"
    r"\b(?:are|is|include|comprise|consist\s+of)\b(?:\s*(?::|as\s+follows))?",
    re.IGNORECASE,
)


# Matches the start of a numbered or bulleted list item. Requires
# punctuation after digits (`.`, `)`, or `]`) so stray numbers in
# prose don't count. Bullet characters (`-`, `*`, `•`) are standard
# markdown / CommonMark list markers. Multiline flag so we scan every
# line of the reply.
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d+[.\)\]]|[-*•])\s+\S",
    re.MULTILINE,
)


_LIST_COUNT_MISMATCH_NUDGE = (
    "[count mismatch — your reply stated a count of items (e.g. 'the "
    "four ... are') but the enumerated list has a different number of "
    "items. Re-answer with a matching count: either add the missing "
    "items if the source supports them, or restate the count to match "
    "what you actually listed. If the question asked for more items "
    "than the source lists, say so plainly — 'The source lists N, not "
    "what was asked for' — rather than padding or agreeing with the "
    "wrong count.]"
)


def _parse_count_word(token: str) -> int | None:
    """Turn '4' / 'four' into 4. Returns None on unparseable tokens
    so the caller can treat them as 'no count claim'."""
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


# ATC-domain vocabulary markers. Presence in a reply signals the
# content is ATC-focused — used by ScopeRedirectHook to confirm the
# model is discussing JO 7110.65 topics. Not used to decide the
# user's intent (too many legit in-scope lay questions lack explicit
# aviation acronyms — 'What document is required...', 'What is the
# purpose of the order?', etc.). Generic single words like 'flight'
# / 'plane' are omitted so travel/hobby prose doesn't false-positive.
_AVIATION_VOCAB_RE = re.compile(
    r"\b(?:"
    r"squawk|transponder|phraseology|radar|ATC|IFR|VFR|NAVAID|RNAV|"
    r"clearance|runway|taxiway|approach|departure|altitude|heading|"
    r"beacon|ADS[-\s]?B|pilot|controller|aircraft|airspace|vector|"
    r"separation|hijack|emergency|NORDO|MVA|MEA|hold\s+short|"
    r"flight\s+level|JO\s*7110|7110\.65|FAA\s+Order|"
    r"Air\s+Traffic\s+Control|RBN|VOR(?:TAC)?|TACAN|DME|ILS"
    r")\b",
    re.IGNORECASE,
)


# Strong out-of-scope markers. Presence in a user message is the
# positive trigger for ScopeRedirectHook — clearly non-aviation
# vocabulary (biology, programming, cooking, chat/greeting,
# supernatural, household, joke-meta) that no in-scope JO 7110.65
# question should contain. Narrow on purpose: prefer false negatives
# (missing an out-of-scope case) over false positives (nudging a
# valid in-scope lay query). Grow this list as new false-negative
# cases surface — the rule of discipline is 'only add terms that are
# almost never in a JO 7110.65 question.'
_CLEARLY_NON_AVIATION_RE = re.compile(
    r"\b(?:"
    # Biology / animals (farm + common pets)
    r"rooster|chicken|hen|egg|eggs|cow|pig|horse|goat|sheep|dog|cat|fish|"
    r"bird|plant|tree|flower|fungus|bacteria|virus|cell|"
    # More animals — wildlife / exotic, common in jokes & riddles
    r"turtle|tortoise|rabbit|bunny|mouse|rat|snake|lizard|frog|toad|"
    r"elephant|monkey|giraffe|lion|tiger|bear|wolf|fox|deer|moose|"
    r"raccoon|squirrel|hamster|kangaroo|penguin|whale|dolphin|shark|"
    r"octopus|spider|ant|bee|butterfly|"
    # Supernatural / fantasy — clear non-aviation joke vocabulary
    r"ghost|zombie|vampire|werewolf|dragon|unicorn|fairy|demon|angel|"
    r"elf|wizard|witch|troll|goblin|mermaid|santa|"
    # Household items / furniture / appliances
    r"refrigerator|fridge|microwave|oven|toaster|washing\s+machine|"
    r"dishwasher|couch|sofa|mattress|pillow|blanket|lamp|"
    # Food & cooking
    r"recipe|cook|bake|ingredient|breakfast|lunch|dinner|meal|"
    r"pizza|burger|sandwich|pasta|salad|soup|cake|cookie|"
    # Programming & software (not ATC software)
    r"python|javascript|typescript|ruby|rust|golang|react|vue|django|flask|"
    r"sql|bash|shell\s+script|variable|compile|debug|commit|git(?:hub)?|"
    # Consumer tech brands / devices. Bare 'mac', 'pc', 'computer'
    # are intentionally omitted — they collide with aviation
    # acronyms (MAC = Military Airlift Command / mean aerodynamic
    # chord; flight-data computer). These terms don't collide.
    r"iphone|ipad|ipod|android\s+phone|smartphone|laptop|"
    # Math / science (unrelated to ATC domain)
    r"equation|theorem|calculus|algebra|geometry|physics|chemistry|"
    r"astronomy|biology|history|literature|philosophy|"
    # Personal / chat / meta
    r"how\s+are\s+you|tell\s+me\s+about\s+yourself|what(?:'s|\sis)\s+your\s+name|"
    r"tell\s+me\s+a\s+joke|sing\s+(?:me\s+)?a\s+song|"
    # Joke-meta vocabulary
    r"joke|riddle|punchline|funny|hilarious|haha"
    # Entertainment
    r"|football|basketball|baseball|soccer|movie|film|song|album|book|novel"
    r")\b",
    re.IGNORECASE,
)


# Joke-frame structural regex. Common joke setups almost never ask
# real in-scope questions even when their specific nouns aren't in
# the non-aviation vocab list. Targeted patterns; add more as new
# shapes surface. 'If a X, a Y, and a Z...' is the lead-in from the
# observed 'ghost/turtle/refrigerator' repro. 'Why did the X...' and
# 'knock knock' are the other evergreen shapes.
_JOKE_FRAME_RE = re.compile(
    r"(?:"
    # 'If a X, a Y, and a Z ...' — multi-noun absurd setup, tolerant
    # of optional commas and 'and' connectors between items.
    r"\bif\s+a\s+\w+(?:\s*,\s*(?:and\s+)?(?:a|an)\s+\w+){2,}"
    # 'Why did the X cross/go/do ...' — classic joke opener
    r"|\bwhy\s+did\s+the\s+\w+\s+(?:cross|go|do|say|want|need)"
    # 'Knock knock' literal
    r"|\bknock[\s,-]+knock\b"
    # 'What do you call a X when ...' / 'What's the difference between X and Y'
    r"|\bwhat\s+do\s+you\s+(?:call|get)\s+(?:a|an|when)"
    r"|\bwhat(?:'s|\sis)\s+the\s+(?:difference|similarity)\s+between\s+\w+\s+and\s+\w+"
    # 'X walks into a bar' / 'three men walk into a bar' — the
    # evergreen bar-joke opener. Matches anything preceding 'walk(s)
    # into a/the bar'.
    r"|\bwalks?\s+(?:in)?to\s+(?:a|the)\s+bar\b"
    # 'Who won?' / 'Who wins?' riddle ending. ATC questions ask
    # procedural / definitional things ('who must issue X'); a
    # 'who won / who wins' tag is characteristic of race / contest
    # jokes ('Five Macs and PC join a computer race. Who won?').
    r"|\bwho\s+(?:won|wins)\b"
    r")",
    re.IGNORECASE,
)


# Phrases a reply uses when it IS correctly scope-redirecting.
# A reply that matches this pattern is already doing the right thing
# (declining, naming its own scope, pointing the user elsewhere) —
# even if it happens to mention JO 7110.65 / ATC by name to say 'that
# is NOT what I cover.' Exempting the reply from the scope mismatch
# catcher prevents a successful refusal from being nudged into a
# pointless retry loop.
_SCOPE_REDIRECT_MARKER_RE = re.compile(
    r"(?:"
    r"outside\s+(?:of\s+|the\s+)?(?:JO|FAA|the\s+order|scope|my\s+scope)"
    r"|outside\s+the\s+scope"
    r"|not\s+(?:in|within|part\s+of)\s+(?:scope|JO|the\s+order|my\s+scope)"
    r"|cannot\s+answer"
    r"|can(?:not|'t|\snot)\s+answer"
    r"|I\s+(?:am|'m)\s+(?:a\s+)?specialist"
    r"|that\s+(?:question\s+)?is\s+outside"
    r"|doesn(?:'t|\snot)\s+cover"
    r"|out\s+of\s+scope"
    r"|not\s+something\s+I\s+cover"
    r"|biological\s+query"
    r")",
    re.IGNORECASE,
)


_SCOPE_REDIRECT_NUDGE = (
    "[scope mismatch — the user's message has no aviation or ATC "
    "terminology, but your reply is discussing JO 7110.65 / ATC "
    "content. airton_c1 is scoped to FAA JO 7110.65 (Air Traffic "
    "Control) only. Respond with an explicit scope-redirect: 'That "
    "question is outside JO 7110.65. I'm a JO 7110.65 specialist — "
    "I can't answer it.' Do NOT answer from priors, do NOT continue "
    "a prior turn's topic into this new unrelated question, and do "
    "NOT fabricate an in-scope interpretation.]"
)


# Ambiguous-term dictionary: maps a user-side term regex to a tuple
# of axes, where each axis is a tuple of mutually-exclusive qualifier
# patterns. Axes structure is load-bearing — a reply with multiple
# alternatives from THE SAME axis ('manned or unmanned') is offering
# a choice / asking for clarification; a reply with exactly one match
# per axis has committed to a specific variant.
#
# If the user's message contains the bare term without ANY qualifier
# (across all axes), and the reply picks exactly one qualifier in at
# least one axis without offering choice in any axis, the hook
# nudges.
#
# Rule of discipline: only add an entry when BOTH (a) the JO treats
# variants of this term with materially different rules AND (b) users
# plausibly ask the bare term without specifying the variant. Each
# entry is session-regression-backed.
_AMBIGUOUS_TERMS: tuple[tuple[re.Pattern[str], tuple[tuple[re.Pattern[str], ...], ...]], ...] = (
    # Balloons — unmanned free balloons fall under §9-6 (distinct
    # controller procedures: traffic advisory, no vertical separation
    # without verified altitude, derelict-balloon handling). Manned
    # balloons are handled as general aircraft. Session 2026-04-24
    # repro: user asked about 'a balloon' without specifying; model
    # answered for the unmanned-free variant silently.
    (
        re.compile(r"\bballoons?\b", re.IGNORECASE),
        (
            # Axis 1: crew
            (
                re.compile(r"\bmanned\b", re.IGNORECASE),
                re.compile(r"\bunmanned\b", re.IGNORECASE),
            ),
            # Axis 2: tether
            (
                re.compile(r"\bfree\s+balloon", re.IGNORECASE),
                re.compile(r"\btethered\b", re.IGNORECASE),
                re.compile(r"\bmoored\b", re.IGNORECASE),
            ),
            # Axis 3: lift medium / use
            (
                re.compile(r"\bhot\s+air\s+balloon", re.IGNORECASE),
                re.compile(r"\bgas\s+balloon", re.IGNORECASE),
                re.compile(r"\bweather\s+balloon", re.IGNORECASE),
            ),
        ),
    ),
)


_AMBIGUOUS_CONTEXT_NUDGE = (
    "[ambiguous context — the user's message contains a term whose "
    "correct handling depends on context they did not provide. For "
    "example, 'balloon' covers unmanned free balloons (JO 7110.65 "
    "§9-6, distinct controller procedures) AND manned balloons "
    "(treated under general aircraft rules) — the answer differs. "
    "Your reply assumed a specific variant instead of asking. "
    "Re-answer by asking the user to clarify the variant FIRST, then "
    "answer only after they provide it. Do not pick a variant and "
    "proceed.]"
)


@dataclass(frozen=True)
class AmbiguousContextHook:
    """Nudge replies that silently assume a specific variant of an
    ambiguous term instead of asking for clarification.

    Failure mode this catches: user asks about 'a balloon'. JO 7110.65
    treats unmanned free balloons (§9-6) with distinct rules from
    manned balloons (general aircraft rules). Model silently fills in
    'unmanned free balloon' and answers — robbing the student of the
    chance to learn that variant matters. Session 2026-04-24 repro:
    'a single prop squawking 1200 and a balloon are intersecting, who
    has the right of way?' — model inserted 'unmanned free' without
    asking.

    Trigger conditions (ALL must hold, checked per ambiguous term):
      1. `user_message` threaded through.
      2. User message contains the ambiguous term (bare form).
      3. User message does NOT contain any of the term's
         disambiguating qualifiers.
      4. Reply DOES contain at least one of the disambiguating
         qualifiers.

    Action: Nudge. Retry-able — the model's next round should ask a
    clarifying question ('manned or unmanned?') and wait for the
    answer instead of proceeding with an assumed variant.

    The `_AMBIGUOUS_TERMS` dictionary is the maintenance surface.
    Extend with new entries only when both (a) the JO treats variants
    differently and (b) a real user-visible miss occurs.

    Placed after ScopeRedirectHook in the bail list so scope
    mismatches (out-of-scope prompts) take priority over ambiguity
    nudges inside in-scope prompts."""

    name: str = "ambiguous_context"

    def check(self, ctx: BailContext) -> BailOutcome:
        user = ctx.user_message
        if user is None or not user.strip():
            return Continue()
        content = ctx.reply.content
        for term_re, axes in _AMBIGUOUS_TERMS:
            if not term_re.search(user):
                continue
            # User already specified any qualifier across any axis —
            # no ambiguity to challenge.
            all_qualifiers = [q for axis in axes for q in axis]
            if any(q.search(user) for q in all_qualifiers):
                continue
            # Per-axis match counts. Reply offers a choice if ANY
            # axis has >= 2 alternatives mentioned; reply commits if
            # at least one axis has exactly 1.
            axis_hits = [sum(1 for q in axis if q.search(content)) for axis in axes]
            if any(h >= 2 for h in axis_hits):
                # At least one axis shows multi-alternative mention —
                # reply is asking / comparing, not silently picking.
                continue
            if any(h == 1 for h in axis_hits):
                return Nudge(_AMBIGUOUS_CONTEXT_NUDGE)
        return Continue()


@dataclass(frozen=True)
class ScopeRedirectHook:
    """Nudge replies that continue JO 7110.65 / ATC content when the
    user's question has no aviation vocabulary at all.

    Failure mode this catches: user asks an out-of-scope question
    ('do roosters lay eggs', 'help me write Python'). Retrieval scores
    are uniformly weak — airton_c1's corpus is JO 7110.65 only, so
    there's nothing relevant to surface. The model, seeing weak
    retrieval + strong prior-turn context in its working history,
    latches onto the previous turn's topic instead of scope-redirecting.
    Session 2026-04-24 repro: 'do roosters lay eggs' got the previous
    turn's phraseology-correction content emitted back.

    Trigger conditions — ANY of the three positive signals trips
    the hook, provided the reply is not already a scope redirect:

      Signal A (user vocab): User message contains clearly
        non-aviation vocabulary (`_CLEARLY_NON_AVIATION_RE` —
        biology/cooking/programming/chat meta/entertainment/
        supernatural/household/joke-meta).
      Signal B (user shape): User message matches a joke-frame
        structural pattern (`_JOKE_FRAME_RE` — 'If a X, a Y, and a
        Z...', 'Why did the X...', 'knock knock', 'X walks into a
        bar'). Catches absurd setups even when the specific nouns
        aren't in vocab.
      Signal C (reply bleed): The REPLY contains BOTH aviation vocab
        AND clearly-non-aviation vocab. This is defense-in-depth for
        context-bleed (model pulled prior-turn non-aviation material
        into the current reply) — even if the user's latest message
        has no suspicious markers, a reply that mixes 'roosters don't
        lay eggs' with 'per §7-6-11' is confused.

    Signals A and B fire regardless of whether the reply has ATC
    vocabulary — a plain off-topic answer ('the bartender says hi')
    is just as much a scope failure as a context-bleed answer.
    airton_c1 must redirect off-topic prompts, not answer them in
    kind. Signal C keeps its reply-bleed gate (needs both aviation
    AND non-aviation vocab) because it exists to catch cases where
    the user's message gave no signal at all.

    Gate on substantiveness (user message >= 15 chars) so short
    follow-ups don't trip; exempt replies already shaped as scope
    redirects (`_SCOPE_REDIRECT_MARKER_RE`) so a correct refusal
    doesn't loop.

    Action: Nudge. Retry-able — on the next round the model should
    emit a scope-redirect sentence. Halt would be too brutal; a
    nudge lets the model actually produce a useful 'not in scope'
    message rather than a canned refusal.

    Placed last in the bail list so domain-specific shape catchers
    (fabrication / count / citation / reserved-code) run first on
    in-scope replies."""

    name: str = "scope_redirect"

    def check(self, ctx: BailContext) -> BailOutcome:
        user = ctx.user_message
        if user is None or len(user.strip()) < 15:
            return Continue()
        content = ctx.reply.content
        # Exempt replies already correctly scope-redirecting.
        if _SCOPE_REDIRECT_MARKER_RE.search(content):
            return Continue()
        # Signal A / B: user-side triggers. Fire regardless of
        # reply shape — an off-topic answer to an off-topic prompt
        # is still a scope failure.
        signal_user_vocab = bool(_CLEARLY_NON_AVIATION_RE.search(user))
        signal_joke_frame = bool(_JOKE_FRAME_RE.search(user))
        # Signal C: reply mixes aviation AND clearly-non-aviation
        # content. Only meaningful when BOTH appear — that's the
        # context-bleed shape.
        signal_reply_bleed = bool(
            _AVIATION_VOCAB_RE.search(content) and _CLEARLY_NON_AVIATION_RE.search(content)
        )
        if not (signal_user_vocab or signal_joke_frame or signal_reply_bleed):
            return Continue()
        return Nudge(_SCOPE_REDIRECT_NUDGE)


# Reserved transponder codes are pilot-initiated emergency signals.
# Controllers OBSERVE them via 5-2-5 / 5-2-8 procedures; they never
# assign them as routine phraseology. Catcher guards against a model
# that uncritically reformats a user-provided reserved code into
# correct-looking ATC phraseology (session 2026-04-24 repro).
#
# 7400 (UAS lost link per §5-2-6) is intentionally excluded — it's
# UAS-specific and airtime is rare in controller phraseology; adding
# it would net-increase false-positive risk without matching the
# observed failure mode.
_RESERVED_SQUAWK_RE = re.compile(
    # Left-anchor: forbid a leading letter or digit (so 'resquawk' or
    # similar doesn't match). `\b` would block matches with leading
    # underscore because `_` is a word char — intentionally permissive
    # here to catch `_squawk ..._` markdown italic.
    r"(?<![A-Za-z0-9])squawk\s+"
    r"(?:"
    r"7500|7600|7700"
    # Digit-by-digit readback (JO phraseology convention):
    # 'seven five zero zero', etc.
    r"|seven\s+five\s+zero\s+zero"
    r"|seven\s+six\s+zero\s+zero"
    r"|seven\s+seven\s+zero\s+zero"
    # Grouped-hundreds (lay / student paraphrase):
    # 'seventy five hundred', 'seven five hundred', etc.
    r"|seven(?:ty)?\s+five\s+hundred"
    r"|seven(?:ty)?\s+six\s+hundred"
    r"|seven(?:ty)?\s+seven\s+hundred"
    r")"
    # Right-anchor: forbid a trailing digit or letter (so '75002' / 'seven
    # five hundredth' don't shadow the match). Underscore / punctuation /
    # whitespace / end-of-string all qualify — this matters because the
    # default `\b` treats `_` as a word char, which would have blocked
    # matches inside markdown italic like `_squawk 7500_`.
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


_RESERVED_SQUAWK_NUDGE = (
    "[reserved-squawk-code — your reply proposes assigning a reserved "
    "transponder code (7500 = hijack/unlawful interference per §5-2-5; "
    "7600 = comm failure; 7700 = emergency per §5-2-8). These are "
    "PILOT-INITIATED emergency signals — controllers observe them, "
    "never assign them as routine phraseology. For VFR radar service "
    "termination the correct code is 'squawk VFR' or 'squawk one two "
    "zero zero' per §5-2-7. Re-answer with the correct code; if the "
    "user's input contained a reserved code, call that out explicitly "
    "rather than silently reformatting it.]"
)


def _is_squawk_assignment_context(content: str, start: int, end: int) -> bool:
    """Decide whether a `squawk <reserved>` match is asserting an
    assignment vs. explaining the code's meaning.

    Assignment signals (any one is sufficient):
      - Match text starts with all-caps 'SQUAWK' (JO phraseology
        convention — actual phraseology lines in the order are
        capitalized).
      - Match is wrapped in paired double-quotes on the same line
        (a quoted phraseology line like
        `"Radar service terminated, squawk seven five hundred."`).
      - Match is wrapped in `_..._` (markdown italic — used in the
        corpus for PHRASEOLOGY-block emphasis).
      - Match is wrapped in `**...**` (markdown bold — reply-side
        formatting for proposed phraseology).

    Explanation / description contexts (none of the above present)
    pass through cleanly — so prose like
    'When you observe Code 7500, apply §10-2-6' or a warning like
    'squawk 7500 is the hijack code' doesn't trip the hook.
    """
    if content[start:end].startswith("SQUAWK"):
        return True
    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", end)
    if line_end == -1:
        line_end = len(content)
    prefix = content[line_start:start]
    suffix = content[end:line_end]
    # Paired quote/italic/bold anchors on the same line. We require
    # BOTH sides to carry the anchor so an unpaired `*` (common in
    # bullet lists) doesn't false-positive.
    if '"' in prefix and '"' in suffix:
        return True
    if "_" in prefix and "_" in suffix:
        return True
    return "**" in prefix and "**" in suffix


@dataclass(frozen=True)
class ReservedSquawkCodeHook:
    """Nudge a reply that assigns a reserved transponder code (7500 /
    7600 / 7700) in routine controller phraseology.

    Failure mode this catches: user transcribes incorrect phraseology
    containing a reserved code ('squawk seventy five hundred'). Model
    correctly fixes the phraseology shell ('Services stopped' ->
    'Radar service terminated') but blindly reformats the code value
    ('seventy five hundred' -> 'seven five hundred' i.e. 7500),
    propagating a hijack-code assignment into a 'correct' reply.
    Session 2026-04-24 repro — the corrected phraseology was still
    unsafe because 7500 is pilot-set when the aircraft is being
    hijacked; controllers do not assign it.

    Corpus anchors:
      §5-2-5 — HIJACK/UNLAWFUL INTERFERENCE (Code 7500 observation)
      §5-2-7 — VFR CODE ASSIGNMENTS (correct VFR code = 1200 / 'VFR')
      §5-2-8 — note on Code 7700 emergency activation

    Trigger conditions (ALL must hold):
      1. Reply contains `squawk <reserved code>` in any of its digit,
         digit-by-digit-readback, or grouped-hundreds forms.
      2. The match is in an ASSIGNMENT context — wrapped in quotes,
         italic, bold, or all-caps SQUAWK (see
         `_is_squawk_assignment_context`). Narrative descriptions
         ('when you observe Code 7500') pass through.

    Action: Nudge. Retry-able — model gets a round to swap in the
    correct code ('squawk VFR' / '1200') and, ideally, to flag that
    the user's input contained a reserved code. Do NOT Halt: the
    model's phraseology shell is usually correct; we want to fix the
    code, not lose the correction entirely."""

    name: str = "reserved_squawk_code"

    def check(self, ctx: BailContext) -> BailOutcome:
        content = ctx.reply.content
        # Lower-cased user_message for verbatim-echo detection. When
        # the model quotes the user's question to flag it as wrong
        # ('"Services stopped, squawk seventy five hundred" is
        # incorrect — the right form is ...'), the reserved-code match
        # lives inside an echo of the prompt, not a new assignment.
        # Compare lowercase since match text may differ in case
        # ('SQUAWK 7500' vs 'squawk 7500').
        user_lower = (ctx.user_message or "").lower()
        for match in _RESERVED_SQUAWK_RE.finditer(content):
            if not _is_squawk_assignment_context(content, match.start(), match.end()):
                continue
            if user_lower and match.group(0).lower() in user_lower:
                # Model echoed the user's exact reserved-code phrase —
                # not a new assignment proposal.
                continue
            return Nudge(_RESERVED_SQUAWK_NUDGE)
        return Continue()


@dataclass(frozen=True)
class ListCountMismatchHook:
    """Nudge replies whose stated count of items disagrees with the
    number of items the reply actually enumerates.

    Failure mode this catches: the user asks for N items, the model
    echoes 'The N ... are' and then emits an enumerated list of M != N
    items. Session 2026-04-24 repro: user asked for 4 primary purposes
    of ATC; model wrote 'The four specific primary purposes ... are as
    follows:' and listed 3. Self-falsifying — the reply contradicts
    itself within three sentences.

    Trigger conditions (ALL must hold):
      1. Exactly one count claim in the reply (matches
         `_COUNT_CLAIM_RE` and the number word / digit resolves to a
         known integer). Multiple claims are ambiguous — skip rather
         than pick the wrong one to check against.
      2. Reply contains an enumerated list of >= 2 items
         (`_LIST_ITEM_RE` multiline count). A single-item 'list' is
         prose, not an enumeration — skip.
      3. The claim count != the list count.

    Action: Nudge the model to re-answer with a consistent count.
    Retry-able — typical failure is 'model over-agreed with user's
    wrong premise'; retry lets the model push back on the premise
    instead of fabricating a missing item.

    Placed AFTER `missing_citation` in the bail list: once compliance
    (citation) is satisfied, internal consistency (count) is the next
    layer of quality.

    Silent cases (by design):
      - Zero count claims -> nothing to verify.
      - Multiple count claims -> ambiguous which is 'the' claim.
      - No enumerated list -> reply is narrative prose; count could
        be correct or the list implied but not formalized, so don't
        fire."""

    name: str = "list_count_mismatch"

    def check(self, ctx: BailContext) -> BailOutcome:
        content = ctx.reply.content
        # Fast-path: no digits/number-words -> no possible claim.
        if not re.search(
            r"\b(?:\d+|two|three|four|five|six|seven|eight|"
            r"nine|ten|eleven|twelve)\b",
            content,
            re.IGNORECASE,
        ):
            return Continue()
        claim_matches = _COUNT_CLAIM_RE.findall(content)
        if len(claim_matches) != 1:
            return Continue()
        claimed = _parse_count_word(claim_matches[0])
        if claimed is None:
            return Continue()
        list_count = len(_LIST_ITEM_RE.findall(content))
        if list_count < 2:
            return Continue()
        if claimed == list_count:
            return Continue()
        return Nudge(_LIST_COUNT_MISMATCH_NUDGE)


@dataclass(frozen=True)
class MissingCitationHook:
    """Nudge a reply that references JO 7110.65 substantively but
    doesn't include a specific `§N-N-N` / `TBL N-N-N` citation. Mirror
    image of `UngroundedCitationHook` — that one catches citation
    without grounding; this one catches grounding without citation.

    Failure mode this catches: airton_c1's constitution says 'Always
    cite at least one JO 7110.65 section when answering.' A grounding
    tool ran, the tool output surfaced a section, the reply paraphrases
    from that section correctly, but the model omits the §-anchor.
    Session 2026-04-24 reproducers: 'What is the purpose of 7110.65?'
    -> correct summary of §1-1-1, no citation. 'What document is
    required for jointly applied procedures?' -> correct summary of
    §1-1-10, no citation. Both leave the student without a way to
    verify or locate the source.

    Trigger conditions (ALL must hold):
      1. A grounding tool ran this turn (`ctx.tools_ran` intersects
         `_GROUNDING_TOOLS`). Without one there's nothing to cite
         from — that's `ungrounded_citation`'s domain.
      2. The reply contains EITHER a JO 7110.65 reference
         (`_ORDER_REFERENCE_RE`) OR a JO-normative phraseology
         marker (`_JO_PHRASEOLOGY_MARKERS_RE`) — all-caps
         controller phraseology like 'RADAR SERVICE TERMINATED' or
         'SQUAWK VFR' that makes the reply a JO claim even without
         naming the order. Non-airton_c1 characters never hit
         either signal, so the hook is naturally character-scoped.
      3. The reply is substantive (>= 80 chars). Short replies are
         usually refusals or scope-redirects that don't need a cite.
      4. The reply does NOT contain a `§N-N-N` / `TBL N-N-N` anchor.

    Action: Nudge the model to re-answer with the citation inline.
    Retry-able — the model gets another round to add the citation.
    Do NOT Halt: losing a correct substantive answer to a canned
    refusal over a missing anchor is worse than the missing anchor
    itself.

    Placed AFTER `tool_intent` in the bail list: fabrication-shape
    catchers (teaser/false_success/meta_confirm/fabricated_*) all run
    first, so a fabricated-looking reply gets the fabrication-specific
    nudge rather than this one. Compliance comes after correctness."""

    name: str = "missing_citation"

    def check(self, ctx: BailContext) -> BailOutcome:
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        content = ctx.reply.content
        if len(content) < 80:
            return Continue()
        in_scope = bool(_ORDER_REFERENCE_RE.search(content)) or bool(
            _JO_PHRASEOLOGY_MARKERS_RE.search(content)
        )
        if not in_scope:
            return Continue()
        if _CITATION_PRESENT_RE.search(content):
            return Continue()
        # Scope-redirect replies name the order to explain what's
        # NOT covered ('That question is outside JO 7110.65'). Those
        # are declining to answer, not making a substantive claim —
        # don't demand a citation.
        if _SCOPE_REDIRECT_MARKER_RE.search(content):
            return Continue()
        # Clarifying-question replies ('Could you clarify manned or
        # unmanned?') are ASKING, not asserting. Exempt — a citation
        # in a question would be weirdly presumptive and looping the
        # retry would discard a perfectly good clarifying reply
        # (harness-5uq follow-up).
        if _CLARIFYING_QUESTION_RE.search(content):
            return Continue()
        return Nudge(_MISSING_CITATION_NUDGE)


# ---------- post-model hooks ----------


@dataclass(frozen=True)
class PairedMetaConfirmStripHook:
    """Strip meta-confirm narrative from a reply that ALSO carries a
    valid tool call ("Would you like me to …? <tool_call>…"). The tool
    call is real but the narrative is noise — clearing it keeps the
    wrap-up round from re-reading hallucinated confirmation prose."""

    name: str = "paired_meta_confirm_strip"

    def check(self, ctx: PostModelContext) -> PostModelOutcome:
        reply = ctx.reply
        if not reply.tool_calls:
            return Continue()
        if not META_CONFIRM_RE.search(reply.content):
            return Continue()
        return Replace(
            ModelReply(
                content="",
                tool_calls=reply.tool_calls,
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


# ---------- pre-tool hooks ----------


@dataclass(frozen=True)
class DuplicateCallHook:
    """Short-circuit an identical (name, arguments) re-invocation this
    turn. Small models sometimes wrap a real answer around a redundant
    re-call; feeding the canned nudge back as the tool-role message lets
    the next round close out (harness-pun)."""

    name: str = "duplicate_call"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        key = _call_key(ctx.call)
        if key not in ctx.seen_calls:
            return Continue()
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=DUPLICATE_CALL_NUDGE,
                success=True,
            )
        )


# Nudge fed back as the tool-role message when the grounding hook
# skips a call. Phrased so the next round knows exactly what failed
# (the arg that didn't trace back to the user) and what the remedy is
# (re-plan from what the user ACTUALLY said). Deliberately avoids
# imperatives like "call this tool instead" — the hook can't know the
# right replacement, only that the current one is off-prompt.
_ARG_GROUNDING_NUDGE = (
    "[grounded-args check failed — your tool arguments named entities "
    "({offenders}) that the user did not mention. The user's message "
    "was: {user_message!r}. Re-read that message and either call the "
    "tool with arguments derived from it, or tell the user you cannot "
    "answer. Do NOT reuse entities from memory / prior turns / your "
    "own prior replies.]"
)


@dataclass(frozen=True)
class ArgumentGroundingHook:
    """Reject tool calls whose argument entities don't trace back to the
    current user turn.

    Failure mode this catches: router or main model picks a specific
    domain / URL that appears nowhere in the user's message. The entity
    leaks in from retrieved episodic memory, a prior conversation, or
    the small router model's training bias. Stackoverflow-asked prompt
    becomes a `search_web(query='dailydrop.fm')` call — coherent-sounding
    but wrong. Without this hook, the orchestrator executes the call,
    the main model wraps a confident-looking summary around the
    irrelevant result, and the user gets a fabricated answer.

    Rule (narrow by design):
      1. Extract domain-like roots from the tool's string arguments.
      2. For each root, check whether it appears as a substring in the
         user message (case-insensitive).
      3. If ANY root has no match in the user message, Skip the call
         with a re-plan nudge.

    Skip conditions (never fire):
      - No user_message threaded through the context (rare; bootstrap
        cases / subagent calls).
      - No domain-like tokens in the args (prose queries pass; we only
        flag entity-specific arguments).
      - Every domain root is grounded in the user message.

    The hook runs AFTER `duplicate_call` so re-calls short-circuit first
    without a spurious grounding nudge. The attribution eval can disable
    it by name ('argument_grounding') to measure which failures it
    uniquely catches."""

    name: str = "argument_grounding"

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if ctx.user_message is None:
            return Continue()
        arg_strings = _arg_string_values(ctx.call.arguments)
        if not arg_strings:
            return Continue()
        roots = _arg_domain_roots(arg_strings)
        if not roots:
            return Continue()
        user_lower = ctx.user_message.lower()
        ungrounded = sorted(root for root in roots if root not in user_lower)
        if not ungrounded:
            return Continue()
        offenders = ", ".join(repr(r) for r in ungrounded)
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_ARG_GROUNDING_NUDGE.format(
                    offenders=offenders,
                    user_message=ctx.user_message,
                ),
                success=False,
                error="argument_grounding",
            )
        )


# ---------- finalize hooks ----------


# JO-style section citation: `§ 12-1-2`, `§12-1-2`, tolerant of hyphen /
# en-dash / unicode-minus. Three numeric segments separated by dashes
# pins it to the JO 7110.65 shape (chapter-section-paragraph) without
# false-positiving on prices (`$19.99`), simple anchors (`§2`), or
# two-segment numbering (`§3-1`). Conservative by design — false
# positives here would replace good replies with a canned refusal.
UNGROUNDED_SECTION_CITATION_RE = re.compile(
    r"§\s*\d+[-–−]\d+[-–−]\d+",  # noqa: RUF001 — en-dash + minus are load-bearing variants seen in corpus
)


# The set of tools whose execution constitutes "grounding" for a
# section-citation reply. Intentionally narrow: only memory-backed
# retrieval counts. `fetch_url`, `search_web`, `read_file`, etc. do
# NOT count — a web-fetch result is not a substitute for having pulled
# the citation from character-owned memory.
_GROUNDING_TOOLS: frozenset[str] = frozenset({"search_memory", "fact_search"})


# Canned refusal shown when the hook fires. Character-agnostic — the
# character's `cite_or_silent` / scope text lives in its constitution
# and drives the system prompt, but this hook runs after the reply is
# already generated and doesn't have a hook into per-character refusal
# templates. Keep the wording in the first person + short so it reads
# as the character's own voice regardless of who's speaking.
UNGROUNDED_CITATION_FALLBACK = (
    "I can't answer that from memory without fabricating a citation. "
    "Either that topic is outside what I cover, or I don't have the "
    "source material for it indexed. If you have the reference handy, "
    "paste it and I'll work from there."
)


LOW_CONFIDENCE_FALLBACK = (
    "I don't have a solid source for this — the best retrieval match I "
    "could find scored below my confidence threshold. Check the relevant "
    "section directly rather than relying on what I said."
)


# Default retrieval-score threshold below which citations that weren't
# explicitly grounded by a tool's structured return are treated as
# low-confidence and refused. 0.5 matches the passive-retrieval floor
# used in cli.py (`--memories-threshold`). Tunable per-hook at
# construction; the threshold-calibration follow-up (harness-5c0) will
# histogram the real distribution and move this to a character-config
# override.
_LOW_CONFIDENCE_THRESHOLD_DEFAULT = 0.5


@dataclass(frozen=True)
class LowConfidenceFallbackHook:
    """Catch section-citation replies where a grounding tool DID run
    but scored below the confidence threshold AND the cited section
    isn't in the tool's declared `citations_grounded` (harness-ywp.3).

    Complementary to UngroundedCitationHook:
      * UngroundedCitationHook fires when NO grounding tool ran.
      * LowConfidenceFallbackHook fires when a grounding tool ran but
        weakly (top hit below threshold) and the reply cites a
        section the tool didn't surface.

    Trigger conditions (ALL must hold):
      1. `ctx.retrieval_top_score` is set and < `threshold`.
      2. Reply contains at least one JO-style `§N-N-N` / `§N-N` /
         TBL/Table/FIG/Figure N-N-N citation (same extractor the
         audit log + tool-declared grounding use — harness-ywp.5).
      3. At least one of those citations is NOT in
         `ctx.citations_grounded` — i.e. the model cited something
         the tool didn't actually ground.

    Action: replace the reply with a character-agnostic fallback that
    tells the user the retrieval was weak and to check the source
    directly. Same reasoning as UngroundedCitationHook's refusal —
    a confident-looking answer riding on weak retrieval is worse than
    an honest 'I'm not sure, go look'.

    Runs BEFORE UngroundedCitationHook in the finalize phase. When
    THIS hook halts, the later hook never runs (its preconditions
    overlap but UngroundedCitationHook's tools_ran check will usually
    Continue here anyway — a tool did run)."""

    name: str = "low_confidence_fallback"
    threshold: float = _LOW_CONFIDENCE_THRESHOLD_DEFAULT

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if ctx.retrieval_top_score is None:
            return Continue()
        if ctx.retrieval_top_score >= self.threshold:
            return Continue()
        from harness.tools.citations import extract_citations

        cited = extract_citations(ctx.reply.content)
        if not cited:
            return Continue()
        ungrounded = cited - ctx.citations_grounded
        if not ungrounded:
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=LOW_CONFIDENCE_FALLBACK,
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


@dataclass(frozen=True)
class UngroundedCitationHook:
    """Catch section-citation replies that never touched a grounding
    tool and had no memory block attached to the prompt (harness-oc8).

    Failure mode this catches: character is scoped to a specific source
    document (e.g. airton_c1 / JO 7110.65). User asks an out-of-scope
    or poorly-retrieved question. Retrieval scores all memory hits
    below the floor, so no memory block is attached. The main loop
    never fires a grounding tool. The model then fabricates a
    parametric-knowledge answer and closes with a real-looking section
    number — real `§N-N-N` string, fabricated attribution. Classic
    ungrounded citation.

    Trigger conditions (ALL must hold):
      1. Reply contains at least one JO-style `§N-N-N` citation.
      2. No memory block was attached for this turn
         (`ctx.memory_block_attached is False`).
      3. No grounding tool (`search_memory` / `fact_search`) ran this
         turn (`ctx.tools_ran` disjoint from `_GROUNDING_TOOLS`).

    Action: replace the reply with a character-agnostic refusal. We
    deliberately do NOT try to repair the reply by stripping just the
    citation — a confident fabricated body minus its source line is
    worse than a refusal.

    Runs BEFORE `fabrication_fallback` in the finalize phase: if this
    hook Halts, the fallback never runs (which is correct — our
    replacement is the terminal answer). If the reply has no section
    citation, we Continue and the fallback (if any) fires normally."""

    name: str = "ungrounded_citation"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if ctx.memory_block_attached:
            return Continue()
        if ctx.tools_ran & _GROUNDING_TOOLS:
            return Continue()
        if not UNGROUNDED_SECTION_CITATION_RE.search(ctx.reply.content):
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=UNGROUNDED_CITATION_FALLBACK,
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


# Matches a markdown pipe-table row: whitespace + `|` + at least two
# cells separated by more `|`s. Column count ≥3 pins it to "real table"
# shape — a stray `a | b` in prose doesn't trigger. We strip the row
# and its cells at check time, so tolerant of surrounding whitespace.
_PIPE_ROW_RE = re.compile(r"^\s*\|[^|\n]+\|[^|\n]+\|[^\n]*\|\s*$", re.MULTILINE)


# Markdown header separator row (e.g. `|---|---|---|`). These carry no
# data; excluded before we check rows against tool outputs so a missing
# separator never fires the catcher.
_PIPE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$", re.MULTILINE)


# Pull the digit-bearing tokens we care about. Two-or-more-digit runs
# avoid coincidental single-digit matches ("3 miles" prose colliding
# with "3,000" in tool output). Accepts commas and decimals as internal
# separators so "1,000" / "1.25" survive as single tokens.
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d,.]{1,}")


def _normalize_for_compare(s: str) -> str:
    """Collapse whitespace + strip commas so "|MH|Under 50|25|" matches
    "| MH | Under 50 | 25 |" and "1,000" matches "1000". We keep the
    `|` delimiter intact — the catcher's signal is cell-pair adjacency,
    and pipes are the row scaffolding we want to preserve."""
    return re.sub(r"\s+", "", s).replace(",", "")


# Canned refusal when the table-fabrication catcher fires. Stays
# character-agnostic — same reasoning as UNGROUNDED_CITATION_FALLBACK:
# by the time we're replacing the reply we don't have a hook into
# per-character refusal templates, and first-person keeps it reading
# as the character's own voice. Short-and-specific beats generic: the
# user gets a clear signal that the table (not the whole answer) was
# the suspect part.
TABLE_FABRICATION_FALLBACK = (
    "I can't reproduce that table from memory without risking a "
    "fabricated row. The retrieved source didn't include the full "
    "table, and I won't fill the gaps from priors. If you have the "
    "section's table in front of you, paste it and I'll work from "
    "there; otherwise ask me about a specific row and I'll try to "
    "retrieve it."
)


@dataclass(frozen=True)
class TableFabricationHook:
    """Catch replies whose pipe-table data rows don't appear verbatim
    in any tool output this turn (harness-5uq).

    Failure mode this catches: retrieval surfaces a section whose full
    numeric table was truncated by `SearchMemoryTool`'s body cap (or
    split across chunk boundaries by the chunker). The model correctly
    cites the section, sees the partial table fragment, and
    reconstructs the missing rows from parametric priors — a
    citation-real, table-fabricated reply. Reproducer (session
    2026-04-24, airton_c1): MH class RBN, cited §4-1-1 correctly,
    fabricated `|MH|50 - 1,999|50|` when the source row is
    `|MH|Under 50|25|`.

    Trigger conditions (ALL must hold):
      1. A grounding tool ran this turn (`ctx.tools_ran` intersects
         `_GROUNDING_TOOLS`). Without a grounding tool we defer to
         `ungrounded_citation` — a fabricated table with no grounding
         tool IS a fabricated citation, caught upstream.
      2. Reply contains ≥2 pipe-table rows (header + ≥1 data row).
         Non-separator, digit-bearing rows are the data rows.
      3. At least one data row's normalized pipe-signature does NOT
         appear in `ctx.tool_outputs` (also normalized).

    Action: Halt with the canned refusal. Deliberately does NOT attempt
    to strip just the offending row — leaving an explanation around a
    partial table is a subtler form of the same problem.

    Runs AFTER `ungrounded_citation` (which halts on the harder shape
    of fully-fabricated citations) and BEFORE `fabrication_fallback`.
    If no grounding tool ran but a table is still present, the earlier
    hook handles it; this one covers the narrower "grounded citation,
    fabricated row" case."""

    name: str = "table_fabrication"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        rows = _PIPE_ROW_RE.findall(ctx.reply.content)
        if len(rows) < 2:
            return Continue()
        # Data rows: skip the markdown header separator (`|---|---|`)
        # and the header row itself (first match with no digits). A
        # row with zero numeric tokens can't be a numeric-fabrication
        # target, so we skip it rather than halt on label-only rows.
        data_rows = [
            r for r in rows if not _PIPE_SEPARATOR_RE.match(r) and _NUMERIC_TOKEN_RE.search(r)
        ]
        if not data_rows:
            return Continue()
        joined_outputs = "\n".join(ctx.tool_outputs)
        normalized_outputs = _normalize_for_compare(joined_outputs)
        for row in data_rows:
            if _normalize_for_compare(row) in normalized_outputs:
                continue
            # Row signature absent from tool outputs → fabricated.
            # Halt immediately; one bad row is enough to poison the
            # whole table from the user's perspective.
            reply = ctx.reply
            return Halt(
                ModelReply(
                    content=TABLE_FABRICATION_FALLBACK,
                    tool_calls=(),
                    was_truncated=reply.was_truncated,
                    had_unparseable_call=reply.had_unparseable_call,
                )
            )
        return Continue()


# Unit keywords we know how to recognize in a table header cell and in
# a prose claim. Kept narrow — only the ATC-corpus units that surface
# in JO 7110.65 tables (distances, altitudes, powers, bearings/speeds,
# times). `nm` and `mile` without trailing s are normalized by
# rstrip("s") at compare time so "miles" and "mile" canonicalize the
# same key.
_KNOWN_UNITS: frozenset[str] = frozenset(
    {
        "miles",
        "mile",
        "nm",
        "ft",
        "feet",
        "foot",
        "watts",
        "watt",
        "knots",
        "knot",
        "kt",
        "minutes",
        "minute",
        "min",
        "seconds",
        "second",
        "sec",
    }
)


# Pulls "<label> class ... <value> <unit>" claims out of prose. Label
# is 1-4 uppercase letters (matches atc class codes CL/MH/H/HH, airspace
# letters A/B/C/D/E/G, category codes CAT/III). `class` anchor word is
# required on one side so generic numbers like "50 miles apart" don't
# get picked up as labeled claims. Number allows commas + decimals.
# `DOTALL` + non-greedy `[^.|]*?` so we can span across a line without
# picking up pipe tables or sentence boundaries.
_LABELED_CLAIM_RE = re.compile(
    r"\b(?:([A-Z]{1,4})\s+class|class\s+([A-Z]{1,4}))\b"
    r"[^.|]*?"
    r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(miles?|NM|nm|ft|feet|foot|watts?|knots?|kt|minutes?|min|seconds?|sec)\b",
    re.IGNORECASE | re.DOTALL,
)


def _canon_unit(unit: str) -> str:
    """Lowercase + strip trailing s so 'Miles' / 'miles' / 'mile' all
    canonicalize to the same lookup key. `nm` stays as-is (not a plural
    form) — the rstrip is safe because `nm` has no trailing s to
    strip."""
    return unit.lower().rstrip("s")


def _extract_labeled_claims(reply: str) -> list[tuple[str, str, str]]:
    """Return (label, numeric_value, canonical_unit) triples from the
    reply's prose. Value is comma-stripped so '1,999' canonicalizes to
    '1999' for comparison against tool-output cells."""
    claims: list[tuple[str, str, str]] = []
    for match in _LABELED_CLAIM_RE.finditer(reply):
        label = (match.group(1) or match.group(2) or "").upper()
        value = match.group(3).replace(",", "")
        unit = _canon_unit(match.group(4))
        if label and unit in _KNOWN_UNITS:
            claims.append((label, value, unit))
    return claims


# Table-row shape after pipe-split: non-empty cell list. We strip
# surrounding whitespace + any emphasis markdown (`**bold**` / `*italic*`)
# from each cell so the header lookup finds "Distance" whether the source
# wrote `**Distance**` or just `Distance`.
def _strip_cell(cell: str) -> str:
    return cell.strip().strip("*").strip()


def _parse_pipe_tables(text: str) -> list[list[list[str]]]:
    """Extract pipe tables from `text`. Returns a list of tables;
    each table is a list of rows; each row is a list of stripped cells.
    A table ends when a non-pipe line breaks the run (blank line,
    prose, etc.) — same heuristic a markdown renderer uses."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.count("|") >= 2:
            if _PIPE_SEPARATOR_RE.match(line):
                continue
            cells = [_strip_cell(c) for c in line.strip("|").split("|")]
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def _unit_of_column(header_cell: str) -> str | None:
    """Best-effort unit-of-column parser. Looks for any known unit
    keyword as a substring of the header cell (case-insensitive),
    e.g. 'Distance (miles)' → 'mile'. None if the column isn't
    unit-tagged — we don't try to guess; unlabeled columns get skipped."""
    lower = header_cell.lower()
    for unit in _KNOWN_UNITS:
        if unit in lower:
            return _canon_unit(unit)
    return None


def _build_table_row_map(table: list[list[str]]) -> dict[str, dict[str, frozenset[str]]]:
    """Map {row_label: {unit: {acceptable_values}}} for a single pipe
    table. Label is the first cell of each row after the header.
    Acceptable_values is the set of numeric tokens found in that row's
    cell under the unit-tagged column — a single cell like 'Under 50'
    yields {'50'}; '14,500 - 17,999' yields {'14500', '17999'} so either
    endpoint matches (the model is free to cite the range's top or
    bottom as long as it doesn't invent a value)."""
    if len(table) < 2:
        return {}
    header = table[0]
    unit_columns: list[tuple[int, str]] = []
    for idx, cell in enumerate(header):
        unit = _unit_of_column(cell)
        if unit is not None:
            unit_columns.append((idx, unit))
    if not unit_columns:
        return {}
    out: dict[str, dict[str, frozenset[str]]] = {}
    for row in table[1:]:
        if not row:
            continue
        label = row[0].upper()
        if not label:
            continue
        for col_idx, unit in unit_columns:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            nums = {
                _NUMERIC_TOKEN_RE.match(tok).group(0).replace(",", "")  # type: ignore[union-attr]
                for tok in _NUMERIC_TOKEN_RE.findall(cell)
            }
            if nums:
                existing = out.setdefault(label, {}).get(unit, frozenset())
                out[label][unit] = existing | nums
    return out


@dataclass(frozen=True)
class NumericFabricationHook:
    """Catch prose-form labeled numeric claims that disagree with the
    source table in the retrieved tool output (harness-5uq).

    Failure mode this catches: model retrieves a pipe table, reads the
    row labels, but swaps a different row's value onto the label the
    user asked about. Reproducer (session 2026-04-24 post-restart):
    tool output contained `|MH|Under 50|25|`, reply wrote
    'MH class RBN is 50 miles' — label MH pointed at 25 miles in the
    retrieved table, but the model chose 50 (the H row's value) from
    priors.

    Why this runs after `table_fabrication`: that hook only fires when
    the reply contains a literal pipe table; this one covers the
    structurally identical failure when the reply is prose ('MH is
    50 miles' instead of `| MH | … | 50 |`).

    Trigger conditions (ALL must hold):
      1. A grounding tool ran this turn (`ctx.tools_ran` intersects
         `_GROUNDING_TOOLS`). A prose numeric claim with no grounding
         is ungrounded-citation territory, handled upstream.
      2. Reply contains ≥1 labeled claim extracted by
         `_LABELED_CLAIM_RE` (e.g., 'MH class ... 50 miles').
      3. Some tool output contains a pipe table whose header has a
         column matching the claim's unit, AND whose first column has
         a row matching the claim's label.
      4. That row's unit-column value does NOT contain the claim's
         numeric value.

    Action: Halt with a canned refusal that names the fabrication
    shape (class label vs. cited value). Deliberately does not try to
    repair the reply by swapping in the true value — the model that
    fabricated once may re-ground wrong again next round; a refusal
    preserves trust.

    Silent cases (by design): reply has a labeled claim but no table
    in tool output matching the label → Continue. A claim the catcher
    can't verify against a structural source is out of scope."""

    name: str = "numeric_fabrication"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        if not (ctx.tools_ran & _GROUNDING_TOOLS):
            return Continue()
        claims = _extract_labeled_claims(ctx.reply.content)
        if not claims:
            return Continue()
        tool_text = "\n".join(ctx.tool_outputs)
        tables = _parse_pipe_tables(tool_text)
        if not tables:
            return Continue()
        # Merge every table's row-map into one lookup. Overlap between
        # tables on the same label is rare in practice (one table per
        # topic) but we union values so a label appearing in two tables
        # passes if EITHER matches.
        merged: dict[str, dict[str, frozenset[str]]] = {}
        for table in tables:
            for label, units in _build_table_row_map(table).items():
                for unit, values in units.items():
                    existing = merged.setdefault(label, {}).get(unit, frozenset())
                    merged[label][unit] = existing | values
        for label, value, unit in claims:
            row_units = merged.get(label)
            if row_units is None:
                continue  # label unknown to any tool-output table
            allowed = row_units.get(unit)
            if allowed is None:
                continue  # unit-tagged column not present for this label
            if value in allowed:
                continue  # claim agrees with a known row value
            # Claim's label is in the table, unit-column exists for it,
            # but the value isn't one of the known cell values for that
            # (label, unit). Structural cross-row fabrication.
            reply = ctx.reply
            return Halt(
                ModelReply(
                    content=TABLE_FABRICATION_FALLBACK,
                    tool_calls=(),
                    was_truncated=reply.was_truncated,
                    had_unparseable_call=reply.had_unparseable_call,
                )
            )
        return Continue()


@dataclass(frozen=True)
class FabricationFallbackHook:
    """Substitute the canned refusal when bail retries are exhausted
    and the reply STILL trips a non-truncated bail outcome. Without
    this, the user would see the model's final hallucinated paragraph
    as the turn's answer (harness-24xj)."""

    name: str = "fabrication_fallback"

    def check(self, ctx: FinalizeContext) -> FinalizeOutcome:
        last = ctx.last_outcome
        # Only fabrication-shaped outcomes trigger the fallback. A
        # Continue means no catcher fired (genuine final reply), and a
        # Truncated means the partial reply is worth keeping.
        if not isinstance(last, Nudge):
            return Continue()
        reply = ctx.reply
        return Halt(
            ModelReply(
                content=EXHAUSTED_FABRICATION_FALLBACK,
                tool_calls=(),
                was_truncated=reply.was_truncated,
                had_unparseable_call=reply.had_unparseable_call,
            )
        )


# ---------- pipeline ----------


@dataclass
class HookPipeline:
    """Owns the four phase lists. Methods are pure functions of their
    context + a `disabled` frozenset — they never mutate module state
    — which makes them trivially testable and safe to reuse across
    concurrent turns. The tool loop constructs one pipeline at wiring
    time and re-uses it for every turn."""

    bail: list[BailHook] = field(default_factory=list)
    post_model: list[PostModelHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    finalize: list[FinalizeHook] = field(default_factory=list)

    def names(self) -> tuple[str, ...]:
        """Canonical ordering: bail → post_model → pre_tool →
        post_tool → finalize. The attribution eval enumerates these
        to disable one catcher at a time; stable order keeps
        diagnostic output reproducible."""
        out: list[str] = []
        for phase in (
            self.bail,
            self.post_model,
            self.pre_tool,
            self.post_tool,
            self.finalize,
        ):
            out.extend(h.name for h in phase)
        return tuple(out)

    def run_bail(self, ctx: BailContext, *, disabled: frozenset[str]) -> BailOutcome:
        for hook in self.bail:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if isinstance(outcome, Nudge) and not outcome.catcher:
                # Attach the hook's name so downstream observers (CLI
                # retry marker, tool-loop events, attribution eval)
                # can report WHICH rule fired without re-inspecting
                # the nudge text.
                return Nudge(text=outcome.text, catcher=hook.name)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_post_model(
        self, ctx: PostModelContext, *, disabled: frozenset[str]
    ) -> PostModelOutcome:
        for hook in self.post_model:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_pre_tool(self, ctx: PreToolContext, *, disabled: frozenset[str]) -> PreToolOutcome:
        for hook in self.pre_tool:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_post_tool(self, ctx: PostToolContext, *, disabled: frozenset[str]) -> PostToolOutcome:
        for hook in self.post_tool:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()

    def run_finalize(self, ctx: FinalizeContext, *, disabled: frozenset[str]) -> FinalizeOutcome:
        for hook in self.finalize:
            if hook.name in disabled:
                continue
            outcome = hook.check(ctx)
            if not isinstance(outcome, Continue):
                return outcome
        return Continue()


def default_hook_pipeline() -> HookPipeline:
    """Build the shipping pipeline. Order mirrors the pre-refactor
    `_diagnose_bail` branch order so first-match semantics stay
    identical — swapping two hooks could change which nudge text the
    user sees for a reply that trips both.

    `post_tool` ships empty by default: the only hook that currently
    targets this phase is `ToolResultSummarizerHook`, which requires
    a summarizer adapter and is registered opt-in by the CLI when
    --summarize-tool-results is set."""
    return HookPipeline(
        bail=[
            TruncatedHook(),
            UnparseableHook(),
            TeaserHook(),
            FalseSuccessHook(),
            MetaConfirmHook(),
            FabricatedSearchHook(),
            FabricatedItemizationHook(),
            AbFabricationHook(),
            ToolIntentHook(),
            # Compliance check runs last — fabrication-shape catchers
            # above all get first pass at a malformed reply. Only a
            # reply that survived every fabrication gate gets asked
            # the compliance question 'did you cite your source?'.
            MissingCitationHook(),
            # Internal-consistency check: the reply's own count claim
            # vs. its enumerated list. Runs after missing_citation so
            # a reply that ADDS a citation on retry doesn't get
            # re-chained into a count-mismatch nudge from its original
            # pre-citation form.
            ListCountMismatchHook(),
            # Domain-safety check: controllers must not assign the
            # pilot-initiated reserved transponder codes (7500/7600/
            # 7700) in routine phraseology. Runs last because a reply
            # that fails any earlier gate should get the shape-specific
            # nudge first; only a reply otherwise structurally fine
            # but proposing an unsafe code reaches this.
            ReservedSquawkCodeHook(),
            # Scope check: user's question has no aviation vocabulary,
            # but the reply is talking ATC. Catches context-bleed and
            # out-of-scope fabrication ('do roosters lay eggs' getting
            # answered with phraseology content).
            ScopeRedirectHook(),
            # Ambiguity check: user asked about a term whose JO handling
            # depends on an unspecified variant ('balloon' = manned vs.
            # unmanned free), and the reply silently picked a variant.
            # Placed last — ambiguity is relevant only when the prompt
            # is otherwise in-scope.
            AmbiguousContextHook(),
        ],
        post_model=[PairedMetaConfirmStripHook()],
        # Order matters inside pre_tool: duplicate_call fires first so
        # a repeat call short-circuits before the grounding check
        # spends cycles analyzing it (and so the user sees the
        # "duplicate — skipped" nudge, not a grounding complaint,
        # when both would fire).
        pre_tool=[DuplicateCallHook(), ArgumentGroundingHook()],
        post_tool=[],
        # Order matters inside finalize: ungrounded_citation runs first so
        # it can Halt with a scope-aware refusal for fabricated-citation
        # replies whose bail outcome was Continue (legitimate-looking
        # wrap-up that no bail hook caught). table_fabrication then
        # covers the narrower "grounded citation, fabricated row" shape
        # when the reply is itself a pipe table. numeric_fabrication
        # covers the same failure mode when the reply is prose
        # ('MH class is 50 miles' vs. the retrieved `|MH|…|25|`).
        # fabrication_fallback only fires when the bail loop itself left
        # a Nudge outcome on the table.
        finalize=[
            # Runs first: when a grounding tool ran but scored weakly
            # AND the reply cites a section the tool didn't ground,
            # halt with a low-confidence fallback. Complementary to
            # UngroundedCitationHook — this catches the 'tool ran but
            # I made up the §' shape; that catches the 'no tool ran at
            # all' shape.
            LowConfidenceFallbackHook(),
            UngroundedCitationHook(),
            TableFabricationHook(),
            NumericFabricationHook(),
            FabricationFallbackHook(),
        ],
    )


# ---------- post-tool summarizer ----------


_SUMMARIZER_SYSTEM_PROMPT = (
    "You compress tool output. Return a concise summary (<= 200 words) "
    "that preserves EVERY file path, line number, identifier, URL, "
    "error message, and proper noun verbatim. Do not add commentary, "
    "recommendations, or lists of next steps. Do not wrap the output "
    "in markdown fences. Do not start with 'Summary:' or similar "
    "preambles. Just the compressed content."
)

_SUMMARIZER_USER_TEMPLATE = (
    "Tool {tool_name!r} was called with arguments: {arguments}\n\n"
    "The raw output was:\n\n"
    "{output}\n\n"
    "Compress it per the rules above."
)


class _SummarizerAdapter(Protocol):
    """Narrow structural type for the summarizer. Only needs
    `complete` — any ModelAdapter satisfies this, as does the router's
    adapter. Reduces the coupling surface from hooks.py into the
    adapter hierarchy."""

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


@dataclass
class ToolResultSummarizerHook:
    """Compress high-noise tool outputs before they're appended to the
    model-visible message thread.

    Fires only when BOTH of:
    - `ctx.spec.high_noise` is True — the tool is flagged as prone to
      dumping bulk (grep / list_dir / search_web / etc).
    - `len(result.output) > threshold_chars` — there's enough text to
      be worth the summarization round-trip.

    Summarization failures (adapter raises, empty response) fall
    through as Continue — the original result reaches the model
    untouched, so context drift is the worst case, never a broken
    turn."""

    summarizer: _SummarizerAdapter
    threshold_chars: int = 1024
    max_summary_tokens: int = 256
    temperature: float = 0.0  # deterministic compression
    name: str = "tool_result_summarizer"

    def check(self, ctx: PostToolContext) -> PostToolOutcome:
        if not ctx.spec.high_noise:
            return Continue()
        if not ctx.result.success:
            # Errors are already small AND load-bearing — the model
            # needs the exact error text to recover. Never summarize.
            return Continue()
        if len(ctx.result.output) <= self.threshold_chars:
            return Continue()
        prompt = _SUMMARIZER_USER_TEMPLATE.format(
            tool_name=ctx.call.name,
            arguments=ctx.call.arguments,
            output=ctx.result.output,
        )
        messages = [
            ChatMessage(role="system", content=_SUMMARIZER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
        try:
            summary = self.summarizer.complete(
                messages,
                max_tokens=self.max_summary_tokens,
                temperature=self.temperature,
            )
        except Exception:
            # Never let a summarization failure break the turn.
            return Continue()
        summary = summary.strip()
        if not summary:
            return Continue()
        # Tag the output so the main model knows what it's looking at
        # and can't accidentally quote it back as literal tool output
        # (fabrication catchers would flag a quoted summary as
        # fabricated-search, for example). Preserve success + name.
        annotated = f"[tool output summarized from {len(ctx.result.output)} chars]\n{summary}"
        return ReplaceResult(
            ToolResult(
                tool_name=ctx.result.tool_name,
                output=annotated,
                success=ctx.result.success,
                error=ctx.result.error,
            )
        )


def _call_key(call: ToolCall) -> tuple[str, str]:
    """Canonical (name, arguments-json) key for duplicate detection.
    Sorting keys means argument order doesn't create false-positive
    uniqueness ({'a':1,'b':2} == {'b':2,'a':1}); default=str keeps the
    key stable if a model emits exotic-but-JSON-stringifiable types
    (dates, Paths). We never decode the key back — only equality
    matters — so lossy coercion is fine."""
    import json

    return (call.name, json.dumps(call.arguments, sort_keys=True, default=str))


__all__ = [
    "AB_DATE_HEADER_RE",
    "AB_TIER_HEADER_RE",
    "ARG_DOMAIN_RE",
    "BARE_CLAIM_RE",
    "DUPLICATE_CALL_NUDGE",
    "EXHAUSTED_FABRICATION_FALLBACK",
    "FABRICATED_AB_CAPTURE_RE",
    "FABRICATED_AB_SCOPE_RE",
    "FABRICATED_BEAD_ID_RE",
    "FABRICATED_ITEMIZATION_RE",
    "FABRICATED_REMEMBER_RE",
    "FABRICATED_SEARCH_RE",
    "FALSE_SUCCESS_RE",
    "META_CONFIRM_RE",
    "TABLE_FABRICATION_FALLBACK",
    "TEASER_RE",
    "TOOL_INTENT_RE",
    "UNGROUNDED_CITATION_FALLBACK",
    "UNGROUNDED_SECTION_CITATION_RE",
    "AbFabricationHook",
    "AmbiguousContextHook",
    "ArgumentGroundingHook",
    "BailContext",
    "BailHook",
    "BailOutcome",
    "Continue",
    "DuplicateCallHook",
    "FabricatedItemizationHook",
    "FabricatedSearchHook",
    "FabricationFallbackHook",
    "FalseSuccessHook",
    "FinalizeContext",
    "FinalizeHook",
    "FinalizeOutcome",
    "Halt",
    "HookPipeline",
    "ListCountMismatchHook",
    "MetaConfirmHook",
    "MissingCitationHook",
    "Nudge",
    "NumericFabricationHook",
    "PairedMetaConfirmStripHook",
    "PostModelContext",
    "PostModelHook",
    "PostModelOutcome",
    "PostToolContext",
    "PostToolHook",
    "PostToolOutcome",
    "PreToolContext",
    "PreToolHook",
    "PreToolOutcome",
    "Replace",
    "ReplaceResult",
    "ReservedSquawkCodeHook",
    "ScopeRedirectHook",
    "Skip",
    "TableFabricationHook",
    "TeaserHook",
    "ToolIntentHook",
    "ToolResultSummarizerHook",
    "Truncated",
    "TruncatedHook",
    "UngroundedCitationHook",
    "UnparseableHook",
    "default_hook_pipeline",
    "looks_like_ab_fabrication",
]
