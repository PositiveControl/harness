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
    user-role message and re-runs the round."""

    text: str


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
    a fabricated web-search narration)."""

    reply: ModelReply
    tools_ran_this_turn: bool
    tools_ran: frozenset[str] = frozenset()


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
            r
            for r in rows
            if not _PIPE_SEPARATOR_RE.match(r) and _NUMERIC_TOKEN_RE.search(r)
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
        # that ungrounded_citation leaves alone (grounding tool DID run).
        # fabrication_fallback only fires when the bail loop itself left
        # a Nudge outcome on the table.
        finalize=[
            UngroundedCitationHook(),
            TableFabricationHook(),
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
    "MetaConfirmHook",
    "Nudge",
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
