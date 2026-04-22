"""Model-backed Router: wraps a ModelAdapter (typically a small one —
Hermes-3-Llama-3.2-3B-4bit is the default, function-call-tuned) and
classifies each turn via strict-JSON generation.

The prompt is regenerated per call so the tool list stays fresh:
enabling/disabling tools between turns doesn't require a new Router.
Parse tolerance mirrors `harness.scribe.extractor` (markdown fences,
leading prose, missing keys) — small models routinely add prose even
when told not to, and we'd rather recover than drop the turn."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from harness.model.adapter import ChatMessage
from harness.router.intent import RouterIntent

if TYPE_CHECKING:
    from harness.model.adapter import ModelAdapter
    from harness.tools.base import ToolSpec


_SYSTEM_TEMPLATE = """\
You are a tool router. Given the user's message and a list of available
tools, decide whether one of the tools should be called to answer them,
and if so, which tool and with what arguments.

Return STRICT JSON only — no prose, no markdown fences — in exactly
this shape:
{{"tool": "<tool_name>" or null, "arguments": {{...}}}}

CRITICAL: when the user's message names a specific entity — a domain
(`stackoverflow`, `example.com`), URL, filename, path, subject, or
identifier — your arguments MUST use that exact entity. Copy it
verbatim. Do NOT substitute a domain or identifier from these examples
when the user named a DIFFERENT one. If the user said "stackoverflow",
the URL is stackoverflow.com, not any example below.

A tool is needed ONLY when the answer depends on:
- Live web data (current prices, news, weather, events, real-world entities)
- Specific file contents, directory listings, or filename searches
- Searching inside files for text
- Memory of past conversations or stored facts about people/projects

Return null (no tool) for:
- Greetings, thanks, acknowledgements, casual chat
- Creative writing (haikus, poems, jokes, stories)
- Explanations of general concepts, definitions, how things work
  ("explain what X is", "how does Y work", "what is a Z")
- Opinion or judgment requests ("what do you think", "which is better")
- Arithmetic, logic, riddles, or anything answerable from general knowledge

Only use tools that appear under "Available tools". If the right tool is
not listed, return null.

Available tools:
{tool_list}

Disambiguation for common overlaps:
- read_file: user wants the CONTENTS of a specific named file. A
  concrete filename like `pyproject.toml` or `README.md` is read_file,
  even when the verb is "open" or "find". A URL or bare domain
  (`example.com`, `foo.fm`, `https://...`) is NOT a file — use
  fetch_url. Never put a tool name or query string into `path`.
- list_dir: user wants to see what's IN a directory.
- glob: user wants to FIND files matching a WILDCARD pattern (must
  contain `*`, `?`, or `[...]`). A bare filename is not a glob.
- grep: user wants to SEARCH INSIDE files for a text pattern.
- fetch_url: user names a concrete URL or bare domain and wants its
  contents. "Go to X", "open this page", "fetch this site", "read
  <domain>", or even "tell me about <domain>.<tld>" when <domain> is
  a real host. Anything with a `.com`/`.org`/`.fm`/`.io`/etc. tail is
  a URL, not a file. Prepend `https://` if the user's URL omits the
  scheme.
- search_web: live external data with NO specific URL. "Online", "on
  the web", "latest", "current", "search for X" are search_web even
  when the verb is "find". If the user names a concrete domain to
  visit, use fetch_url instead. Pass ONLY `query` by default — do not
  set `max_results`. Only include `max_results` when the user names a
  specific count ("top 3", "first result", "give me 10"); otherwise
  omit it and let the tool return its default broad set.
- search_memory: ONLY for past conversations between us and specific
  events we experienced together. REQUIRES an explicit memory signal:
  "remember", "recall", "what did we discuss", "last time", "before",
  "earlier". The bare verb "search" on its own is NOT a memory signal.
  If the query is about an external person, place, company, or any
  live-world fact — even when the user says "search for X" — use
  search_web instead.
- search_facts: stored attributes or preferences about a subject the
  user has previously told us about. The word "fact(s)" is a strong
  cue for search_facts over search_memory. External people or entities
  the user has not previously discussed are search_web, not
  search_facts.
- list / plan / status / drift / ready (ab ops tools): for browsing
  questions like "what tasks do we have coming up?", "what's on the
  plate?", "what's in our future?", call the tool with NO arguments.
  bd's defaults already filter to open work. Only fill in `status`,
  `scope`, `priority`, or `issue_type` when the user explicitly names
  one ("list closed items" → status='closed'; "professional
  things" → scope='professional'). NEVER pass status='in_progress'
  unless the user said "in progress", "currently working on",
  "started", or similar — vague time phrasings ("upcoming", "future",
  "next") are NOT in_progress signals.

Examples:

User message: search the web for bbq restaurants in 85048
Output: {{"tool": "search_web", "arguments": {{"query": "bbq restaurants 85048"}}}}

User message: find a good recipe for X online
Output: {{"tool": "search_web", "arguments": {{"query": "good recipe for X"}}}}

User message: search the web for "Jane Doe" and give me a short biography
Output: {{"tool": "search_web", "arguments": {{"query": "Jane Doe biography"}}}}

User message: do a search for brad hintze in phoenix, tell me about this person
Output: {{"tool": "search_web", "arguments": {{"query": "brad hintze phoenix"}}}}

User message: search for acme corp's ceo
Output: {{"tool": "search_web", "arguments": {{"query": "acme corp ceo"}}}}

User message: give me the top 3 articles on claude 4.7 pricing
Output: {{"tool": "search_web", "arguments": {{"query": "claude 4.7 pricing", "max_results": 3}}}}

User message: go to stackoverflow and summarize the first question
Output: {{"tool": "fetch_url", "arguments": {{"url": "https://stackoverflow.com"}}}}

User message: go to dailydrop.fm and tell me about it
Output: {{"tool": "fetch_url", "arguments": {{"url": "https://dailydrop.fm"}}}}

User message: open https://example.com/post and summarize it
Output: {{"tool": "fetch_url", "arguments": {{"url": "https://example.com/post"}}}}

User message: show me the contents of src/main.py
Output: {{"tool": "read_file", "arguments": {{"path": "src/main.py"}}}}

User message: open README.md and tell me what it says
Output: {{"tool": "read_file", "arguments": {{"path": "README.md"}}}}

User message: list the files in src/tests
Output: {{"tool": "list_dir", "arguments": {{"path": "src/tests"}}}}

User message: grep for CONFIG_FLAG in the code
Output: {{"tool": "grep", "arguments": {{"pattern": "CONFIG_FLAG"}}}}

User message: recall what we discussed about retrieval thresholds
Output: {{"tool": "search_memory", "arguments": {{"query": "retrieval thresholds"}}}}

User message: what facts do you have about Mark's workflow
Output: {{"tool": "search_facts", "arguments": {{"query": "Mark's workflow"}}}}

User message: what tasks do we have coming up?
Output: {{"tool": "list", "arguments": {{}}}}

User message: list closed personal items
Output: {{"tool": "list", "arguments": {{"status": "closed", "scope": "personal"}}}}

User message: hey how's it going
Output: {{"tool": null, "arguments": {{}}}}

User message: write a haiku about autumn
Output: {{"tool": null, "arguments": {{}}}}

User message: explain how a hash table works
Output: {{"tool": null, "arguments": {{}}}}"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_STRIP = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def _format_spec(spec: ToolSpec) -> str:
    """Render a ToolSpec for the router system prompt. Each tool gets:

    - `- name(arg: type|enum, ...) — description`
    - optional indented per-arg hints when the JSON schema carries a
      non-trivial `description` (the contracts that say things like
      'scope is categorical, not who-or-when' — without these the
      router invents bad values for ambiguous params, harness-nom)

    Enums are rendered inline (`enum[a|b|c]`) instead of as `string`
    so the router model has an immediate constraint signal. Hints
    are collapsed to their first line so the prompt stays tight —
    the small model loses focus with sprawling schemas."""
    props = spec.parameters.get("properties", {}) or {}
    required = set(spec.parameters.get("required", []) or [])
    args: list[str] = []
    hint_lines: list[str] = []
    if isinstance(props, dict):
        for arg_name, schema in props.items():
            if not isinstance(schema, dict):
                args.append(f"{arg_name}: any?")
                continue
            enum = schema.get("enum")
            if isinstance(enum, list) and enum:
                type_label = f"enum[{'|'.join(str(v) for v in enum)}]"
            else:
                type_label = schema.get("type", "any")
            label = f"{arg_name}: {type_label}"
            if arg_name not in required:
                label += "?"
            args.append(label)
            desc = (schema.get("description") or "").strip()
            if len(desc) > 30:
                first_line = desc.splitlines()[0].strip()
                hint_lines.append(f"  · {arg_name}: {first_line}")
    head = f"- {spec.name}({', '.join(args)}) — {spec.description.strip()}"
    if hint_lines:
        return head + "\n" + "\n".join(hint_lines)
    return head


def _build_system_prompt(tool_specs: Sequence[ToolSpec]) -> str:
    if tool_specs:
        tool_list = "\n".join(_format_spec(s) for s in tool_specs)
    else:
        tool_list = '(no tools available — always return {"tool": null})'
    return _SYSTEM_TEMPLATE.format(tool_list=tool_list)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_STRIP.sub("", text).strip()
    return text


def _extract_json_block(text: str) -> str | None:
    """Regex-fallback extractor — first `{...}` span when the text
    contains a JSON object buried in prose. Only used after a
    whole-text parse has failed; see `parse_router_output` below."""
    match = _JSON_BLOCK.search(text)
    return match.group(0) if match is not None else None


def _coerce_intent(data: Any) -> RouterIntent | None:
    """Validate shape. Missing `arguments` is tolerated (defaults to
    empty dict — small models often drop it when there are no args).
    Wrong types for `tool` or `arguments` are not.

    The string forms `"null"` / `"none"` (case-insensitive) are folded
    to Python None — Hermes-3-3B and other tuned-for-JSON models will
    sometimes emit the literal word instead of real JSON null."""
    if not isinstance(data, dict):
        return None
    tool = data.get("tool", ...)
    if tool is ...:  # missing key entirely — malformed
        return None
    if tool is not None and not isinstance(tool, str):
        return None
    if isinstance(tool, str):
        stripped = tool.strip()
        tool = None if not stripped or stripped.lower() in {"null", "none"} else stripped
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return RouterIntent(tool_name=tool, arguments=dict(arguments))


def parse_router_output(raw: str) -> RouterIntent | None:
    """Parse the router model's raw text into a RouterIntent. Returns
    None on any extraction or validation failure — advisory, never
    raises. Separate from ModelRouter so evals and tests can exercise
    the parser without a live model.

    Strategy: try a full-text JSON parse first (after fence strip) so
    top-level lists `[{"tool": "x"}]` are rejected by the shape check
    rather than quietly unwrapping their first element. Fall back to
    regex-extracting the first `{...}` block only when the whole-text
    parse fails (prose wrapping the JSON)."""
    text = _strip_fences(raw)
    data: Any
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        block = _extract_json_block(text)
        if block is None:
            return None
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            return None
    return _coerce_intent(data)


@dataclass
class ModelRouter:
    """A Router backed by a ModelAdapter. Low temperature by default so
    JSON output is deterministic; `max_tokens` is sized for a one-line
    verdict, not prose."""

    adapter: ModelAdapter
    max_tokens: int = 256
    temperature: float = 0.0

    def classify(
        self,
        user_message: str,
        tool_specs: Sequence[ToolSpec],
    ) -> RouterIntent | None:
        system = ChatMessage(role="system", content=_build_system_prompt(tool_specs))
        user = ChatMessage(role="user", content=user_message)
        try:
            raw = self.adapter.complete(
                [system, user],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception:
            # Advisory contract: never raise out of classify(). Adapter
            # failures (OOM, transport errors) degrade to "router could
            # not decide" and the orchestrator falls through.
            return None
        return parse_router_output(raw)
