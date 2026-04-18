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
  even when the verb is "open" or "find".
- list_dir: user wants to see what's IN a directory.
- glob: user wants to FIND files matching a WILDCARD pattern (must
  contain `*`, `?`, or `[...]`). A bare filename is not a glob.
- grep: user wants to SEARCH INSIDE files for a text pattern.
- search_web: anything about live external data. "Online", "on the
  web", "latest", "current" are search_web even when the verb is
  "find".
- search_memory: past conversations, specific events ("remember",
  "recall", "what did we discuss").
- search_facts: stored attributes or preferences about a subject. The
  word "fact(s)" is a strong cue for search_facts over search_memory.

Examples:

User message: search the web for bbq restaurants in 85048
Output: {{"tool": "search_web", "arguments": {{"query": "bbq restaurants 85048"}}}}

User message: find a good recipe for X online
Output: {{"tool": "search_web", "arguments": {{"query": "good recipe for X"}}}}

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

User message: hey how's it going
Output: {{"tool": null, "arguments": {{}}}}

User message: write a haiku about autumn
Output: {{"tool": null, "arguments": {{}}}}

User message: explain how a hash table works
Output: {{"tool": null, "arguments": {{}}}}"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_STRIP = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def _format_spec(spec: ToolSpec) -> str:
    """Render a ToolSpec as one `- name(args) — description` line for
    the router system prompt. Compact on purpose: the small model
    loses focus with sprawling schemas."""
    props = spec.parameters.get("properties", {}) or {}
    required = set(spec.parameters.get("required", []) or [])
    args: list[str] = []
    if isinstance(props, dict):
        for arg_name, schema in props.items():
            arg_type = schema.get("type", "any") if isinstance(schema, dict) else "any"
            label = f"{arg_name}: {arg_type}"
            if arg_name not in required:
                label += "?"
            args.append(label)
    return f"- {spec.name}({', '.join(args)}) — {spec.description.strip()}"


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
