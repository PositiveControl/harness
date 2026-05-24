#!/usr/bin/env python3
"""Diagnose vLLM tool-calling end-to-end.

The drive-halt cascade 2026-05-23 (loop runs ed58231d / 8eedfd02 /
bbd29017 / 9ab1c245) all showed the same shape: vLLM/Qwen2.5-Coder 32B
either emits prose preamble with NO tool-call shape (no <tool_call>,
no <function=, no bare JSON) or goes silent within 1-5 tokens.
tool_bail.jsonl confirms: `has_open_tag=false / has_close_tag=false /
has_function_open=false` even when content is generated.

That's the symptom of tools not reaching the model — either vLLM isn't
injecting the tool definitions into the prompt via the chat template,
or the model isn't trained to emit them in the expected format. This
script isolates the layer by sending requests DIRECTLY to vLLM,
bypassing every layer of the harness. If vLLM returns structured
`tool_calls` here, the bug is in the harness's prompt stack. If vLLM
returns prose-only / empty, the bug is in vLLM's config (likely the
`--tool-call-parser` setting against Qwen2.5-Coder).

Usage:
    uv run python scripts/vllm_diagnose_tools.py
    uv run python scripts/vllm_diagnose_tools.py --url http://gx10-5fb9:8000/v1
    uv run python scripts/vllm_diagnose_tools.py --model Qwen2.5-Coder-32B-Instruct-AWQ

Runs six probes against the endpoint:

  1. /v1/models — confirms the server is up + shows what model is loaded.
  2. No tools, simple prompt — proves baseline generation works.
  3. With tools, tool_choice="auto" — the harness's actual code path.
  4. With tools, tool_choice="required" — forces tool-call output;
     if the model CAN emit a tool call but the prompt isn't asking
     strongly enough, this will surface it.
  5. With tools + heavy persona/system stack — checks whether system
     prompt overload tips the model toward narration.
  6. write_file + multi-line content (harness-9jt2) — reproduces the
     loop-run 86699d81 turn 6 failure where tool calls came back with
     `arguments={}`. Dissects every parsed tool_call and reports which
     keys were extracted, which were dropped, and how long each value
     is. The diagnostic answer for whether qwen3_xml is dropping
     multi-line <parameter=content> values.
  7. Same as 6, but streaming. The harness uses streaming in production;
     vLLM's qwen3_xml parser may behave differently when accumulating
     `arguments` from chunk deltas vs the non-streaming path.

Each probe prints the request body (so you can see what was sent) and
the response body (so you can see what came back). A verdict line at
the end of each probe summarizes the outcome:

  ✓ structured tool_calls returned  — vLLM extracted the call cleanly
  ⚠ content has tool-call shape     — vLLM didn't extract, but the
                                       model emitted a parseable shape
                                       (the harness's fallback parsers
                                       would catch this in production)
  ✗ no tool_calls anywhere          — the model isn't emitting tool
                                       calls; check `--tool-call-parser`
                                       config on the vLLM startup
  ✗ empty content                   — the model is producing nothing
                                       (likely immediate stop-token hit)

Exit code: 0 if all probes succeeded (regardless of verdict — the
script's job is to surface signal, not to grade it).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Any

_DEFAULT_URL = "http://gx10-5fb9:8000/v1"
_DEFAULT_TIMEOUT_S = 60


# Minimal tool spec — list_dir is read-only, has a tiny schema, and
# any drive-loop turn that needs to inspect a workspace would call it
# first. If the model can't call this in response to "list the files
# in the current directory," it can't call anything.
_LIST_DIR_TOOL = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": (
            "List the files and directories under the given path. Returns "
            "names only; does not recurse."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Relative to workspace root.",
                }
            },
            "required": ["path"],
        },
    },
}


# write_file shape — mirrors src/harness/tools/write_file.py exactly.
# Two required params, the second of which carries multi-line content.
# This is the call shape that came back with args={} during loop run
# 86699d81 turn 6 (vjb6 attempt 2). If vLLM extracts both `path` and
# `content` here, the parser isn't the culprit; if `content` is empty
# or `arguments` is `{}`, the qwen3_xml parser is dropping multi-line
# parameter values.
_WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write the entire file at `path` with the given `content`. "
            "Overwrites any existing file. Both arguments are required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file contents to write (may be multi-line).",
                },
            },
            "required": ["path", "content"],
        },
    },
}


# The actual payload from loop run 86699d81 (vjb6 §2 World map). A 30-row
# by 40-char tile grid string with the §2.2 alphabet. This is the kind
# of content the model needs to embed in a write_file call. If the
# qwen3_xml parser drops multi-line param values, this is what gets
# eaten. Format: each row is one string entry in a JS array literal.
_TILE_GRID_JS = """// game.js — tile grid (§2 World map)
const MAP_WIDTH = 40;
const MAP_HEIGHT = 30;
const tileGrid = [
  "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
  "B......................................B",
  "B.====================.================B",
  "B.=B................=.=B..............=B",
  "B.=B.BBBB.BBBB.BBBB.=.=B.BBBB.BBBB.BBB=B",
  "B.=B.B..B.B..B.B..B.=.=B.B..B.B..B.B..=B",
  "B.=B.BBBB.BBBB.BBBB.=.=B.BBBB.BBBB.BBB=B",
  "B.=B................=.=B..............=B",
  "B.====================.================B",
  "B......................................B",
  "B.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|B",
  "B.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|B",
  "B......................................B",
  "B.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-B",
  "B......................................B",
  "B.====================.================B",
  "B.=B................=.=B..............=B",
  "B.=B.BBBB.BBBB.BBBB.=.=B.BBBB.BBBB.BBB=B",
  "B.=B.B..B.B..B.B..B.=.=B.B..B.B..B.B..=B",
  "B.=B.BBBB.BBBB.BBBB.=.=B.BBBB.BBBB.BBB=B",
  "B.=B................=.=B..............=B",
  "B.====================.================B",
  "B......................................B",
  "B.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|B",
  "B.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|B",
  "B......................................B",
  "B.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-B",
  "B......................................B",
  "B......................................B",
  "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
];
"""


# Same shape (Qwen-style) we'd look for in the raw content as the
# harness fallback parsers do. If the model emitted text with one of
# these tags, vLLM didn't extract the structured call, but the harness
# would still recover it via _parse_qwen_tool_calls.
_TOOL_CALL_SHAPE_RE = re.compile(
    r"<tool_call>|<tools>\s*\{|<function=|^\s*\{[\s\S]*\"name\"[\s\S]*\"arguments\"",
    re.MULTILINE,
)


def _post_json(
    url: str, body: dict[str, Any], *, timeout: int = _DEFAULT_TIMEOUT_S
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 — operator-supplied http(s) URL
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        result: dict[str, Any] = json.loads(resp.read())
    return result


def _get_json(url: str, *, timeout: int = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        result: dict[str, Any] = json.loads(resp.read())
    return result


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_kv(key: str, value: object) -> None:
    print(f"  {key:<24} {value}")


def _print_body(label: str, body: object) -> None:
    print(f"  --- {label} ---")
    rendered = json.dumps(body, indent=2)
    # Cap each line to 200 chars so a fat schema doesn't blow the terminal.
    for line in rendered.splitlines():
        print(f"  {line[:200]}")


def _verdict_for(response: dict[str, Any]) -> str:
    """Render a one-line verdict from a /chat/completions response."""
    choices = response.get("choices") or []
    if not choices:
        return "✗ no choices in response"
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    finish_reason = choices[0].get("finish_reason")

    if tool_calls:
        names = [(tc.get("function") or {}).get("name") or "<?>" for tc in tool_calls]
        return f"✓ structured tool_calls returned: {names} (finish={finish_reason})"
    if not content.strip():
        return f"✗ empty content, no tool_calls (finish={finish_reason})"
    if _TOOL_CALL_SHAPE_RE.search(content):
        return (
            f"⚠ no structured tool_calls, but content has tool-call shape "
            f"(finish={finish_reason}) — fallback parsers would catch this"
        )
    return f"✗ no tool_calls anywhere; prose only (finish={finish_reason})"


def _dissect_tool_calls(response: dict[str, Any], expected_keys: tuple[str, ...]) -> None:
    """Walk every parsed tool_call in the response and report on its
    arguments shape. Used by the heavy-content probe to surface the
    empty-args case: vLLM returns a tool_calls entry whose arguments
    parses to {} or is missing required keys.

    For each tool_call:
      - raw arguments string (head + tail, capped at 300 chars each)
      - parsed arguments dict (success/failure of json.loads)
      - keys present vs `expected_keys`
      - per-key value lengths (so we can see if `content` came back
        as empty string vs absent vs truncated)
    """
    choices = response.get("choices") or []
    if not choices:
        print("  (no choices to dissect)")
        return
    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        print("  (no tool_calls in message — verdict line covers this case)")
        return
    print(f"  --- tool_calls dissection ({len(tool_calls)} call(s)) ---")
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        name = fn.get("name")
        args_raw = fn.get("arguments")
        print(f"  [{i}] name={name!r}")
        print(f"      arguments type: {type(args_raw).__name__}")
        if isinstance(args_raw, str):
            print(f"      arguments len: {len(args_raw)}")
            head = args_raw[:300].replace("\n", "\\n")
            tail = args_raw[-300:].replace("\n", "\\n") if len(args_raw) > 300 else ""
            print(f"      arguments head: {head!r}")
            if tail:
                print(f"      arguments tail: {tail!r}")
            try:
                parsed = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError as exc:
                print(f"      ✗ json.loads FAILED: {exc}")
                continue
        elif isinstance(args_raw, dict):
            parsed = args_raw
        else:
            print(f"      ✗ arguments has unexpected type ({type(args_raw).__name__})")
            continue
        if not isinstance(parsed, dict):
            print(f"      ✗ parsed arguments is not a dict (got {type(parsed).__name__})")
            continue
        keys_present = sorted(parsed.keys())
        keys_missing = [k for k in expected_keys if k not in parsed]
        print(f"      keys present: {keys_present}")
        if keys_missing:
            print(f"      ✗ keys MISSING: {keys_missing}")
        for k in expected_keys:
            if k not in parsed:
                continue
            v = parsed[k]
            if isinstance(v, str):
                print(f"      {k!r}: str(len={len(v)}) head={v[:120]!r}")
            else:
                print(f"      {k!r}: {type(v).__name__} = {v!r}")
        if not keys_missing and all(
            isinstance(parsed.get(k), str) and parsed[k] for k in expected_keys
        ):
            print("      ✓ all required keys present and non-empty")


def probe_models(url: str) -> str | None:
    """Probe 1: GET /models. Confirms server is up + returns the model id."""
    _print_section("PROBE 1: GET /models — server up + loaded model")
    try:
        data = _get_json(f"{url}/models")
    except urllib.error.URLError as exc:
        print(f"  ✗ server unreachable at {url}: {exc}")
        return None
    models = [m.get("id") for m in (data.get("data") or [])]
    _print_kv("models loaded", models)
    if not models:
        print("  ✗ no models found")
        return None
    print(f"  ✓ server up at {url}")
    return str(models[0])


def probe_no_tools(url: str, model: str) -> None:
    """Probe 2: chat completion with NO tools.

    Baseline. If this returns empty, something's wrong before we even
    get to tool calling — model load issue, chat-template mismatch,
    immediate stop-token hit. If this returns prose, the model can
    generate; the question becomes whether tools reach it."""
    _print_section("PROBE 2: chat completion, NO tools (baseline generation)")
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say hello in exactly one sentence."},
        ],
        "temperature": 0.5,
        "max_tokens": 64,
    }
    _print_body("request", body)
    try:
        response = _post_json(f"{url}/chat/completions", body)
    except urllib.error.HTTPError as exc:
        print(f"  ✗ HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}")
        return
    _print_body("response", response)
    choices = response.get("choices") or []
    msg = (choices[0].get("message") if choices else {}) or {}
    content = msg.get("content") or ""
    finish_reason = choices[0].get("finish_reason") if choices else None
    if content.strip():
        print(f"  ✓ baseline generation works (finish={finish_reason})")
    else:
        print(f"  ✗ EMPTY content with no tools sent (finish={finish_reason})")
        print("    └─ baseline generation is broken — model load or chat-template issue.")


def probe_tools_auto(url: str, model: str) -> None:
    """Probe 3: chat completion WITH tools, tool_choice='auto' (default).

    Mirrors the harness's actual code path. If vLLM extracts the call
    as structured tool_calls, the model + parser work as expected. If
    the content has tool-call shape but tool_calls is empty, the
    parser config (`--tool-call-parser`) is wrong for this model. If
    there's no tool-call shape anywhere, tools didn't reach the model
    via the chat template."""
    _print_section("PROBE 3: chat completion + tools, tool_choice='auto'")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You have file-system tools. Use them when the user's "
                    "request requires file-system information."
                ),
            },
            {
                "role": "user",
                "content": (
                    "List the files in the current workspace directory. "
                    "Use the list_dir tool to do this; do not guess."
                ),
            },
        ],
        "tools": [_LIST_DIR_TOOL],
        "tool_choice": "auto",
        "temperature": 0.5,
        "max_tokens": 256,
    }
    _print_body("request", body)
    try:
        response = _post_json(f"{url}/chat/completions", body)
    except urllib.error.HTTPError as exc:
        print(f"  ✗ HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}")
        return
    _print_body("response", response)
    print(f"  VERDICT: {_verdict_for(response)}")


def probe_tools_with_heavy_prompt(url: str, model: str) -> None:
    """Probe 5 (harness-d6ak followup): chat completion + tools +
    a HEAVY system prompt stack mirroring the drive loop's actual
    prompt shape.

    Drives ed58231d / 8eedfd02 / bbd29017 showed the model emitting
    prose preamble with NO tool-call shape, but a minimal-prompt
    probe (probe 3 above) showed the same model emitting valid XML
    tool calls. This probe layers in the kind of bulky system
    content the drive sends — a persona-shaped block + a tool-use
    rules block — to confirm prompt overload tips the model toward
    narration. If THIS probe also returns ⚠ tool-call-shape, the
    parser-only fix (harness-yez7) is enough; if it falls back to
    prose, the system-prompt overload is the secondary bug
    (harness-d6ak)."""
    _print_section("PROBE 5: chat completion + tools + heavy system stack")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tool-use rules. (1) When the user's request contains a "
                    "synthesis verb (rank, prioritize, compare, summarize, "
                    "group, score, categorize, order, sort, filter), do not "
                    "terminate after the data-gathering tool call. The tool "
                    "result is input, not output — continue the turn and "
                    "produce the requested synthesis. (2) When the user's "
                    "request contains multiple distinct asks, each sub-ask "
                    "gets its own tool budget. (3) Plan-then-execute: when "
                    "asked for multi-step work, open with a short numbered "
                    "plan of the concrete tool-call steps, then walk it."
                ),
            },
            {
                "role": "system",
                "content": (
                    "You are airton. Pronoun: he/him. Era: present day.\n\n"
                    "Premise: you are a software engineer doing real work for "
                    "Mark.\n\n"
                    "Self-awareness: you are software. Never hide that.\n\n"
                    "Values: be specific. Cite sources. Refuse to pad.\n\n"
                    "Directives: produce concrete output. Use tools when "
                    "available. Do not invent acceptance criteria.\n\n"
                    "Constitution: a one-page document the user wrote about "
                    "who you are and what you do not do.\n\n"
                    "How you speak:\n"
                    "  - Prose by default, not bullets.\n"
                    "  - 1-4 sentences is the usual length.\n"
                    "  - When you don't know, say \"Don't know\" plainly.\n\n"
                    "Session handoff:\n"
                    "  bd-id: harness-90j0\n"
                    "  title: §1 Project shape — index.html + game.js skeleton\n"
                    "  description: Create index.html that references game.js. "
                    "Skeleton only. Defer real game logic to §2-§15.\n"
                    "  acceptance: index.html exists with a <canvas> tag and "
                    "loads game.js via <script>. game.js exists with a "
                    "requestAnimationFrame loop stub.\n\n"
                    "Phase instructions (ASSESS):\n"
                    "Read the file(s) referenced in the session handoff, "
                    "then call submit_assessment with current_state, gap, "
                    "and approach."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Drive the bd issue described in the session handoff to "
                    "closure. Run `bd close <issue-id>` via shell when the "
                    "acceptance criteria are met. Do not invent acceptance "
                    "criteria the issue doesn't list."
                ),
            },
        ],
        "tools": [_LIST_DIR_TOOL],
        "tool_choice": "auto",
        "temperature": 0.5,
        "max_tokens": 256,
    }
    _print_body("request", body)
    try:
        response = _post_json(f"{url}/chat/completions", body)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        print(f"  ✗ HTTP {exc.code}: {body_text[:400]}")
        return
    _print_body("response", response)
    print(f"  VERDICT: {_verdict_for(response)}")


def probe_tools_heavy_content(url: str, model: str) -> None:
    """Probe 6 (harness-9jt2): write_file tool call with multi-line
    content. Reproduces the loop-run 86699d81 failure mode.

    Loop run 86699d81 turn 6 (vjb6 §2 World map) showed write_file
    and edit_file tool calls coming back with `arguments={}` despite
    the issue requiring a 30-row x 40-char tile grid string in the
    file content. Hypothesis: vLLM's --tool-call-parser qwen3_xml
    drops multi-line <parameter=content>...</parameter> values when
    the value contains newlines, JS punctuation, or characters that
    look like XML delimiters.

    This probe sends a request that should produce a single
    write_file tool call carrying a path AND a multi-line content
    string. We then dissect the returned tool_calls to see whether
    `arguments` is empty (parser-dropping multi-line content),
    has only `path` (parser dropping the second param), or carries
    both keys with full content (parser working — bug is elsewhere).
    """
    _print_section("PROBE 6: write_file + multi-line content (qwen3_xml repro)")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write files via the write_file tool. Always pass both "
                    "`path` and `content` as parameters. `content` may contain "
                    "newlines, quotes, and JS code — emit it verbatim."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write the following JavaScript file at path 'game.js'. "
                    "Use the write_file tool. Pass the path and the FULL "
                    "content verbatim — do not summarize or truncate.\n\n"
                    "```javascript\n"
                    f"{_TILE_GRID_JS}"
                    "```"
                ),
            },
        ],
        "tools": [_WRITE_FILE_TOOL],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    _print_body("request", body)
    try:
        response = _post_json(f"{url}/chat/completions", body)
    except urllib.error.HTTPError as exc:
        print(f"  ✗ HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}")
        return
    _print_body("response", response)
    print(f"  VERDICT: {_verdict_for(response)}")
    _dissect_tool_calls(response, expected_keys=("path", "content"))


def probe_tools_heavy_content_multitool(url: str, model: str) -> None:
    """Probe 8 (harness-9jt2): write_file + multi-line content, but
    with the driver's full 13-tool roster in the request — not just
    write_file. If having many sibling tools in the schema changes
    how vLLM parses the model's output, this probe will surface it.

    Drives ship the executor with read_file, list_dir, grep, glob,
    edit_file, write_file, stream_edit, python_stream, shell,
    git_status, git_diff, git_log, fetch_url (and a few more). 13+
    tools = ~5-7k tokens of schema overhead. The probe 6 baseline
    runs with one tool to isolate the multi-line-content question;
    probe 8 layers tool-count on top to confirm or rule out
    schema-size interference."""
    _print_section("PROBE 8: write_file + multi-line content + 13-tool roster")

    def _stub_tool(name: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Driver-roster stub: {name}.",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": list(params.keys()),
                },
            },
        }

    roster = [
        _WRITE_FILE_TOOL,
        _stub_tool("read_file", {"path": {"type": "string"}}),
        _stub_tool("list_dir", {"path": {"type": "string"}}),
        _stub_tool(
            "grep",
            {"pattern": {"type": "string"}, "path": {"type": "string"}},
        ),
        _stub_tool("glob", {"pattern": {"type": "string"}}),
        _stub_tool(
            "edit_file",
            {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
        ),
        _stub_tool(
            "stream_edit",
            {"path": {"type": "string"}, "expr": {"type": "string"}},
        ),
        _stub_tool(
            "python_stream",
            {"expr": {"type": "string"}, "paths": {"type": "array"}},
        ),
        _stub_tool("shell", {"cmd": {"type": "string"}}),
        _stub_tool("git_status", {}),
        _stub_tool("git_diff", {}),
        _stub_tool("git_log", {"n": {"type": "integer"}}),
        _stub_tool("fetch_url", {"url": {"type": "string"}}),
    ]
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You have a coding tool roster. Use write_file when you "
                    "need to overwrite a file with new contents. Pass both "
                    "`path` and `content` parameters. `content` may span many "
                    "lines and contain code."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write 'game.js' with this content verbatim — use the "
                    "write_file tool, do not summarize:\n\n"
                    "```javascript\n"
                    f"{_TILE_GRID_JS}"
                    "```"
                ),
            },
        ],
        "tools": roster,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    # Skip the request body dump — 13 tools is too much to print usefully.
    print(f"  request: model={model}, tools={len(roster)}, multi-line content payload")
    try:
        response = _post_json(f"{url}/chat/completions", body)
    except urllib.error.HTTPError as exc:
        print(f"  ✗ HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}")
        return
    _print_body("response", response)
    print(f"  VERDICT: {_verdict_for(response)}")
    _dissect_tool_calls(response, expected_keys=("path", "content"))


def probe_tools_heavy_content_streaming(url: str, model: str) -> None:
    """Probe 7 (harness-9jt2): same as probe 6 but with stream=true.

    The harness uses streaming in production. vLLM accumulates
    `arguments` as a sequence of delta chunks during streaming;
    if the qwen3_xml parser's incremental state machine drops
    chunks differently than the non-streaming path, this probe
    will surface the divergence.

    We reassemble the full message ourselves from SSE chunks and
    feed it through the same dissector probe 6 uses."""
    _print_section("PROBE 7: write_file + multi-line content, STREAMING")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write files via the write_file tool. Always pass "
                    "both `path` and `content`. Multi-line content is fine."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write 'game.js' with this content verbatim — use the "
                    "write_file tool, do not summarize:\n\n"
                    "```javascript\n"
                    f"{_TILE_GRID_JS}"
                    "```"
                ),
            },
        ],
        "tools": [_WRITE_FILE_TOOL],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 2048,
        "stream": True,
    }
    _print_body("request", body)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 — operator-supplied http(s) URL
        f"{url}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Accumulate streaming deltas into a single message dict so the
    # dissector can read it the same way it reads a non-streaming
    # response.
    acc_content_parts: list[str] = []
    tc_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    chunk_count = 0
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT_S) as resp:  # noqa: S310
            for line_bytes in resp:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                chunk_count += 1
                for choice in chunk.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        acc_content_parts.append(delta["content"])
                    tc_deltas = delta.get("tool_calls") or []
                    for tcd in tc_deltas:
                        if not isinstance(tcd, dict):
                            continue
                        idx = int(tcd.get("index", 0) or 0)
                        slot = tc_acc.setdefault(
                            idx,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if isinstance(tcd.get("id"), str) and tcd["id"]:
                            slot["id"] = tcd["id"]
                        fn = tcd.get("function") or {}
                        if isinstance(fn.get("name"), str) and fn["name"]:
                            slot["function"]["name"] = fn["name"]
                        if isinstance(fn.get("arguments"), str):
                            slot["function"]["arguments"] += fn["arguments"]
                    fr = choice.get("finish_reason")
                    if isinstance(fr, str):
                        finish_reason = fr
    except urllib.error.HTTPError as exc:
        print(f"  ✗ HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}")
        return
    full_content = "".join(acc_content_parts)
    parsed_calls = [tc_acc[k] for k in sorted(tc_acc.keys())]
    reassembled = {
        "choices": [
            {
                "message": {
                    "content": full_content,
                    "tool_calls": parsed_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    print(f"  chunks received: {chunk_count}")
    print(f"  finish_reason:   {finish_reason}")
    print(f"  content length:  {len(full_content)}")
    print(f"  tool_calls:      {len(parsed_calls)}")
    _print_body("reassembled message", reassembled["choices"][0]["message"])
    print(f"  VERDICT: {_verdict_for(reassembled)}")
    _dissect_tool_calls(reassembled, expected_keys=("path", "content"))


def probe_tools_required(url: str, model: str) -> None:
    """Probe 4: chat completion WITH tools, tool_choice='required'.

    Forces the model to emit a tool call. If 'auto' produced nothing
    structured but 'required' does, the model CAN tool-call — the
    issue is prompt strength (model needs to be told more directly).
    If 'required' ALSO fails, the model genuinely can't emit valid
    tool-call format for this configuration."""
    _print_section("PROBE 4: chat completion + tools, tool_choice='required'")
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "List the files in the current workspace directory.",
            },
        ],
        "tools": [_LIST_DIR_TOOL],
        "tool_choice": "required",
        "temperature": 0.5,
        "max_tokens": 256,
    }
    _print_body("request", body)
    try:
        response = _post_json(f"{url}/chat/completions", body)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        print(f"  ✗ HTTP {exc.code}: {body_text[:400]}")
        # vLLM versions that don't support tool_choice='required' return
        # 400 with a clear message — flag it so the operator sees what's
        # wrong rather than blaming the model.
        if exc.code == 400 and "tool_choice" in body_text:
            print(
                "    └─ vLLM rejected tool_choice='required'. Either the "
                "vLLM version pre-dates this option or the parser doesn't "
                "support it. Probe 3's 'auto' verdict is the load-bearing "
                "signal."
            )
        return
    _print_body("response", response)
    print(f"  VERDICT: {_verdict_for(response)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=_DEFAULT_URL,
        help=f"vLLM /v1 endpoint (default: {_DEFAULT_URL})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id to use. If omitted, picks the first model from /v1/models.",
    )
    args = parser.parse_args(argv[1:])

    url = args.url.rstrip("/")
    discovered = probe_models(url)
    if discovered is None:
        return 1
    model = args.model or discovered

    probe_no_tools(url, model)
    probe_tools_auto(url, model)
    probe_tools_required(url, model)
    probe_tools_with_heavy_prompt(url, model)
    probe_tools_heavy_content(url, model)
    probe_tools_heavy_content_streaming(url, model)
    probe_tools_heavy_content_multitool(url, model)

    print()
    print("=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)
    print(
        "\n"
        "  - All probes 2-4 show prose only with no tool-call shape →\n"
        "    vLLM `--tool-call-parser` is misconfigured OR the chat\n"
        "    template isn't injecting tools. Check the vLLM startup\n"
        "    command for `--enable-auto-tool-choice --tool-call-parser X`\n"
        "    where X matches the model family (`hermes` / `qwen` /\n"
        "    `llama3_json` / etc.).\n"
        "\n"
        "  - Probe 2 produces content; probe 3 returns ⚠ (tool-call\n"
        "    shape but unstructured) → the model is emitting tool calls\n"
        "    but vLLM isn't extracting them. The harness's fallback\n"
        "    parsers handle this in production, but the parser config\n"
        "    is still wrong. Probably want `--tool-call-parser qwen`\n"
        "    for Qwen2.5-Coder if the running version supports it.\n"
        "\n"
        "  - Probe 2 produces content; probes 3-4 produce empty →\n"
        "    Something about the tools-bearing request shape breaks the\n"
        "    chat template. Could be vLLM version bug, could be a\n"
        "    malformed tool schema. Try with the openai SDK to confirm.\n"
        "\n"
        "  - All probes 2-4 are ✓ structured → the harness's drive\n"
        "    halts are NOT a vLLM issue. Re-examine the harness's\n"
        "    system prompt stack and message-construction path.\n"
        "\n"
        "  - Probes 6/7 dissection shows `path` present but `content`\n"
        "    missing or empty → qwen3_xml parser is dropping the\n"
        "    multi-line parameter value. Swap to a different\n"
        "    --tool-call-parser (try `hermes`) or upgrade vLLM.\n"
        "\n"
        "  - Probes 6/7 dissection shows BOTH keys present and `content`\n"
        "    matches the asked-for length → parser handles multi-line\n"
        "    content fine; the drive-halt was something model-side\n"
        "    (model abridged the content, model emitted no tool call\n"
        "    on that turn, etc.).\n"
        "\n"
        "  - Probe 6 succeeds but probe 7 returns empty `arguments` →\n"
        "    the streaming-delta accumulation path is the failure mode.\n"
        "    Bypass streaming for write-tool calls, or upgrade vLLM.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
