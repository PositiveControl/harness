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

Runs four probes against the endpoint:

  1. /v1/models — confirms the server is up + shows what model is loaded.
  2. No tools, simple prompt — proves baseline generation works.
  3. With tools, tool_choice="auto" — the harness's actual code path.
  4. With tools, tool_choice="required" — forces tool-call output;
     if the model CAN emit a tool call but the prompt isn't asking
     strongly enough, this will surface it.

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
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
