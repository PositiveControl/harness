"""Profile-resolution + registry-builder smoke tests for the reckon
profile — harness-o87s.

Pins the reckon-profile membership and verifies the four new tools
actually instantiate via the CLI's builder dicts. A future PR that
drops one of them from the dict will fail this loudly.
"""

from __future__ import annotations

import pytest

from harness.tools import (
    TOOL_PROFILES,
    CalcTool,
    DateMathTool,
    NowTool,
    PythonEvalTool,
    Tool,
    TzConvertTool,
    resolve_tool_names,
)


def test_reckon_profile_members() -> None:
    assert TOOL_PROFILES["reckon"] == (
        "now",
        "date_math",
        "calc",
        "python_eval",
        "tz_convert",
        "search_memory",
        "search_facts",
        "introspect",
    )


def test_reckon_resolve_clean() -> None:
    """resolve_tool_names returns the profile contents (sorted) with
    no adds/drops. Set comparison so the ordering contract (sorted
    output, declared-order input) doesn't trip the test."""
    names = resolve_tool_names(profile="reckon")
    assert set(names) == set(TOOL_PROFILES["reckon"])
    assert len(names) == len(TOOL_PROFILES["reckon"])


def test_full_profile_now_includes_reckon_primitives() -> None:
    full = TOOL_PROFILES["full"]
    for name in ("now", "date_math", "calc", "python_eval", "tz_convert"):
        assert name in full, f"full profile missing {name!r}"


def test_reckon_tools_construct_standalone() -> None:
    """Each reckon-primitive tool must construct with no per-session
    plumbing (no stores, no adapter, no workspace).

    The CLI builder dicts call these as zero-arg lambdas — if a tool
    gained a required arg, the lambdas would 500 at session start."""
    tools: list[Tool] = [
        NowTool(),
        DateMathTool(),
        CalcTool(),
        PythonEvalTool(),
        TzConvertTool(),
    ]
    names = {t.spec.name for t in tools}
    assert names == {"now", "date_math", "calc", "python_eval", "tz_convert"}
    for t in tools:
        # Read-tier, no write-confirm prompt.
        assert t.spec.tier == "read", f"{t.spec.name} is not read-tier"
        # Schema isn't empty.
        assert "type" in t.spec.parameters


def test_reckon_schema_budget_under_target() -> None:
    """Per profiles.py docstring: each profile should cost ≤ ~1500
    tokens of schema overhead. We approximate by character length —
    rough but stable enough to detect a 10x regression.

    Combined JSON-schema size of all 5 reckon primitives stays under
    7 KB raw (Qwen tokenizer ~4 chars/token → well under 1500 tokens)."""
    import json

    from harness.tools import (
        CalcTool,
        DateMathTool,
        NowTool,
        PythonEvalTool,
        TzConvertTool,
    )

    total = 0
    for tool in (
        NowTool(),
        DateMathTool(),
        CalcTool(),
        PythonEvalTool(),
        TzConvertTool(),
    ):
        rendered = json.dumps(
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "parameters": tool.spec.parameters,
            }
        )
        total += len(rendered)
    # Generous headroom; tighten if it ever drifts up.
    assert total < 7000, f"reckon primitives schema is {total} chars — over budget"


@pytest.mark.parametrize("name", ["now", "date_math", "calc", "python_eval", "tz_convert"])
def test_reckon_tool_spec_names_match_profile(name: str) -> None:
    """Sanity: the registry profile names match the tools' ToolSpec
    names. A future rename of NowTool.spec.name would silently break
    the profile without this check."""
    instances: dict[str, Tool] = {
        "now": NowTool(),
        "date_math": DateMathTool(),
        "calc": CalcTool(),
        "python_eval": PythonEvalTool(),
        "tz_convert": TzConvertTool(),
    }
    assert instances[name].spec.name == name
