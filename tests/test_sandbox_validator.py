"""Tests for the tool sandbox validator (harness-l2ak).

Each test compiles a source string in a real subprocess — no mocks.
The fixtures below are intentionally tiny: the validator's job is to
trip on contract violations and import escapes, not to exercise real
business logic. Failure paths run faster than the happy path because
they short-circuit at the bootstrap stage.
"""

from __future__ import annotations

import textwrap

from harness.tools.sandbox_validator import (
    MAX_SOURCE_BYTES,
    ValidationResult,
    validate_tool_module,
)

# --- happy path -----------------------------------------------------------


_VALID_TOOL = textwrap.dedent(
    """
    from dataclasses import dataclass
    from harness.tools.base import ToolSpec


    @dataclass
    class HelloTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="hello",
                description="Say hello to a name.",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                tier="read",
            )

        def call(self, *, name: str) -> str:
            return f"hello, {name}"
    """
)


def test_valid_tool_compiles_and_returns_spec() -> None:
    result = validate_tool_module(source=_VALID_TOOL)
    assert isinstance(result, ValidationResult)
    assert result.ok is True
    assert result.error is None
    assert result.spec is not None
    assert result.spec.name == "hello"
    assert result.spec.tier == "read"
    assert result.spec.parameters["properties"]["name"]["type"] == "string"


def test_valid_tool_with_smoke_call_returns_output() -> None:
    """smoke_args run the call() entry; valid call → smoke_output set."""
    result = validate_tool_module(source=_VALID_TOOL, smoke_args={"name": "world"})
    assert result.ok is True
    assert result.smoke_output == "hello, world"


# --- contract violations -------------------------------------------------


def test_module_without_tool_class_fails() -> None:
    """A module that doesn't expose a Tool fails with a clear reason."""
    source = "x = 42\n"
    result = validate_tool_module(source=source)
    assert result.ok is False
    assert result.spec is None
    assert "no Tool class" in (result.error or "")


def test_module_with_multiple_tool_classes_fails() -> None:
    """Catalog stores one tool per source file; two candidates is ambiguous."""
    source = textwrap.dedent(
        """
        from harness.tools.base import ToolSpec

        class AlphaTool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(name="alpha", description="a", parameters={}, tier="read")
            def call(self) -> str: return "a"

        class BetaTool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(name="beta", description="b", parameters={}, tier="read")
            def call(self) -> str: return "b"
        """
    )
    result = validate_tool_module(source=source)
    assert result.ok is False
    assert "multiple Tool candidates" in (result.error or "")


def test_module_with_missing_spec_field_fails() -> None:
    """ToolSpec is a frozen dataclass; missing-required-arg surfaces at
    instantiation time inside the child."""
    source = textwrap.dedent(
        """
        from harness.tools.base import ToolSpec

        class BrokenTool:
            @property
            def spec(self) -> ToolSpec:
                # Missing `tier` — ToolSpec(...) will raise.
                return ToolSpec(name="x", description="x", parameters={})  # type: ignore[call-arg]
            def call(self) -> str: return "x"
        """
    )
    result = validate_tool_module(source=source)
    assert result.ok is False
    assert "reading .spec" in (result.error or "") or "tier" in (result.error or "")


def test_module_with_bad_tier_fails() -> None:
    source = textwrap.dedent(
        """
        from harness.tools.base import ToolSpec

        class WrongTierTool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(name="x", description="x", parameters={}, tier="admin")
            def call(self) -> str: return "x"
        """
    )
    result = validate_tool_module(source=source)
    assert result.ok is False
    assert "tier" in (result.error or "")


# --- import allowlist + smoke failure -----------------------------------


def test_banned_import_fails_with_clear_reason() -> None:
    """`os` is not on the default allowlist — module load must fail."""
    source = textwrap.dedent(
        """
        import os
        from harness.tools.base import ToolSpec

        class EscapeTool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(name="x", description="x", parameters={}, tier="read")
            def call(self) -> str: return os.getcwd()
        """
    )
    result = validate_tool_module(source=source)
    assert result.ok is False
    assert "blocked by tool sandbox" in (result.error or "") or "module load failed" in (
        result.error or ""
    )


def test_smoke_call_that_raises_marks_failure() -> None:
    source = textwrap.dedent(
        """
        from harness.tools.base import ToolSpec

        class RaiserTool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="raiser",
                    description="x",
                    parameters={"type": "object", "properties": {}},
                    tier="read",
                )
            def call(self) -> str:
                raise RuntimeError("boom")
        """
    )
    result = validate_tool_module(source=source, smoke_args={})
    assert result.ok is False
    # The spec compiled cleanly even though the smoke call failed —
    # informative for synthesis to know the contract was sound.
    assert result.spec is not None
    assert result.spec.name == "raiser"
    assert "boom" in (result.error or "")


# --- size + input guards -------------------------------------------------


def test_empty_source_rejected_before_subprocess() -> None:
    result = validate_tool_module(source="   \n  \n")
    assert result.ok is False
    assert "non-empty" in (result.error or "")


def test_oversized_source_rejected_before_subprocess() -> None:
    """8 KB cap protects the bootstrap from pathological parse times."""
    source = "x = '" + ("a" * (MAX_SOURCE_BYTES + 100)) + "'\n"
    result = validate_tool_module(source=source)
    assert result.ok is False
    assert "exceeds" in (result.error or "")


def test_extra_allowed_import_is_unioned_with_default() -> None:
    """Caller-supplied allowlist extras must add (not replace) the
    defaults — otherwise every synthesised tool would have to repeat
    `harness.tools.base`."""
    source = textwrap.dedent(
        """
        import base64
        from harness.tools.base import ToolSpec

        class B64Tool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="b64",
                    description="x",
                    parameters={
                        "type": "object",
                        "properties": {"s": {"type": "string"}},
                        "required": ["s"],
                    },
                    tier="read",
                )
            def call(self, *, s: str) -> str:
                return base64.b64encode(s.encode()).decode()
        """
    )
    # `base64` isn't on DEFAULT_ALLOWED_IMPORTS, so the caller adds it.
    result = validate_tool_module(
        source=source,
        smoke_args={"s": "hi"},
        allowed_imports=("base64",),
    )
    assert result.ok is True, result.error
    assert result.smoke_output == "aGk="
