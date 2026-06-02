"""Typed tool-argument validation (harness-5cjj9).

Tools may declare a pydantic `args_model`; the registry validates and
coerces the raw argument dict through it before dispatch, generating the
model-visible JSON Schema from the same source. Covers the schema
generator, the structured error formatter, and end-to-end validation /
coercion through `ToolRegistry.call` for the three pilot tools
(read_file, write_file, stream_edit) — including the stream_edit int
coercion that closes harness-ln7j.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from harness.tools.base import (
    ToolRegistry,
    format_validation_error,
    tool_schema_from_model,
)
from harness.tools.read_file import ReadFileArgs, ReadFileTool
from harness.tools.stream_edit import StreamEditArgs, StreamEditTool
from harness.tools.write_file import WriteFileArgs, WriteFileTool

# ---------- schema generation ----------


def test_schema_collapses_optional_union_to_bare_type() -> None:
    schema = tool_schema_from_model(ReadFileArgs)
    # `offset: int | None` renders as a bare integer, not anyOf[int,null].
    assert schema["properties"]["offset"] == {
        "type": "integer",
        "description": schema["properties"]["offset"]["description"],
    }
    assert "anyOf" not in schema["properties"]["offset"]
    assert schema["required"] == ["path"]


def test_schema_preserves_enum_and_array_items() -> None:
    schema = tool_schema_from_model(StreamEditArgs)
    assert schema["properties"]["tool"]["enum"] == ["awk", "sed", "cut", "tr"]
    assert schema["properties"]["tool"]["type"] == "string"
    assert schema["properties"]["args"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": schema["properties"]["args"]["description"],
    }
    assert schema["required"] == ["tool", "args"]


def test_schema_strips_title_and_default_noise() -> None:
    schema = tool_schema_from_model(WriteFileArgs)
    for prop in schema["properties"].values():
        assert "title" not in prop
        assert "default" not in prop
    assert set(schema["properties"]) == {"path", "content", "overwrite"}
    assert schema["required"] == ["path", "content"]


# ---------- error formatting ----------


def test_format_validation_error_names_field_and_value() -> None:
    class M(BaseModel):
        model_config = ConfigDict(extra="forbid")
        n: int

    with pytest.raises(ValidationError) as ei:
        M.model_validate({"n": "not-a-number"})
    msg = format_validation_error("widget", ei.value)
    assert "widget" in msg
    assert "n:" in msg
    assert "not-a-number" in msg
    assert "retry" in msg.lower()


# ---------- end-to-end through the registry ----------


def test_read_file_coerces_numeric_string_offset(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("l1\nl2\nl3\n")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    # offset as a JSON string — pydantic coerces to int (harness-296bh).
    result = registry.call("read_file", {"path": "f.txt", "offset": "2", "limit": "1"})
    assert result.success
    assert result.output.startswith("l2")


def test_write_file_missing_required_field_is_validation_error(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WriteFileTool(root=tmp_path))
    result = registry.call("write_file", {"path": "x.txt"})  # content missing
    assert not result.success
    assert result.error == "validation_error"
    assert "content" in result.output


def test_write_file_rejects_unknown_field_with_accepts_hint(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WriteFileTool(root=tmp_path))
    result = registry.call("write_file", {"path": "x.txt", "content": "hi", "mode": "0644"})
    assert not result.success
    assert (result.error or "").startswith("unknown_kwarg:")
    assert "rejected unknown argument 'mode'" in result.output
    assert "Accepts:" in result.output


@pytest.mark.skipif(shutil.which("cut") is None, reason="cut not on PATH")
def test_stream_edit_coerces_int_args(tmp_path: Path) -> None:
    """harness-ln7j: a model that emits argv elements as ints
    (args=['-f', 2, ...]) must succeed via coercion, not fail on a
    TypeError that burns a round."""
    (tmp_path / "data.csv").write_text("a,b,c\n")
    registry = ToolRegistry()
    registry.register(StreamEditTool(root=tmp_path))
    result = registry.call(
        "stream_edit",
        {"tool": "cut", "args": ["-f", 2, "-d", ","], "paths": ["data.csv"]},
    )
    assert result.success, result.output
    assert result.output.strip() == "b"


def test_stream_edit_rejects_unknown_verb(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(StreamEditTool(root=tmp_path))
    result = registry.call("stream_edit", {"tool": "grep", "args": ["x"]})
    assert not result.success
    assert result.error == "validation_error"
    assert "tool" in result.output


def test_stream_edit_args_literal_matches_runtime_verbs() -> None:
    # Drift guard mirror: the typed Literal and the schema enum agree.
    schema = tool_schema_from_model(StreamEditArgs)
    assert set(schema["properties"]["tool"]["enum"]) == {"awk", "sed", "cut", "tr"}
