"""Tool layer. Each tool is a callable with a JSON-schema spec the model
can reason about. The orchestrator runs the tool loop; individual tools
do not know about chat, retrieval, persona, or authorization — those
concerns live in `src/harness/orchestrator/`."""

from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from harness.tools.read_file import ReadFileTool
from harness.tools.search_facts import SearchFactsTool
from harness.tools.search_memory import SearchMemoryTool
from harness.tools.shell import ShellTool
from harness.tools.write_file import WriteFileTool

__all__ = [
    "ModelReply",
    "ReadFileTool",
    "SearchFactsTool",
    "SearchMemoryTool",
    "ShellTool",
    "StreamChunk",
    "StreamComplete",
    "StreamText",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "WriteFileTool",
]
