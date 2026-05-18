"""Tool layer. Each tool is a callable with a JSON-schema spec the model
can reason about. The orchestrator runs the tool loop; individual tools
do not know about chat, retrieval, persona, or authorization — those
concerns live in `src/harness/orchestrator/`."""

from harness.tools.assemble_context import AssembleContextTool
from harness.tools.base import (
    ModelReply,
    StreamChunk,
    StreamComplete,
    StreamText,
    Tool,
    ToolCall,
    ToolHit,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from harness.tools.calc import CalcTool
from harness.tools.date_math import DateMathTool
from harness.tools.edit_file import EditFileTool
from harness.tools.fetch_url import FetchUrlTool
from harness.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from harness.tools.glob import GlobTool
from harness.tools.grep import GrepTool
from harness.tools.introspect import IntrospectContext, IntrospectTool
from harness.tools.list_dir import ListDirTool
from harness.tools.now import NowTool
from harness.tools.ops import ConsolidateMemoryTool, ScribeSessionTool
from harness.tools.phraseology_lint import (
    PhraseologyLintTool,
    PhraseologyVerdict,
    lint_utterance,
)
from harness.tools.profiles import (
    DEFAULT_PROFILE,
    TOOL_PROFILES,
    resolve_tool_names,
)
from harness.tools.python_eval import PythonEvalTool
from harness.tools.read_file import ReadFileTool
from harness.tools.remember import RememberEventTool, RememberFactTool
from harness.tools.search_facts import SearchFactsTool
from harness.tools.search_memory import SearchMemoryTool
from harness.tools.search_scholar import SearchScholarTool
from harness.tools.search_web import SearchWebTool
from harness.tools.shell import ShellTool
from harness.tools.stats import StatsTool
from harness.tools.subagent import SpawnSubagentTool
from harness.tools.transcript_ingest import TranscriptIngestTool
from harness.tools.tz_convert import TzConvertTool
from harness.tools.write_file import WriteFileTool

__all__ = [
    "DEFAULT_PROFILE",
    "TOOL_PROFILES",
    "AssembleContextTool",
    "CalcTool",
    "ConsolidateMemoryTool",
    "DateMathTool",
    "EditFileTool",
    "FetchUrlTool",
    "GitDiffTool",
    "GitLogTool",
    "GitStatusTool",
    "GlobTool",
    "GrepTool",
    "IntrospectContext",
    "IntrospectTool",
    "ListDirTool",
    "ModelReply",
    "NowTool",
    "PhraseologyLintTool",
    "PhraseologyVerdict",
    "PythonEvalTool",
    "ReadFileTool",
    "RememberEventTool",
    "RememberFactTool",
    "ScribeSessionTool",
    "SearchFactsTool",
    "SearchMemoryTool",
    "SearchScholarTool",
    "SearchWebTool",
    "ShellTool",
    "SpawnSubagentTool",
    "StatsTool",
    "StreamChunk",
    "StreamComplete",
    "StreamText",
    "Tool",
    "ToolCall",
    "ToolHit",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TranscriptIngestTool",
    "TzConvertTool",
    "WriteFileTool",
    "lint_utterance",
    "resolve_tool_names",
]
