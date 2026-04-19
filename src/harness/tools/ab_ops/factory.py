"""Single-point constructor for the full ab ops tool set
(harness-u73g)."""

from __future__ import annotations

from harness.tools.ab_ops._shared import _Adapter
from harness.tools.ab_ops.graph import (
    CommentsTool,
    DepTool,
    FindDuplicatesTool,
    LabelTool,
)
from harness.tools.ab_ops.memory import ForgetTool, MemoriesTool, RememberTool, RetroTool
from harness.tools.ab_ops.mutation import (
    CaptureTool,
    CloseTool,
    DeferTool,
    DeleteTool,
    ReopenTool,
    UpdateTool,
)
from harness.tools.ab_ops.plan_status import (
    DriftTool,
    PlanTool,
    ReprioritizeTool,
    StatusTool,
)
from harness.tools.ab_ops.resume import PersistFocusNoteTool
from harness.tools.ab_ops.search import ListTool, SearchTool


def make_ops_tools(
    adapter: _Adapter,
) -> tuple[
    PlanTool,
    CaptureTool,
    StatusTool,
    DriftTool,
    ReprioritizeTool,
    CloseTool,
    DeferTool,
    RetroTool,
    ReopenTool,
    DeleteTool,
    UpdateTool,
    SearchTool,
    ListTool,
    MemoriesTool,
    RememberTool,
    ForgetTool,
    DepTool,
    LabelTool,
    CommentsTool,
    FindDuplicatesTool,
    PersistFocusNoteTool,
]:
    """Construct the full ab ops tool set. The CLI calls this once per
    session and passes the tuple to the registry."""
    return (
        PlanTool(adapter),
        CaptureTool(adapter),
        StatusTool(adapter),
        DriftTool(adapter),
        ReprioritizeTool(adapter),
        CloseTool(adapter),
        DeferTool(adapter),
        RetroTool(adapter),
        ReopenTool(adapter),
        DeleteTool(adapter),
        UpdateTool(adapter),
        SearchTool(adapter),
        ListTool(adapter),
        MemoriesTool(adapter),
        RememberTool(adapter),
        ForgetTool(adapter),
        DepTool(adapter),
        LabelTool(adapter),
        CommentsTool(adapter),
        FindDuplicatesTool(adapter),
        PersistFocusNoteTool(adapter),
    )
