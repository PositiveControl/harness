#!/usr/bin/env python3
"""Collapsible follow-tail viewer for JSONL logs.

Tails a JSONL file the way ``tail -f`` does, but renders each line as a
collapsible tree node instead of a flat wall of text. Each record shows a
one-line header (``time │ mode │ model │ summary``); expand it to walk the full
nested object. Starts at the end of the file — it never ingests the whole thing.

Tuned for the harness model-IO drive logs (``/tmp/test_drive_*.jsonl``): the
summary line is synthesized per ``mode`` (tool calls + finish_reason for
``stream_with_tools``, token usage for ``complete``), but it falls back to a
generic summary for any other JSONL shape.

Usage:
    uv run python scripts/jsonl_tail.py /tmp/test_drive_1780094923.jsonl
    uv run python scripts/jsonl_tail.py LOG --tail 50   # seed last 50 lines
    uv run python scripts/jsonl_tail.py LOG --all       # ingest whole file

Keys: arrows/click to move + expand, ``e`` expand-all, ``c`` collapse-all,
``f`` toggle follow (auto-scroll), ``q`` quit.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Tree
    from textual.widgets.tree import TreeNode
except ImportError:  # pragma: no cover - depends on --extra tui
    sys.stderr.write(
        "textual not installed. Run: uv sync --extra tui\n"
        "(or: uv run --extra tui python scripts/jsonl_tail.py ...)\n"
    )
    raise SystemExit(1) from None

import typer

# --- truncation knobs --------------------------------------------------------
LEAF_MAX = 200  # max chars shown for a scalar leaf before eliding the middle
LABEL_MAX = 120  # max chars for a collapsed container's inline preview


def _short_ts(ts: Any) -> str:
    """``2026-05-29T22:48:51.589119+00:00`` -> ``22:48:51``. Best-effort."""
    s = str(ts)
    if "T" in s:
        clock = s.split("T", 1)[1]
        return clock.split(".", 1)[0].split("+", 1)[0]
    return s


def _elide(s: str, limit: int) -> str:
    s = s.replace("\n", "⏎")
    if len(s) <= limit:
        return s
    head = limit * 2 // 3
    tail = limit - head - 1
    return f"{s[:head]}…{s[-tail:]}"


def _summary(rec: dict[str, Any]) -> str:
    """One-line, mode-aware context string for the record header."""
    mode = rec.get("mode")
    if mode == "stream_with_tools":
        calls = [c.get("name", "?") for c in rec.get("parsed_calls") or []]
        bits = [f"→ {', '.join(calls)}"] if calls else []
        if fr := rec.get("finish_reason"):
            bits.append(f"[{fr}]")
        req = rec.get("request") or {}
        bits.append(f"{len(req.get('messages') or [])}msg/{len(req.get('tools') or [])}tool")
        return "  ".join(bits)
    if mode == "complete":
        resp = rec.get("response") or {}
        usage = resp.get("usage") or {}
        choices = resp.get("choices") or []
        fr = choices[0].get("finish_reason") if choices else None
        bits = []
        if fr:
            bits.append(f"[{fr}]")
        if usage:
            bits.append(
                f"{usage.get('prompt_tokens', '?')}→{usage.get('completion_tokens', '?')}tok"
            )
        return "  ".join(bits)
    # generic fallback: a few non-structural scalar fields
    scalars = [
        f"{k}={_elide(str(v), 40)}"
        for k, v in rec.items()
        if k not in {"ts", "mode", "model"} and not isinstance(v, dict | list)
    ]
    return "  ".join(scalars[:4])


def _preview(value: Any) -> str:
    """Inline preview for a collapsed dict/list node."""
    if isinstance(value, dict):
        return _elide("{" + ", ".join(value.keys()) + "}", LABEL_MAX)
    if isinstance(value, list):
        return f"[{len(value)} items]"
    return ""


def _add_value(node: TreeNode, key: str, value: Any) -> None:
    """Attach ``key: value`` under ``node``, recursing into containers."""
    if isinstance(value, dict):
        child = node.add(f"[bold]{key}[/]  [dim]{_preview(value)}[/]")
        for k, v in value.items():
            _add_value(child, str(k), v)
    elif isinstance(value, list):
        child = node.add(f"[bold]{key}[/]  [dim]{_preview(value)}[/]")
        for i, v in enumerate(value):
            _add_value(child, str(i), v)
    else:
        rendered = _elide(json.dumps(value, ensure_ascii=False), LEAF_MAX)
        node.add_leaf(f"[bold]{key}[/]: {rendered}")


class JsonlTail(App[None]):
    CSS = "Tree { padding: 0 1; }"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("e", "expand_all", "Expand all"),
        Binding("c", "collapse_all", "Collapse all"),
        Binding("f", "toggle_follow", "Follow"),
    ]

    def __init__(self, path: Path, seed_tail: int, ingest_all: bool) -> None:
        super().__init__()
        self.path = path
        self.seed_tail = seed_tail
        self.ingest_all = ingest_all
        self.follow = True
        self._pos = 0  # byte offset read up to
        self._count = 0

    def compose(self) -> ComposeResult:
        yield Header()
        self._tree: Tree[None] = Tree(self.path.name)
        self._tree.show_root = False
        self._tree.guide_depth = 2
        yield self._tree
        yield Footer()

    def on_mount(self) -> None:
        self._seed()
        self.set_interval(0.4, self._poll)
        self._sync_follow_title()

    # --- ingest ------------------------------------------------------------
    def _seed(self) -> None:
        if not self.path.exists():
            self._tree.root.add_leaf(f"[red]waiting for {self.path}…[/]")
            return
        data = self.path.read_bytes()
        self._pos = len(data)
        text = data.decode("utf-8", "replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not self.ingest_all:
            lines = list(deque(lines, maxlen=self.seed_tail))
        for ln in lines:
            self._add_record(ln)

    def _poll(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < self._pos:  # truncated / rotated
            self._pos = 0
        if size == self._pos:
            return
        with self.path.open("rb") as fh:
            fh.seek(self._pos)
            chunk = fh.read()
            self._pos = fh.tell()
        for ln in chunk.decode("utf-8", "replace").splitlines():
            if ln.strip():
                self._add_record(ln)
        if self.follow and self._tree.last_line:
            self._tree.scroll_end(animate=False)

    def _add_record(self, line: str) -> None:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            self._tree.root.add_leaf(f"[red]✗ unparseable[/] {_elide(line, LABEL_MAX)}")
            self._count += 1
            return
        if not isinstance(rec, dict):
            node = self._tree.root.add(f"[cyan]#{self._count}[/]  {_preview(rec)}")
            _add_value(node, "value", rec)
            self._count += 1
            return
        header = (
            f"[dim]{_short_ts(rec.get('ts', ''))}[/]  "
            f"[yellow]{rec.get('mode', '?')}[/]  "
            f"[cyan]{rec.get('model', '')}[/]  "
            f"{_summary(rec)}"
        )
        node = self._tree.root.add(header)
        for k, v in rec.items():
            _add_value(node, str(k), v)
        self._count += 1

    # --- actions -----------------------------------------------------------
    def action_expand_all(self) -> None:
        self._tree.root.expand_all()

    def action_collapse_all(self) -> None:
        for child in self._tree.root.children:
            child.collapse_all()

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        self._sync_follow_title()

    def _sync_follow_title(self) -> None:
        self.title = f"jsonl-tail · {self.path.name}"
        self.sub_title = f"{self._count} records · follow {'on' if self.follow else 'off'}"


def main(
    path: Path = typer.Argument(..., help="JSONL file to tail"),
    tail: int = typer.Option(20, "--tail", "-n", help="seed the last N lines on open"),
    ingest_all: bool = typer.Option(False, "--all", help="ingest the whole file (ignores --tail)"),
) -> None:
    JsonlTail(path, seed_tail=tail, ingest_all=ingest_all).run()


if __name__ == "__main__":
    typer.run(main)
