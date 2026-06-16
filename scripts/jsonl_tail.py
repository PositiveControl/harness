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

Two views, toggled with ``v``:
  * record — one collapsible node per round (the raw tail).
  * drive  — rounds grouped issue → segment → round, reconstructing the
    auto-iterate drive structure (attempts, re-feeds ↺, ✓close) live from the
    records, no external map needed.

Keys: ``v`` switch view, arrows/click to move + expand, ``e`` expand-all,
``c`` collapse-all, ``f`` toggle follow (auto-scroll), ``q`` quit.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, ClassVar

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding, BindingType
    from textual.widgets import Footer, Header, Tree
    from textual.widgets.tree import TreeNode
except ImportError:  # pragma: no cover - depends on --extra tui
    sys.stderr.write(
        "textual not installed. Run: uv sync --extra all\n"
        "(or: uv run --extra all python scripts/jsonl_tail.py ...)\n"
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
        bits.append(
            f"{len(req.get('messages') or [])}msg"
            f"/{len(calls)}call"
            f"/{len(req.get('tools') or [])}avail"
        )
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


def _add_value(node: TreeNode[None], key: str, value: Any) -> None:
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


# --- drive-view inference ----------------------------------------------------
# A drive log is a flat stream of model rounds, but the run has structure: an
# auto-iterate loop re-feeds the same bd issue until it closes (or gives up).
# We reconstruct that structure live from the records themselves — no external
# map needed. Signals: the handoff names the target issue (harness-XXXX); a
# `bd close <id>` shell call marks a close; the same issue appearing in more
# than one contiguous segment means it was re-fed (reopened / restored).

_ID_RE = re.compile(r"harness-[a-z0-9]{4,}")
_CLOSE_RE = re.compile(r"bd close\s+(harness-[a-z0-9]{4,})")
# ids that appear only because the tool-rules system prompt cites them
_RULE_IDS = frozenset({"harness-zcxw", "harness-k52f", "harness-lpsq"})


def _record_messages(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return (rec.get("request") or {}).get("messages") or []


def _target_issue(rec: dict[str, Any]) -> str | None:
    """The bd issue this round is driving, read from the handoff text."""
    for msg in _record_messages(rec):
        if msg.get("role") not in {"user", "system"}:
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for hit in _ID_RE.findall(content):
            if hit not in _RULE_IDS:
                return str(hit)
    return None


def _closed_ids(rec: dict[str, Any]) -> set[str]:
    """Issue ids this round ran `bd close` on (real shell calls only)."""
    out: set[str] = set()
    for call in rec.get("parsed_calls") or []:
        if call.get("name") == "shell":
            cmd = str((call.get("arguments") or {}).get("cmd", ""))
            out.update(_CLOSE_RE.findall(cmd))
    return out


def _call_names(rec: dict[str, Any]) -> list[str]:
    return [c.get("name", "?") for c in rec.get("parsed_calls") or []]


class JsonlTail(App[None]):
    CSS = "Tree { padding: 0 1; }"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("v", "toggle_view", "View: record/drive"),
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
        self.view = "record"  # "record" (per-round) | "drive" (issue→segment)
        self._pos = 0  # byte offset read up to
        # parsed records retained so the drive view can be rebuilt on demand.
        # each entry: (index, parsed_obj_or_None, raw_line)
        self._records: list[tuple[int, Any, str]] = []

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
        new = self._store(lines)
        self._render(new)

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
        lines = [ln for ln in chunk.decode("utf-8", "replace").splitlines() if ln.strip()]
        if not lines:
            return
        new = self._store(lines)
        self._render(new)
        if self.follow and self._tree.last_line:
            self._tree.scroll_end(animate=False)

    def _store(self, lines: list[str]) -> list[tuple[int, Any, str]]:
        """Parse + retain new records; return the batch just added."""
        added: list[tuple[int, Any, str]] = []
        for ln in lines:
            try:
                obj: Any = json.loads(ln)
            except json.JSONDecodeError:
                obj = None
            entry = (len(self._records), obj, ln)
            self._records.append(entry)
            added.append(entry)
        return added

    # --- render dispatch ---------------------------------------------------
    def _render(self, new: list[tuple[int, Any, str]]) -> None:
        if self.view == "drive":
            self._build_drive()  # cheap full rebuild; record count is bounded
        else:
            for idx, obj, raw in new:  # incremental append
                self._record_node(self._tree.root, idx, obj, raw)
        self._sync_follow_title()

    def _record_node(self, parent: TreeNode[None], idx: int, obj: Any, raw: str) -> TreeNode[None]:
        """Render one round as a collapsible node under ``parent``."""
        if obj is None:
            return parent.add_leaf(f"[red]✗ unparseable #{idx}[/] {_elide(raw, LABEL_MAX)}")
        if not isinstance(obj, dict):
            node = parent.add(f"[cyan]#{idx}[/]  {_preview(obj)}")
            _add_value(node, "value", obj)
            return node
        header = (
            f"[dim]#{idx} {_short_ts(obj.get('ts', ''))}[/]  "
            f"[yellow]{obj.get('mode', '?')}[/]  "
            f"[cyan]{obj.get('model', '')}[/]  "
            f"{_summary(obj)}"
        )
        node = parent.add(header)
        for k, v in obj.items():
            _add_value(node, str(k), v)
        return node

    # --- drive view --------------------------------------------------------
    def _segments(self) -> list[tuple[str | None, list[tuple[int, Any, str]]]]:
        """Group rounds into contiguous runs sharing a target issue.

        None targets (rounds with no handoff id, e.g. finish_reason=null
        continuations) carry forward the previous issue so an attempt stays one
        segment instead of fragmenting.
        """
        segs: list[tuple[str | None, list[tuple[int, Any, str]]]] = []
        carried: str | None = None
        for entry in self._records:
            obj = entry[1]
            tgt = _target_issue(obj) if isinstance(obj, dict) else None
            if tgt is None:
                tgt = carried
            else:
                carried = tgt
            if segs and segs[-1][0] == tgt:
                segs[-1][1].append(entry)
            else:
                segs.append((tgt, [entry]))
        return segs

    def _build_drive(self) -> None:
        self._tree.clear()
        segs = self._segments()
        # per-issue rollup: how many segments (attempts), rounds, closes
        attempts: Counter[str] = Counter()
        rounds: Counter[str] = Counter()
        closes: Counter[str] = Counter()
        order: list[str] = []
        for tgt, entries in segs:
            if not tgt:
                continue
            if tgt not in attempts:
                order.append(tgt)
            attempts[tgt] += 1
            rounds[tgt] += len(entries)
            for _, obj, _ in entries:
                if isinstance(obj, dict):
                    closes[tgt] += len(_closed_ids(obj) & {tgt})

        issue_nodes: dict[str, TreeNode[None]] = {}
        for tgt in order:
            reopened = " [red]↺reopened[/]" if attempts[tgt] > 1 else ""
            won = f" [green]✓{closes[tgt]}[/]" if closes[tgt] else ""
            label = f"[bold cyan]{tgt}[/]  x{attempts[tgt]} seg · {rounds[tgt]}r{won}{reopened}"
            issue_nodes[tgt] = self._tree.root.add(label)

        # segments in stream order, each under its issue rollup (orphans → root)
        attempt_seen: Counter[str] = Counter()
        for tgt, entries in segs:
            parent = issue_nodes.get(tgt) if tgt else None
            if parent is None:
                parent = self._tree.root
            lo, hi = entries[0][0], entries[-1][0]
            names = Counter(n for _, o, _ in entries if isinstance(o, dict) for n in _call_names(o))
            fins = Counter(
                o.get("finish_reason")
                for _, o, _ in entries
                if isinstance(o, dict) and o.get("finish_reason")
            )
            seg_closes = sum(
                len(_closed_ids(o) & {tgt}) for _, o, _ in entries if isinstance(o, dict) and tgt
            )
            attempt_seen[tgt or "—"] += 1
            calls = " ".join(f"{n}:{c}" for n, c in names.most_common())
            fin = " ".join(f"{k}:{v}" for k, v in fins.most_common())
            mark = " [green]✓close[/]" if seg_closes else ""
            seg_label = (
                f"[dim]rows {lo}-{hi}[/] [{len(entries)}r] "
                f"#{attempt_seen[tgt or '—']}{mark}  "
                f"[dim]calls[/] {calls or '—'}  [dim]fin[/] {fin or '—'}"
            )
            seg_node = parent.add(seg_label)
            for idx, obj, raw in entries:
                self._record_node(seg_node, idx, obj, raw)

    # --- actions -----------------------------------------------------------
    def action_toggle_view(self) -> None:
        self.view = "drive" if self.view == "record" else "record"
        self._tree.clear()
        if self.view == "drive":
            self._build_drive()
        else:
            for idx, obj, raw in self._records:
                self._record_node(self._tree.root, idx, obj, raw)
        self._sync_follow_title()

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
        self.sub_title = (
            f"{len(self._records)} records · {self.view} view · "
            f"follow {'on' if self.follow else 'off'}"
        )


def main(
    path: Path = typer.Argument(..., help="JSONL file to tail"),
    tail: int = typer.Option(20, "--tail", "-n", help="seed the last N lines on open"),
    ingest_all: bool = typer.Option(False, "--all", help="ingest the whole file (ignores --tail)"),
) -> None:
    JsonlTail(path, seed_tail=tail, ingest_all=ingest_all).run()


if __name__ == "__main__":
    typer.run(main)
