"""Per-request debug logging for the web layer.

When `HARNESS_WEB_DEBUG_DIR` is set, every request handled by an app
built via `build_character_app` captures a structured record of the
turn — the input, the model's raw output, retrieved chunks, citations,
and the rendered response — to a per-request JSON file under that
directory. The files are replayable: they grow the fixture set behind
`harness eval *` over time without manual transcription.

Empty / unset env var means no capture; the helper is a no-op so it's
safe to leave the wiring in production.

Generalizable — not TFR-specific. Character extensions (the TFR
explainer and any future character on the same factory) write into the
same record shape so the eval fixture pipeline is reusable.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEBUG_DIR_ENV = "HARNESS_WEB_DEBUG_DIR"


def debug_dir() -> Path | None:
    """Resolved per-request debug directory, or `None` when capture is
    disabled. Caller invokes this once per request; the cost is cheap.
    Creates the directory on first use so deployments can set the env
    var to a non-existent path and the harness will materialize it."""
    raw = os.environ.get(_DEBUG_DIR_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RequestDebugLog:
    """Mutable record built during one request, flushed to disk in
    `finalize()`. Fields are intentionally permissive (`dict[str, Any]`)
    so character extensions can stuff their own structured outputs in
    without coordinating with this module. The TFR extension, for
    example, populates `parsed_outputs["parsed_notam"]` with the
    deterministic parser's struct.

    Use as a context manager so finalization happens even when the
    endpoint raises:

        with RequestDebugLog.start(character_name="airton_c_tfr",
                                    endpoint="/explain") as log:
            log.raw_input = {"text": text}
            ...
    """

    character_name: str
    endpoint: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at_unix: float = field(default_factory=time.time)
    duration_ms: float | None = None
    raw_input: Any = None
    parsed_outputs: dict[str, Any] = field(default_factory=dict)
    raw_model_completion: str | None = None
    parsed_json_reply: Any = None
    retrieved_chunks: list[Any] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def start(cls, *, character_name: str, endpoint: str) -> RequestDebugLog:
        return cls(character_name=character_name, endpoint=endpoint)

    def __enter__(self) -> RequestDebugLog:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        if exc is not None:
            self.error = f"{type(exc).__name__}: {exc}"
        self.duration_ms = (time.time() - self.started_at_unix) * 1000.0
        self.finalize()

    def finalize(self) -> Path | None:
        """Write the record to `debug_dir() / <iso-ts>-<endpoint>-<id>.json`
        if capture is on; otherwise no-op. Returns the written path so
        tests can assert on it."""
        target = debug_dir()
        if target is None:
            return None
        # Endpoint slug for the filename — strip leading slash and
        # replace path separators so `/explain` -> `explain` and
        # `/debug/parse` -> `debug-parse`.
        slug = self.endpoint.lstrip("/").replace("/", "-") or "root"
        iso_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.started_at_unix))
        filename = f"{iso_ts}-{slug}-{self.request_id}.json"
        path = target / filename
        payload: dict[str, Any] = {
            "character": self.character_name,
            "endpoint": self.endpoint,
            "request_id": self.request_id,
            "started_at_unix": self.started_at_unix,
            "duration_ms": self.duration_ms,
            "raw_input": self.raw_input,
            "parsed_outputs": self.parsed_outputs,
            "raw_model_completion": self.raw_model_completion,
            "parsed_json_reply": self.parsed_json_reply,
            "retrieved_chunks": self.retrieved_chunks,
            "citations": self.citations,
            "error": self.error,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


__all__ = ["RequestDebugLog", "debug_dir"]
