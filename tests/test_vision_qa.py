"""Tests for advisory vision-QA (harness-ke4hx.3): verdict parsing,
prompt assembly, and the one-shot run path with a stub adapter. No
network and no real VLM — the stub returns canned text so the parser and
message assembly are what's under test."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from harness.driver.vision_qa import (
    QaVerdict,
    build_qa_prompt,
    build_rubric,
    parse_verdict,
    run_advisory_qa,
    run_vision_qa,
)
from harness.model.adapter import ChatMessage


class _RaisingAdapter:
    """Adapter whose complete() always raises — stands in for an
    unreachable / erroring vision endpoint."""

    id = "raising"
    context_window = 16_384

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        raise RuntimeError("endpoint down")


class _StubAdapter:
    """Minimal ModelAdapter that records its call and returns a fixed
    reply. Satisfies the `complete` protocol used by run_vision_qa."""

    id = "stub"
    context_window = 16_384

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[tuple[list[ChatMessage], int, float]] = []

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append((list(messages), max_tokens, temperature))
        return self._reply


def _tmp_image(tmp_path: Path) -> Path:
    f = tmp_path / "shot.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    return f


# ---------- parse_verdict ------------------------------------------------


def test_parse_pass_with_reason() -> None:
    v = parse_verdict("PASS: the car and score are visible")
    assert v.verdict == "pass"
    assert v.reason == "the car and score are visible"
    assert v.raw == "PASS: the car and score are visible"


def test_parse_fail_with_dash_reason() -> None:
    v = parse_verdict("FAIL - canvas is blank")
    assert v.verdict == "fail"
    assert v.reason == "canvas is blank"


def test_parse_tolerates_markdown_and_preamble() -> None:
    v = parse_verdict("Verdict: **FAIL** — no controls rendered")
    assert v.verdict == "fail"
    assert "no controls rendered" in v.reason


def test_parse_first_token_wins() -> None:
    # A reply that mentions both leads with PASS → pass.
    v = parse_verdict("PASS: works, would not FAIL on input")
    assert v.verdict == "pass"


def test_parse_no_token_is_unknown() -> None:
    v = parse_verdict("the image shows a blue square")
    assert v.verdict == "unknown"
    assert v.reason == "the image shows a blue square"


def test_parse_reason_is_truncated() -> None:
    v = parse_verdict("PASS: " + "x" * 500)
    assert v.verdict == "pass"
    assert len(v.reason) <= 280


def test_parse_reason_stops_at_newline() -> None:
    v = parse_verdict("FAIL: blank screen\nmore detail on the next line")
    assert v.reason == "blank screen"


# ---------- build_qa_prompt ----------------------------------------------


def test_build_prompt_embeds_rubric() -> None:
    prompt = build_qa_prompt("score must be visible")
    assert "score must be visible" in prompt
    assert "PASS" in prompt
    assert "FAIL" in prompt


def test_build_prompt_handles_empty_rubric() -> None:
    prompt = build_qa_prompt("   ")
    assert "(no rubric provided)" in prompt


# ---------- run_vision_qa ------------------------------------------------


def test_run_vision_qa_attaches_image_and_parses(tmp_path: Path) -> None:
    adapter = _StubAdapter("PASS: looks like a working game")
    shot = _tmp_image(tmp_path)

    verdict = run_vision_qa(adapter, shot, "a playable game with a score")

    assert isinstance(verdict, QaVerdict)
    assert verdict.verdict == "pass"
    assert verdict.reason == "looks like a working game"

    # One call, deterministic settings, a single image+text user turn.
    assert len(adapter.calls) == 1
    messages, max_tokens, temperature = adapter.calls[0]
    assert temperature == 0.0
    assert max_tokens == 64
    assert len(messages) == 1
    msg = messages[0]
    assert msg.role == "user"
    assert len(msg.images) == 1
    assert msg.images[0].url.startswith("data:image/png;base64,")
    assert "a playable game with a score" in msg.content


def test_run_vision_qa_unknown_on_garbage_reply(tmp_path: Path) -> None:
    adapter = _StubAdapter("I cannot tell from this angle")
    verdict = run_vision_qa(adapter, _tmp_image(tmp_path), "rubric")
    assert verdict.verdict == "unknown"


# ---- harness-ke4hx.4: build_rubric + run_advisory_qa -------------------


def test_build_rubric_title_only() -> None:
    assert build_rubric("score must show") == "score must show"


def test_build_rubric_title_and_body() -> None:
    r = build_rubric("score must show", "top-right corner, white text")
    assert "score must show" in r
    assert "top-right corner" in r


def test_build_rubric_empty_falls_back() -> None:
    assert build_rubric("", "") == "(no rubric provided)"


def test_advisory_skips_when_adapter_none(tmp_path: Path) -> None:
    """No vision adapter → logged skip, no verdict, no JSONL written."""
    logs: list[str] = []
    jsonl = tmp_path / "vision_qa.jsonl"
    out = run_advisory_qa(
        None,
        _tmp_image(tmp_path),
        "rubric",
        gate_passed=True,
        issue_id="x-1",
        jsonl_path=jsonl,
        log=logs.append,
    )
    assert out is None
    assert any("no vision adapter" in m for m in logs)
    assert not jsonl.exists()


def test_advisory_skips_when_screenshot_missing(tmp_path: Path) -> None:
    logs: list[str] = []
    out = run_advisory_qa(
        _StubAdapter("PASS: ok"),
        tmp_path / "nope.png",
        "rubric",
        gate_passed=True,
        issue_id="x-1",
        log=logs.append,
    )
    assert out is None
    assert any("no screenshot" in m for m in logs)


def test_advisory_skips_when_adapter_raises(tmp_path: Path) -> None:
    """Unreachable / erroring endpoint → logged skip, NEVER propagates.
    This is the graceful-degradation contract: a down VLM must not fail
    the drive."""
    logs: list[str] = []
    out = run_advisory_qa(
        _RaisingAdapter(),
        _tmp_image(tmp_path),
        "rubric",
        gate_passed=True,
        issue_id="x-1",
        log=logs.append,
    )
    assert out is None
    assert any("vision call failed" in m for m in logs)


def test_advisory_logs_and_writes_jsonl_on_success(tmp_path: Path) -> None:
    logs: list[str] = []
    jsonl = tmp_path / "sub" / "vision_qa.jsonl"
    out = run_advisory_qa(
        _StubAdapter("PASS: car and score visible"),
        _tmp_image(tmp_path),
        "a playable game",
        gate_passed=True,
        issue_id="harness-zz1",
        jsonl_path=jsonl,
        log=logs.append,
    )
    assert out is not None
    assert out.verdict == "pass"
    assert jsonl.is_file()
    record = json.loads(jsonl.read_text().strip())
    assert record["issue_id"] == "harness-zz1"
    assert record["verdict"] == "pass"
    assert record["gate_passed"] is True
    assert record["agree"] is True  # VLM pass + gate pass → agree


def test_advisory_agree_false_when_verdict_disagrees_with_gate(tmp_path: Path) -> None:
    """VLM says FAIL but the gate passed → agree=False (the disagreement
    column .5's precision/recall report is built on)."""
    jsonl = tmp_path / "vision_qa.jsonl"
    out = run_advisory_qa(
        _StubAdapter("FAIL: canvas is blank"),
        _tmp_image(tmp_path),
        "a playable game",
        gate_passed=True,
        issue_id="harness-zz2",
        jsonl_path=jsonl,
        log=lambda _m: None,
    )
    assert out is not None
    assert out.verdict == "fail"
    record = json.loads(jsonl.read_text().strip())
    assert record["agree"] is False
