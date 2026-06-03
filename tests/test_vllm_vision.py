"""Vision boundary tests — the ImageRef envelope + the vLLM adapter's
multimodal content-parts rendering. No network: requests route through a
MockTransport and we assert on the captured request body.

Pins two contracts:
  1. A text-only ChatMessage still serializes `content` as a plain string
     (byte-identical to the pre-vision adapter — a regression guard).
  2. A ChatMessage carrying images serializes `content` as an OpenAI
     content-parts list: text part first, then one image_url part each.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import httpx

from harness.model.adapter import (
    ChatMessage,
    ImageRef,
    image_from_path,
    image_from_url,
)
from harness.model.vllm import VllmAdapter, _render_content


def _make_factory(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Callable[..., httpx.Client]:
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    return factory


def _json_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload)


# ---------- ImageRef helpers ---------------------------------------------


def test_image_from_path_builds_base64_data_uri(tmp_path: Any) -> None:
    raw = b"\x89PNG\r\n\x1a\n fake png bytes"
    f = tmp_path / "shot.png"
    f.write_bytes(raw)

    ref = image_from_path(f)

    assert ref.detail == "auto"
    assert ref.url.startswith("data:image/png;base64,")
    encoded = ref.url.split(",", 1)[1]
    assert base64.b64decode(encoded) == raw


def test_image_from_path_sniffs_jpeg_mime(tmp_path: Any) -> None:
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"\xff\xd8\xff fake jpeg")
    ref = image_from_path(f, detail="high")
    assert ref.url.startswith("data:image/jpeg;base64,")
    assert ref.detail == "high"


def test_image_from_path_falls_back_to_png_for_unknown_suffix(tmp_path: Any) -> None:
    f = tmp_path / "frame.weirdext"
    f.write_bytes(b"bytes")
    ref = image_from_path(f)
    assert ref.url.startswith("data:image/png;base64,")


def test_image_from_url_passthrough() -> None:
    ref = image_from_url("https://example.com/cat.jpg", detail="low")
    assert ref == ImageRef(url="https://example.com/cat.jpg", detail="low")


# ---------- _render_content ----------------------------------------------


def test_render_content_text_only_is_plain_string() -> None:
    m = ChatMessage(role="user", content="hello")
    assert _render_content(m) == "hello"


def test_render_content_with_image_is_parts_list() -> None:
    m = ChatMessage(
        role="user",
        content="what is this?",
        images=(ImageRef(url="data:image/png;base64,QUJD", detail="high"),),
    )
    rendered = _render_content(m)
    assert rendered == [
        {"type": "text", "text": "what is this?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,QUJD", "detail": "high"},
        },
    ]


def test_render_content_empty_text_omits_text_part() -> None:
    m = ChatMessage(
        role="user",
        content="",
        images=(ImageRef(url="https://example.com/a.png"),),
    )
    rendered = _render_content(m)
    assert rendered == [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "auto"}},
    ]


def test_render_content_multiple_images() -> None:
    m = ChatMessage(
        role="user",
        content="compare",
        images=(ImageRef(url="u1"), ImageRef(url="u2")),
    )
    rendered = _render_content(m)
    assert isinstance(rendered, list)
    assert [p["type"] for p in rendered] == ["text", "image_url", "image_url"]


# ---------- end-to-end through complete() --------------------------------


def test_complete_sends_multimodal_content_for_image_message() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"message": {"content": "a cat"}, "finish_reason": "stop"}]}
        )

    adapter = VllmAdapter(model="some-vlm", base_url="http://gx10:8001/v1")
    message = ChatMessage(
        role="user",
        content="what is this?",
        images=(ImageRef(url="data:image/png;base64,QUJD"),),
    )
    with patch("httpx.Client", _make_factory(handler)):
        reply = adapter.complete([message])

    assert reply == "a cat"
    sent = captured["body"]["messages"][0]
    assert sent["role"] == "user"
    assert isinstance(sent["content"], list)
    assert sent["content"][0] == {"type": "text", "text": "what is this?"}
    assert sent["content"][1]["type"] == "image_url"
    assert sent["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_complete_text_only_message_still_sends_plain_string() -> None:
    """Regression guard: a turn with no images must serialize exactly as
    before the vision boundary landed."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return _json_response(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
        )

    adapter = VllmAdapter(model="some-model", base_url="http://gx10:8001/v1")
    with patch("httpx.Client", _make_factory(handler)):
        adapter.complete([ChatMessage(role="user", content="hello")])

    assert captured["body"]["messages"][0]["content"] == "hello"
