"""Vision spike — drive a vLLM-served VLM (Qwen3-VL on gx10:8001) through
the harness vision boundary. Image-in / text-out only; this is a VLM, not
a diffusion model, so there is no text-to-image here.

Exercises the boundary directly (VllmAdapter + ChatMessage.images) rather
than going through `harness chat`. If it proves useful it graduates into
the chat CLI (bd: harness-pom2o) and this script goes away.

Modes (combine freely):
    # single image + question
    uv run python scripts/vision_spike.py img.png "what is this?"

    # several images at once (compare / spot-the-difference)
    uv run python scripts/vision_spike.py "compare these" -i a.png -i b.png

    # remote image (vLLM fetches it server-side — must be reachable from gx10)
    uv run python scripts/vision_spike.py "describe" --url https://example.com/cat.jpg

    # grab a screen region and ask about it (macOS screencapture)
    uv run python scripts/vision_spike.py "what's on my screen?" --screenshot

    # stream tokens as they generate
    uv run python scripts/vision_spike.py img.png "describe in detail" --stream

    # grounding: model returns boxes, we draw them back on the image + open it
    #   (needs Pillow — rerun under `uv run --with pillow` if not installed)
    uv run --with pillow python scripts/vision_spike.py shot.png "find every button" --ground

    # multi-turn: attach image(s) once, ask follow-ups about them
    uv run python scripts/vision_spike.py img.png "what is this?" --repl

Served model is auto-discovered via GET /v1/models, so no model id needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.model.adapter import ChatMessage, ImageRef, image_from_path, image_from_url
from harness.model.vllm import VllmAdapter

# Qwen3-VL emits grounding boxes as a JSON array of {"bbox_2d":[x1,y1,x2,y2],
# "label":...}. We append this instruction in --ground mode and parse the
# first JSON array out of the reply.
_GROUND_SUFFIX = (
    " Return ONLY a JSON array. Each element must be "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"} '
    "with absolute pixel coordinates. No prose."
)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def grab_screenshot() -> Path:
    """Interactive region grab via macOS screencapture (-i). Returns the
    saved PNG path. The crosshair lets you pick a region or hit space for
    a window; Esc cancels (we exit cleanly)."""
    fd, name = tempfile.mkstemp(prefix="vision_spike_shot_", suffix=".png")
    os.close(fd)
    Path(name).unlink()  # screencapture wants to create it itself
    print("screencapture: select a region (space = window, esc = cancel)…", file=sys.stderr)
    subprocess.run(["screencapture", "-i", name], check=True)  # noqa: S603,S607
    p = Path(name)
    if not p.exists() or p.stat().st_size == 0:
        print("screenshot cancelled.", file=sys.stderr)
        raise SystemExit(1)
    return p


def parse_boxes(text: str) -> list[dict[str, object]]:
    """Pull the first JSON array out of a grounding reply. Tolerant: the
    model sometimes wraps it in a ```json fence or trailing prose."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [b for b in data if isinstance(b, dict) and "bbox_2d" in b]


def draw_boxes(image_path: Path, boxes: list[dict[str, object]]) -> Path | None:
    """Draw labeled boxes onto a copy of the image, save it, return the
    path. Returns None (with a hint) when Pillow isn't available — the
    caller falls back to printing the raw JSON."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print(
            "\n[Pillow not installed — rerun under `uv run --with pillow …` to draw boxes]",
            file=sys.stderr,
        )
        return None
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        coords = b.get("bbox_2d")
        if not (isinstance(coords, list) and len(coords) == 4):
            continue
        x1, y1, x2, y2 = (int(c) for c in coords)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        label = str(b.get("label", ""))
        if label:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(255, 0, 0))
    out = image_path.with_name(f"{image_path.stem}_boxes.png")
    img.save(out)
    return out


def _open(path: Path) -> None:
    subprocess.run(["open", str(path)], check=False)  # noqa: S603,S607 — macOS


def build_images(args: argparse.Namespace) -> tuple[list[ImageRef], Path | None]:
    """Assemble the image list from --image / --url / --screenshot.
    Returns (images, first_local_path) — the local path is what --ground
    draws boxes back onto."""
    refs: list[ImageRef] = []
    first_local: Path | None = None
    if args.screenshot:
        shot = grab_screenshot()
        first_local = shot
        refs.append(image_from_path(shot, detail=args.detail))
    for p in args.image or []:
        path = Path(p)
        if first_local is None:
            first_local = path
        refs.append(image_from_path(path, detail=args.detail))
    for u in args.url or []:
        refs.append(image_from_url(u, detail=args.detail))
    return refs, first_local


def run_once(adapter: VllmAdapter, messages: list[ChatMessage], args: argparse.Namespace) -> str:
    """One model round. Streams to stdout when --stream, else prints whole.
    Returns the full reply text."""
    if args.stream:
        parts: list[str] = []
        for tok in adapter.stream(messages, max_tokens=args.max_tokens):
            sys.stdout.write(tok)
            sys.stdout.flush()
            parts.append(tok)
        print()
        return "".join(parts)
    reply = adapter.complete(messages, max_tokens=args.max_tokens)
    print(reply)
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", nargs="?", help="image path, or prompt if other sources used")
    parser.add_argument("rest", nargs="?", help="prompt, when first arg is an image path")
    parser.add_argument("-i", "--image", action="append", help="image file (repeatable)")
    parser.add_argument("--url", action="append", help="remote image URL (repeatable)")
    parser.add_argument("--screenshot", action="store_true", help="grab a screen region first")
    parser.add_argument("--ground", action="store_true", help="ask for boxes, draw them back")
    parser.add_argument("--stream", action="store_true", help="stream tokens")
    parser.add_argument("--repl", action="store_true", help="multi-turn follow-ups on image(s)")
    parser.add_argument("--base-url", default="http://gx10-5fb9:8001/v1")
    parser.add_argument("--detail", default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    # Positional resolution: if extra image sources were given, `first` is
    # the prompt. Otherwise `first` is an image path and `rest` is the prompt.
    extra_sources = bool(args.image or args.url or args.screenshot)
    if extra_sources:
        prompt = args.first or args.rest or ""
    else:
        if not args.first or not args.rest:
            parser.error("give an image path + a prompt, or use -i/--url/--screenshot + a prompt")
        args.image = [args.first]
        prompt = args.rest
    if not prompt:
        parser.error("no prompt given")

    images, first_local = build_images(args)
    if not images:
        parser.error("no image source resolved")

    grounding = args.ground
    user_text = prompt + _GROUND_SUFFIX if grounding else prompt

    # context_window pinned to this model's max_model_len (16384), not the
    # adapter's 32k default — keeps the budget clamp honest.
    with VllmAdapter(base_url=args.base_url, context_window=16384) as adapter:
        print(f"server: {args.base_url}", file=sys.stderr)
        print(f"model:  {adapter.model}\n", file=sys.stderr)

        messages = [ChatMessage(role="user", content=user_text, images=tuple(images))]
        reply = run_once(adapter, messages, args)

        if grounding:
            boxes = parse_boxes(reply)
            if boxes and first_local is not None:
                out = draw_boxes(first_local, boxes)
                if out is not None:
                    print(f"\n[{len(boxes)} boxes drawn → {out}]", file=sys.stderr)
                    _open(out)
            elif not boxes:
                print("\n[no parseable boxes in reply]", file=sys.stderr)

        if not args.repl:
            return

        # Multi-turn: history carries the image context forward; follow-ups
        # are text-only. Ctrl-D / blank / 'exit' ends.
        messages.append(ChatMessage(role="assistant", content=reply))
        while True:
            try:
                follow = input("\nyou > ").strip()
            except EOFError:
                print()
                break
            if not follow or follow in {"exit", "quit"}:
                break
            messages.append(ChatMessage(role="user", content=follow))
            reply = run_once(adapter, messages, args)
            messages.append(ChatMessage(role="assistant", content=reply))


if __name__ == "__main__":
    main()
