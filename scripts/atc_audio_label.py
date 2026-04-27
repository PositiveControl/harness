"""ATC audio label: fix transcripts + mark expected verdict/cite (harness-hp8k).

Third leg of the harness-o92o pipeline. Walks meta/ + stt/ produced by
the prior beads and presents each whisper-segmented utterance to a
human for review. Each labelled utterance lands as one JSONL row in
utt/<clip_id>.jsonl with the fields the audio-mode eval (harness-gy5z)
needs to score WER + verdict accuracy:

    clip_id, utt_index, start_s, end_s,
    speaker_role (pilot|controller|unknown),
    facility, position, frequency, recorded_at, event_tag,
    transcript_text (human-corrected),
    transcript_seed (whisper original — kept for WER vs human),
    transcript_confidence (whisper avg_logprob),
    expected_verdict (ok|wrong|incomplete|out_of_scope),
    expected_section, expected_phraseology, mismatch, notes,
    model_repo, human_verified, labeled_at.

Flow (sequential CLI):

  • Top-level prompt per utterance: (o) one-key out-of-scope, (l) full
    label, (p) play audio segment via ffplay, (s) skip rest of clip,
    (q) save+quit. Most ATC clips are 80%+ OOS for our purposes
    (chatter / pilot side / ground noise) and the `o` fast-path keeps
    that volume tractable.

  • `l` enters a 5-field prompt sequence: transcript, speaker, verdict,
    cite-block (section/phraseology/mismatch — only when verdict needs
    them), notes. ENTER accepts the bracketed default. The seed
    transcript + heuristic speaker default + verdict=out_of_scope
    keep keystrokes minimal on the common case.

Idempotency: existing utt/<clip_id>.jsonl rows are read at startup;
their utt_index values are skipped. Re-running picks up where the
human left off. Re-labelling an already-labelled utterance isn't
supported in v0 — delete the offending row from utt/<clip_id>.jsonl
and re-run if you need to redo a specific entry.

Design note (vs the bead spec): labelling at scale is keystroke-bound,
not navigation-bound. A sequential CLI ships in a day and labels
faster than a Textual widget app for a corpus this size. If labelling
friction hits, a follow-up bead can wrap a Textual layer over the
same per-utterance row builder.

Usage (from repo root):
    uv run python scripts/atc_audio_label.py
    uv run python scripts/atc_audio_label.py --only 8cf1e5c1f6c6
    uv run python scripts/atc_audio_label.py --auto-play
    uv run python scripts/atc_audio_label.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "character" / "airton_c1" / "atc_audio"

VERDICTS = ("ok", "wrong", "incomplete", "out_of_scope")
SPEAKERS = ("pilot", "controller", "unknown")

# Top-level keystroke menu per utterance. Single chars map to actions
# that don't tab through the field-by-field prompt sequence — the OOS
# fast-path is what makes labelling tractable on this corpus.
_TOP_CHOICES = ("o", "l", "p", "s", "q")


# ---------- pure helpers (testable) ----------


def default_speaker(position: str | None, event_tag: str | None) -> str:
    """Heuristic speaker default for the seed-and-pick flow.

    CTAF / unknown-position clips are pilot-only by definition (no
    controller on frequency). Tower/Ground/Approach/Departure clips
    are mixed but mostly controller-issued (that's what we recorded
    them for). 'Guard' is 121.5 — usually pilots, sometimes ATC; flag
    `unknown` so the human picks. Same for free-form events that have
    no position context (e.g. callsign-only filenames)."""
    if position is None:
        return "unknown"
    pos_lower = position.lower()
    if "ctaf" in pos_lower:
        return "pilot"
    if "guard" in pos_lower:
        return "unknown"
    if any(token in pos_lower for token in ("twr", "gnd", "app", "dep", "ctr", "center")):
        return "controller"
    if event_tag and event_tag.lower() in {"groundchatter"}:
        return "unknown"
    return "unknown"


def _frequency_from_extras(extras: list[Any] | None) -> str | None:
    """Pick the first all-digit token from `extras` as a frequency
    candidate. Numeric tokens like 121100 → '121.100' MHz; runway
    pairs like 0826 are also possible. Don't try to disambiguate at
    this layer — the labelling step preserves the raw string and the
    human can correct in `notes` if it matters."""
    if not extras:
        return None
    for value in extras:
        text = str(value)
        if text.isdigit():
            return text
    return None


def row_template(
    meta: dict[str, Any],
    segment: dict[str, Any],
    *,
    clip_id: str,
    utt_index: int,
    model_repo: str,
) -> dict[str, Any]:
    """Build the seed row for one utterance — defaults filled, awaiting
    human edits. Pure function so the prompt loop and the test suite
    share the schema definition."""
    parsed = meta.get("parsed") or {}
    facility = parsed.get("icao")
    position = parsed.get("position")
    event_tag = parsed.get("event_tag")
    speaker = default_speaker(position, event_tag)

    start_s = float(segment.get("start", 0.0) or 0.0)
    end_s = float(segment.get("end", 0.0) or 0.0)
    seed_text = str(segment.get("text", "")).strip()
    confidence = segment.get("avg_logprob")

    return {
        "clip_id": clip_id,
        "utt_index": utt_index,
        "start_s": start_s,
        "end_s": end_s,
        "speaker_role": speaker,
        "facility": facility,
        "position": position,
        "frequency": _frequency_from_extras(parsed.get("extras")),
        "recorded_at": parsed.get("recorded_at"),
        "event_tag": event_tag,
        "transcript_text": seed_text,
        "transcript_seed": seed_text,
        "transcript_confidence": confidence,
        "expected_verdict": "out_of_scope",
        "expected_section": None,
        "expected_phraseology": None,
        "mismatch": None,
        "notes": "",
        "model_repo": model_repo,
        "human_verified": False,
        "labeled_at": None,
    }


def validate_row(row: dict[str, Any]) -> None:
    """Cross-field validation. Raises ValueError on invariant break.

    Verdict-specific shape:
      ok           → expected_section required; phraseology optional.
      wrong        → section + phraseology + mismatch required.
      incomplete   → section + mismatch required; phraseology optional.
      out_of_scope → section, phraseology, mismatch all None.
    """
    verdict = row.get("expected_verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"expected_verdict must be one of {VERDICTS}, got {verdict!r}")
    speaker = row.get("speaker_role")
    if speaker not in SPEAKERS:
        raise ValueError(f"speaker_role must be one of {SPEAKERS}, got {speaker!r}")

    section = row.get("expected_section")
    phraseology = row.get("expected_phraseology")
    mismatch = row.get("mismatch")

    if verdict == "out_of_scope":
        if section or phraseology or mismatch:
            raise ValueError("out_of_scope rows must leave section / phraseology / mismatch null")
        return
    if not section:
        raise ValueError(f"verdict={verdict} requires expected_section")
    if verdict in {"wrong", "incomplete"} and not mismatch:
        raise ValueError(f"verdict={verdict} requires a mismatch reason")
    if verdict == "wrong" and not phraseology:
        raise ValueError("verdict=wrong requires expected_phraseology (canonical form)")


def stamp_verified(row: dict[str, Any]) -> dict[str, Any]:
    """Mark a row as human-verified at now(). Returns the same dict."""
    row["human_verified"] = True
    row["labeled_at"] = datetime.now(tz=UTC).isoformat()
    return row


def load_existing_indices(utt_path: Path) -> set[int]:
    """Return the set of utt_index values already persisted for a clip.
    Tolerates an absent file (returns empty) and skips malformed lines
    so a partial-write crash doesn't block resume."""
    if not utt_path.exists():
        return set()
    indices: set[int] = set()
    with utt_path.open(encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            try:
                indices.add(int(row["utt_index"]))
            except (KeyError, TypeError, ValueError):
                continue
    return indices


def append_row(utt_path: Path, row: dict[str, Any]) -> None:
    utt_path.parent.mkdir(parents=True, exist_ok=True)
    with utt_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, sort_keys=True, default=str))
        fp.write("\n")


def iter_unlabeled(
    segments: list[dict[str, Any]], skip_indices: set[int]
) -> Iterator[tuple[int, dict[str, Any]]]:
    for idx, segment in enumerate(segments):
        if idx in skip_indices:
            continue
        yield idx, segment


# ---------- audio playback ----------


def play_segment(
    audio_path: Path, start_s: float, duration_s: float
) -> subprocess.Popen[bytes] | None:
    """Spawn ffplay in the background to play one segment. Returns the
    Popen so the caller can kill it on the next play. Returns None if
    ffplay isn't installed (silent fallback so labelling still works
    headless)."""
    if shutil.which("ffplay") is None:
        return None
    return subprocess.Popen(  # noqa: S603 — ffplay on PATH, args fully controlled
        [  # noqa: S607
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{max(duration_s, 0.05):.3f}",
            str(audio_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_player(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------- discovery ----------


def list_clip_ids(target: Path, only: tuple[str, ...] | None = None) -> list[str]:
    """Discover clip ids from meta/*.json, sorted by recorded_at then id."""
    meta_dir = target / "meta"
    if not meta_dir.exists():
        return []
    rows: list[tuple[str, str]] = []
    for path in meta_dir.glob("*.json"):
        with path.open(encoding="utf-8") as fp:
            row = json.load(fp)
        cid = str(row.get("clip_id") or path.stem)
        if only and cid not in only:
            continue
        parsed = row.get("parsed") or {}
        recorded = ""
        if isinstance(parsed, dict):
            value = parsed.get("recorded_at")
            if isinstance(value, str):
                recorded = value
        rows.append((recorded, cid))
    rows.sort()
    return [cid for _, cid in rows]


# ---------- prompt loop ----------


def _render_clip_header(console: Console, meta: dict[str, Any], stt: dict[str, Any]) -> None:
    parsed = meta.get("parsed") or {}
    duration = meta.get("duration_s")
    duration_text = f"{duration:.1f}s" if isinstance(duration, int | float) else "?"
    body = Text()
    body.append(f"file:        {meta.get('source_filename')}\n")
    body.append(
        f"icao:        {parsed.get('icao') or '?'}    "
        f"position:    {parsed.get('position') or '?'}\n"
    )
    body.append(
        f"event:       {parsed.get('event_tag') or '?'}    "
        f"recorded:    {parsed.get('recorded_at') or '?'}\n"
    )
    body.append(
        f"duration:    {duration_text}    "
        f"segments:    {len(stt.get('segments', []))}    "
        f"model:       {stt.get('model_repo') or '?'}"
    )
    console.print(Panel(body, title=f"clip {meta.get('clip_id')}", border_style="cyan"))


def _render_utterance(console: Console, idx: int, total: int, segment: dict[str, Any]) -> None:
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", 0.0) or 0.0)
    conf = segment.get("avg_logprob")
    conf_text = f"{conf:.2f}" if isinstance(conf, int | float) else "?"
    text = str(segment.get("text", "")).strip()
    body = Text()
    body.append(f"span:        {start:.2f}s — {end:.2f}s    conf: {conf_text}\n")
    body.append(f"whisper:     {text}")
    console.print(Panel(body, title=f"utt {idx + 1}/{total}", border_style="green"))


def _prompt_top(console: Console) -> str:
    return Prompt.ask(
        "(o)os  (l)abel  (p)lay  (s)kip-clip  (q)uit",
        choices=list(_TOP_CHOICES),
        default="o",
        console=console,
    )


_VERDICT_KEYS = {"o": "ok", "w": "wrong", "i": "incomplete", "x": "out_of_scope"}
_SPEAKER_KEYS = {"p": "pilot", "c": "controller", "u": "unknown"}


def _prompt_full_label(console: Console, row: dict[str, Any]) -> dict[str, Any] | None:
    """Walk the field-by-field label sequence on a copy of `row`. Returns
    the edited row, or None if the human cancels at the confirm step."""
    work = dict(row)

    seed = work["transcript_seed"]
    work["transcript_text"] = Prompt.ask("transcript", default=seed, console=console).strip()

    speaker_default_letter = next(
        (k for k, v in _SPEAKER_KEYS.items() if v == work["speaker_role"]), "u"
    )
    speaker_letter = Prompt.ask(
        "speaker (p)ilot/(c)ontroller/(u)nknown",
        choices=list(_SPEAKER_KEYS.keys()),
        default=speaker_default_letter,
        console=console,
    )
    work["speaker_role"] = _SPEAKER_KEYS[speaker_letter]

    verdict_default_letter = next(
        (k for k, v in _VERDICT_KEYS.items() if v == work["expected_verdict"]), "x"
    )
    verdict_letter = Prompt.ask(
        "verdict (o)k/(w)rong/(i)ncomplete/(x)out_of_scope",
        choices=list(_VERDICT_KEYS.keys()),
        default=verdict_default_letter,
        console=console,
    )
    work["expected_verdict"] = _VERDICT_KEYS[verdict_letter]

    if work["expected_verdict"] == "out_of_scope":
        work["expected_section"] = None
        work["expected_phraseology"] = None
        work["mismatch"] = None
    else:
        work["expected_section"] = (
            Prompt.ask("section (e.g. 3-9-10)", default="", console=console).strip() or None
        )
        if work["expected_verdict"] in {"ok", "wrong"}:
            work["expected_phraseology"] = (
                Prompt.ask("phraseology (canonical)", default="", console=console).strip() or None
            )
        else:
            work["expected_phraseology"] = None
        if work["expected_verdict"] in {"wrong", "incomplete"}:
            work["mismatch"] = (
                Prompt.ask("mismatch (one-line reason)", default="", console=console).strip()
                or None
            )
        else:
            work["mismatch"] = None

    work["notes"] = Prompt.ask("notes", default=work.get("notes", ""), console=console).strip()

    try:
        validate_row(work)
    except ValueError as exc:
        console.print(f"[red]invalid row:[/red] {exc}")
        if Prompt.ask("retry? (y/n)", choices=["y", "n"], default="y", console=console) == "y":
            return _prompt_full_label(console, row)
        return None

    return stamp_verified(work)


def _label_clip(
    clip_id: str,
    target: Path,
    *,
    auto_play: bool,
    console: Console,
) -> str:
    """Run the label loop for one clip. Returns one of:
    'next'  — clip done, advance to the next clip.
    'quit'  — user asked to save+exit; outer loop should stop.
    """
    meta_path = target / "meta" / f"{clip_id}.json"
    stt_path = target / "stt" / f"{clip_id}.json"
    utt_path = target / "utt" / f"{clip_id}.jsonl"
    if not meta_path.exists():
        console.print(f"[yellow]meta missing for {clip_id}; skipping[/yellow]")
        return "next"
    if not stt_path.exists():
        console.print(f"[yellow]stt missing for {clip_id}; run atc_audio_transcribe first[/yellow]")
        return "next"
    with meta_path.open(encoding="utf-8") as fp:
        meta = json.load(fp)
    with stt_path.open(encoding="utf-8") as fp:
        stt = json.load(fp)

    audio_relpath = meta.get("norm_relpath")
    audio_path = target / str(audio_relpath) if isinstance(audio_relpath, str) else None
    segments = list(stt.get("segments") or [])
    if not segments:
        console.print(f"[yellow]no whisper segments for {clip_id}; skipping[/yellow]")
        return "next"

    skip_indices = load_existing_indices(utt_path)
    pending = list(iter_unlabeled(segments, skip_indices))
    if not pending:
        console.print(f"[dim]{clip_id}: all {len(segments)} utterances already labelled[/dim]")
        return "next"

    _render_clip_header(console, meta, stt)
    console.print(
        f"[dim]{len(pending)}/{len(segments)} unlabeled — resuming from utt "
        f"{pending[0][0] + 1}[/dim]"
    )

    proc: subprocess.Popen[bytes] | None = None
    model_repo = str(stt.get("model_repo") or "?")
    try:
        for idx, segment in pending:
            _render_utterance(console, idx, len(segments), segment)
            seed_row = row_template(
                meta, segment, clip_id=clip_id, utt_index=idx, model_repo=model_repo
            )

            if auto_play and audio_path is not None and audio_path.exists():
                stop_player(proc)
                proc = play_segment(
                    audio_path,
                    seed_row["start_s"],
                    seed_row["end_s"] - seed_row["start_s"],
                )

            while True:
                action = _prompt_top(console)
                if action == "p":
                    if audio_path is None or not audio_path.exists():
                        console.print("[yellow]no audio file to play[/yellow]")
                        continue
                    stop_player(proc)
                    proc = play_segment(
                        audio_path,
                        seed_row["start_s"],
                        seed_row["end_s"] - seed_row["start_s"],
                    )
                    continue
                break

            if action == "o":
                stamp_verified(seed_row)
                append_row(utt_path, seed_row)
                continue
            if action == "l":
                edited = _prompt_full_label(console, seed_row)
                if edited is None:
                    console.print("[dim]cancelled, leaving utterance unlabelled[/dim]")
                    continue
                append_row(utt_path, edited)
                continue
            if action == "s":
                console.print(f"[dim]skipping rest of {clip_id}[/dim]")
                return "next"
            if action == "q":
                return "quit"

        return "next"
    finally:
        stop_player(proc)


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"atc_audio dir (default: {DEFAULT_TARGET.relative_to(REPO)}).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=(),
        metavar="CLIP_ID",
        help="Label only these clips (default: all clips with stt rows).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many clips (0 = no limit).",
    )
    parser.add_argument(
        "--auto-play",
        action="store_true",
        help="Auto-play each utterance via ffplay as it's presented.",
    )
    args = parser.parse_args(argv)

    target: Path = args.target
    only = tuple(args.only)

    clip_ids = list_clip_ids(target, only=only or None)
    if not clip_ids:
        print(
            f"no clips found under {target}/meta — run atc_audio_ingest.py first",
            file=sys.stderr,
        )
        return 1

    if args.limit > 0:
        clip_ids = clip_ids[: args.limit]

    console = Console()
    console.print(f"[bold]labelling {len(clip_ids)} clip(s)[/bold]")
    for cid in clip_ids:
        result = _label_clip(cid, target, auto_play=args.auto_play, console=console)
        if result == "quit":
            console.print("[bold]saved + quit[/bold]")
            return 0
    console.print("[bold]done[/bold]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
