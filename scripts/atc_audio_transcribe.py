"""ATC audio transcribe: mlx-whisper batch (harness-xko1).

Second leg of the harness-o92o pipeline. Walks the meta/ output of
`atc_audio_ingest.py`, finds the matching norm/<clip_id>.flac file,
runs mlx-whisper over it with word-level timestamps, and writes the
full output (segments + words + per-segment confidence stats) to
stt/<clip_id>.json.

Idempotency: stt/<clip_id>.json gates re-runs. Re-running is a no-op
unless --force is set. Switching models with --model is not detected
automatically — pass --force when changing models if you want to
overwrite previous transcripts.

Default model is `mlx-community/whisper-large-v3-mlx` (best accuracy
on aviation radio audio in this corpus's class). Override with
--model for tiny/base/turbo runs during iteration.

Usage (from repo root, requires `--extra asr`):
    uv sync --extra all --extra asr
    uv run python scripts/atc_audio_transcribe.py
    uv run python scripts/atc_audio_transcribe.py --model mlx-community/whisper-tiny
    uv run python scripts/atc_audio_transcribe.py --only 04dd0a5b26d4 --force
    uv run python scripts/atc_audio_transcribe.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import types

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "character" / "airton_c1" / "atc_audio"
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def _load_mlx_whisper() -> types.ModuleType:
    """Import mlx_whisper lazily so dry-run / tests work without the
    `--extra asr` install. Raises with a clear hint if missing."""
    try:
        import mlx_whisper
    except ImportError as exc:
        raise SystemExit(
            "mlx-whisper not installed. Run `uv sync --extra all --extra asr` and retry."
        ) from exc
    return mlx_whisper  # type: ignore[no-any-return]


def transcribe_clip(audio_path: Path, model_repo: str) -> dict[str, Any]:
    """Run mlx-whisper over `audio_path`. Returns the raw mlx_whisper
    result dict (text + segments[] + language). Word timestamps are
    on so the labelling step can split on word boundaries."""
    mlx_whisper = _load_mlx_whisper()
    return mlx_whisper.transcribe(  # type: ignore[no-any-return]
        str(audio_path),
        path_or_hf_repo=model_repo,
        word_timestamps=True,
    )


def _envelope(
    clip_id: str,
    audio_path: Path,
    meta: dict[str, Any],
    raw: dict[str, Any],
    model_repo: str,
) -> dict[str, Any]:
    """Wrap the mlx-whisper result with provenance + cross-references
    so consumers can join stt rows back to the meta/index without
    needing the audio file."""
    return {
        "clip_id": clip_id,
        "source_filename": meta.get("source_filename"),
        "norm_relpath": meta.get("norm_relpath"),
        "audio_path": str(audio_path),
        "model_repo": model_repo,
        "transcribed_at": datetime.now(tz=UTC).isoformat(),
        "language": raw.get("language"),
        "text": raw.get("text"),
        "segments": raw.get("segments", []),
    }


def transcribe_one(
    clip_id: str,
    target_dir: Path,
    model_repo: str,
    *,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Transcribe one clip by id. Returns `(envelope, was_new)`.

    `was_new=False` indicates an idempotent skip (the existing stt row
    is loaded from disk and returned unchanged)."""
    meta_path = target_dir / "meta" / f"{clip_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta missing for clip_id={clip_id}: {meta_path}")
    with meta_path.open(encoding="utf-8") as fp:
        meta = json.load(fp)

    stt_dir = target_dir / "stt"
    stt_path = stt_dir / f"{clip_id}.json"
    if stt_path.exists() and not force:
        with stt_path.open(encoding="utf-8") as fp:
            return json.load(fp), False

    norm_relpath = meta.get("norm_relpath")
    if not isinstance(norm_relpath, str):
        raise ValueError(f"meta {meta_path} missing norm_relpath")
    audio_path = target_dir / norm_relpath
    if not audio_path.exists():
        raise FileNotFoundError(f"normalized audio missing: {audio_path}")

    raw = transcribe_clip(audio_path, model_repo)
    envelope = _envelope(clip_id, audio_path, meta, raw, model_repo)

    stt_dir.mkdir(parents=True, exist_ok=True)
    with stt_path.open("w", encoding="utf-8") as fp:
        json.dump(envelope, fp, indent=2, sort_keys=True, default=str)
        fp.write("\n")

    return envelope, True


def _list_clip_ids(target_dir: Path, only: tuple[str, ...] | None = None) -> list[str]:
    """Discover clip ids from meta/*.json. Sorted by recorded_at then
    clip_id so progress output reads chronologically."""
    meta_dir = target_dir / "meta"
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
        recorded_at = ""
        if isinstance(parsed, dict):
            value = parsed.get("recorded_at")
            if isinstance(value, str):
                recorded_at = value
        rows.append((recorded_at, cid))
    rows.sort()
    return [cid for _, cid in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"atc_audio dir (default: {DEFAULT_TARGET.relative_to(REPO)}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"mlx-whisper HF repo (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=(),
        metavar="CLIP_ID",
        help="Only transcribe these clip ids (default: every clip in meta/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many clips (0 = no limit). Useful for smoke runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe clips whose stt/<clip_id>.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the clips that would be transcribed; don't load the model.",
    )
    args = parser.parse_args(argv)

    target_dir: Path = args.target
    only = tuple(args.only)

    clip_ids = _list_clip_ids(target_dir, only=only or None)
    if not clip_ids:
        print(
            f"no clips found under {target_dir}/meta — run atc_audio_ingest.py first",
            file=sys.stderr,
        )
        return 1

    if args.limit > 0:
        clip_ids = clip_ids[: args.limit]

    if args.dry_run:
        print(f"[dry-run] {len(clip_ids)} clips would be transcribed with model={args.model}")
        for cid in clip_ids:
            stt_path = target_dir / "stt" / f"{cid}.json"
            tag = "skip" if stt_path.exists() and not args.force else "run "
            print(f"  [{tag}] {cid}")
        return 0

    new_count = 0
    existing_count = 0
    failed_count = 0
    print(f"model: {args.model}")
    print(f"clips: {len(clip_ids)}")
    for cid in clip_ids:
        try:
            _, was_new = transcribe_one(cid, target_dir, args.model, force=args.force)
        except (FileNotFoundError, ValueError) as exc:
            failed_count += 1
            print(f"  ! {cid}  ({exc})", file=sys.stderr)
            continue
        if was_new:
            new_count += 1
            print(f"  + {cid}")
        else:
            existing_count += 1
            print(f"  · {cid}  (already transcribed)")

    print(
        f"\ntranscribed: {new_count} new, {existing_count} already present, {failed_count} failed"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
