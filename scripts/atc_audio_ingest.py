"""ATC audio ingest: filename parse + ffmpeg normalize + meta write (harness-d8o2).

Walks a source directory of LiveATC-style mp3/m4a/wav clips, parses the
metadata embedded in each filename (event tag, ICAO, position, optional
frequency, recorded-at UTC, trailing comment), normalises each clip to
16k mono FLAC via ffmpeg, and lands the artefacts under
`character/airton_c1/atc_audio/`:

    raw/   <original-filename>.mp3   (copy of source — audit + replay)
    norm/  <clip_id>.flac            (16k mono FLAC, what STT consumes)
    meta/  <clip_id>.json            (filename parse + provenance + duration)
    index.jsonl                       (rebuilt each run from meta/*.json)

clip_id = sha256(source-bytes)[:12]. Idempotent on clip_id: re-running is
a no-op unless --force is set. Files whose names don't fit the dominant
shape (`<event_tag><ICAO>[<src#>]-<position>...-Mon-DD-YYYY-HHMMZ`) are
still ingested; their parse rows just carry `parse_failed=true` so the
labelling step can sort them out instead of dropping them silently.

Usage (from repo root):
    uv run python scripts/atc_audio_ingest.py
    uv run python scripts/atc_audio_ingest.py --source /some/where
    uv run python scripts/atc_audio_ingest.py --dry-run
    uv run python scripts/atc_audio_ingest.py --force

ffmpeg is required on PATH for real runs. --dry-run only parses
filenames so it works without ffmpeg installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "character" / "airton_c1" / "atc_audio"
DEFAULT_SOURCE = REPO.parent / "atc-audio"

AUDIO_EXTS = frozenset({".mp3", ".m4a", ".wav", ".flac"})
NORM_SAMPLE_RATE = 16_000
NORM_CHANNELS = 1

_MONTHS = {
    name: idx
    for idx, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}

_DUP_SUFFIX = re.compile(r"^(?P<base>.*?) \((?P<n>\d+)\)$")

_DATE_TAIL = re.compile(
    r"^(?P<head>.*?)"
    r"-?"
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"-(?P<day>\d{1,2})"
    r"-(?P<year>\d{4})"
    r"-(?P<hhmm>\d{4})Z"
    r"(?P<comment>.*)$"
)

# ICAO: K (US contiguous), C (Canada), T (Caribbean US territories) are
# the prefixes that occur in this corpus. Match candidates anywhere in
# the pre-first-hyphen segment, then keep the *rightmost* — that's the
# canonical disambiguator when an event tag contains a stray K/C/T (e.g.
# `TFRKPBI2` would otherwise pick `TFRK` before reaching `KPBI`). The
# zero-width lookahead is required because re.finditer is non-overlapping
# — `TFRK`'s consume would skip past `KPBI` with a plain pattern. The
# trailing alnum suffix between the ICAO and the first `-` (LiveATC
# source-feed numbering — `KAVQ2`, `KORD1N2`) is captured as `icao_source`.
_ICAO_RE = re.compile(r"(?=(?P<icao>[KCT][A-Z]{3}))")


@dataclass(frozen=True)
class Parsed:
    event_tag: str | None
    icao: str | None
    icao_source: str | None
    position: str | None
    extras: list[str] = field(default_factory=list)
    recorded_at: str | None = None
    comment: str | None = None
    duplicate_index: int = 0
    parse_failed: bool = False
    raw_stem: str = ""


def _strip_dup_suffix(stem: str) -> tuple[str, int]:
    match = _DUP_SUFFIX.match(stem)
    if match is None:
        return stem, 0
    return match.group("base"), int(match.group("n"))


def _split_extras(rest: str) -> tuple[str | None, list[str]]:
    """Split the post-ICAO tail into a position string + numeric extras.

    Tokens that are pure digits (5-6 digit frequency-like numbers, runway
    pairs like `0826`) get peeled off into `extras`; everything else
    rejoins with `-` as the position string. The split is intentionally
    lossy — labelling fixes ambiguous cases (e.g. is `0826` runway 8/26
    or freq 082.6?). v0 only needs the structure stable."""
    if not rest:
        return None, []
    parts = [p for p in rest.split("-") if p]
    extras = [p for p in parts if p.isdigit()]
    pos = [p for p in parts if not p.isdigit()]
    return ("-".join(pos) or None), extras


def parse_filename(name: str) -> Parsed:
    """Parse a LiveATC-style audio filename into structured metadata.

    See module docstring for the dominant shape. Returns
    `Parsed(parse_failed=True)` for filenames that don't carry the
    `Mon-DD-YYYY-HHMMZ` tail — those still get ingested but go to the
    labelling step's messy bucket."""
    stem = Path(name).stem
    base, dup_idx = _strip_dup_suffix(stem)

    match = _DATE_TAIL.match(base)
    if match is None:
        return Parsed(
            event_tag=None,
            icao=None,
            icao_source=None,
            position=None,
            duplicate_index=dup_idx,
            parse_failed=True,
            raw_stem=base,
        )

    head = match.group("head").rstrip("-").strip()
    comment_raw = match.group("comment")
    comment = comment_raw.lstrip(" -").rstrip() or None

    recorded_at = datetime(
        year=int(match.group("year")),
        month=_MONTHS[match.group("month")],
        day=int(match.group("day")),
        hour=int(match.group("hhmm")[:2]),
        minute=int(match.group("hhmm")[2:]),
        tzinfo=UTC,
    ).isoformat()

    first_hyphen = head.find("-")
    segment = head if first_hyphen == -1 else head[:first_hyphen]
    candidates = list(_ICAO_RE.finditer(segment))
    if not candidates:
        return Parsed(
            event_tag=head or None,
            icao=None,
            icao_source=None,
            position=None,
            recorded_at=recorded_at,
            comment=comment,
            duplicate_index=dup_idx,
            raw_stem=base,
        )

    best = candidates[-1]
    icao = best.group("icao")
    icao_end_in_segment = best.start() + len(icao)
    src_suffix = segment[icao_end_in_segment:]
    event_tag = segment[: best.start()].rstrip("-").strip() or None
    icao_source = src_suffix or None
    rest_start = first_hyphen if first_hyphen != -1 else len(head)
    rest = head[rest_start:].lstrip("-").strip()
    position, extras = _split_extras(rest)

    return Parsed(
        event_tag=event_tag,
        icao=icao,
        icao_source=icao_source,
        position=position,
        extras=extras,
        recorded_at=recorded_at,
        comment=comment,
        duplicate_index=dup_idx,
        raw_stem=base,
    )


def hash_source_bytes(path: Path) -> str:
    """sha256 of the source file. clip_id = first 12 hex chars."""
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_id_for(path: Path) -> str:
    return hash_source_bytes(path)[:12]


def ffmpeg_normalize(src: Path, dst: Path) -> None:
    """Encode `src` to 16k mono FLAC at `dst`. Overwrites."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 — ffmpeg on PATH, args fully controlled
        [  # noqa: S607 — partial path is intentional; ffmpeg discovered via PATH
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            str(NORM_CHANNELS),
            "-ar",
            str(NORM_SAMPLE_RATE),
            "-c:a",
            "flac",
            str(dst),
        ],
        check=True,
    )


def ffprobe_duration(path: Path) -> float | None:
    """Best-effort duration probe via ffprobe. Returns None on failure
    so a duration glitch never blocks the meta write."""
    try:
        out = subprocess.run(  # noqa: S603 — ffprobe on PATH, args fully controlled
            [  # noqa: S607 — partial path is intentional; ffprobe discovered via PATH
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    raw = out.stdout.strip()
    try:
        return float(raw)
    except ValueError:
        return None


def ingest_clip(
    src: Path,
    target_dir: Path,
    *,
    force: bool = False,
    copy_raw: bool = True,
) -> tuple[dict[str, object], bool]:
    """Ingest one source audio file. Returns `(meta, was_new)`.

    Idempotent on clip_id: if `meta/<clip_id>.json` exists and `force` is
    False, returns the persisted row unchanged with `was_new=False`."""
    cid = clip_id_for(src)
    meta_dir = target_dir / "meta"
    meta_path = meta_dir / f"{cid}.json"

    if meta_path.exists() and not force:
        with meta_path.open(encoding="utf-8") as fp:
            return json.load(fp), False

    raw_dir = target_dir / "raw"
    norm_dir = target_dir / "norm"
    norm_path = norm_dir / f"{cid}.flac"

    raw_relpath: str | None = None
    if copy_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_dst = raw_dir / src.name
        if not raw_dst.exists() or force:
            shutil.copy2(src, raw_dst)
        raw_relpath = raw_dst.relative_to(target_dir).as_posix()

    ffmpeg_normalize(src, norm_path)
    duration_s = ffprobe_duration(norm_path)

    parsed = parse_filename(src.name)

    meta: dict[str, object] = {
        "clip_id": cid,
        "source_filename": src.name,
        "source_path": str(src),
        "raw_relpath": raw_relpath,
        "norm_relpath": norm_path.relative_to(target_dir).as_posix(),
        "norm_format": "flac",
        "norm_sample_rate": NORM_SAMPLE_RATE,
        "norm_channels": NORM_CHANNELS,
        "duration_s": duration_s,
        "ingested_at": datetime.now(tz=UTC).isoformat(),
        "parsed": dataclasses.asdict(parsed),
    }

    meta_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, sort_keys=True)
        fp.write("\n")

    return meta, True


def rebuild_index(target_dir: Path) -> int:
    """Rebuild `index.jsonl` from `meta/*.json`. Returns row count.

    Sort key: (recorded_at, clip_id). Parse-failed rows have no
    recorded_at; they sort to the top of the file under the empty-string
    key, which is fine for v0 (the labelling step will surface them)."""
    meta_dir = target_dir / "meta"
    if not meta_dir.exists():
        return 0
    rows: list[dict[str, object]] = []
    for path in sorted(meta_dir.glob("*.json")):
        with path.open(encoding="utf-8") as fp:
            rows.append(json.load(fp))

    def _sort_key(row: dict[str, object]) -> tuple[str, str]:
        parsed = row.get("parsed")
        recorded = ""
        if isinstance(parsed, dict):
            recorded_value = parsed.get("recorded_at")
            if isinstance(recorded_value, str):
                recorded = recorded_value
        return recorded, str(row.get("clip_id", ""))

    rows.sort(key=_sort_key)
    out_path = target_dir / "index.jsonl"
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _list_sources(source_dir: Path) -> list[Path]:
    """List audio files in `source_dir`, sorted so canonical filenames
    come before their ` (N)` duplicates. Otherwise alphabetical sort
    would land `DAL1082 (1).mp3` before `DAL1082.mp3` (space < `.` in
    ASCII), and the dup-suffixed name would win idempotency on a
    byte-identical pair — visually wrong even though the audio is the
    same."""

    def key(path: Path) -> tuple[int, str]:
        _, dup_idx = _strip_dup_suffix(path.stem)
        return dup_idx, path.name

    return sorted(
        (p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
        key=key,
    )


def _print_dry_run(sources: list[Path]) -> int:
    print(f"[dry-run] {len(sources)} source files")
    parsed_ok = 0
    parsed_failed = 0
    for src in sources:
        parsed = parse_filename(src.name)
        if parsed.parse_failed:
            parsed_failed += 1
            print(f"  ! {src.name}  (parse_failed)")
            continue
        parsed_ok += 1
        print(
            f"  · {src.name}  → "
            f"event={parsed.event_tag} icao={parsed.icao} pos={parsed.position} "
            f"at={parsed.recorded_at}"
        )
    print(f"\n[dry-run] parsed: {parsed_ok}, parse_failed: {parsed_failed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Source directory of audio files (default: ../atc-audio).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Target dir (default: {DEFAULT_TARGET.relative_to(REPO)}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest clips whose meta/<clip_id>.json already exists.",
    )
    parser.add_argument(
        "--no-copy-raw",
        action="store_true",
        help="Skip copying source files into target/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse filenames only; don't run ffmpeg or write artefacts.",
    )
    args = parser.parse_args(argv)

    source_dir: Path = args.source
    target_dir: Path = args.target

    if not source_dir.exists():
        print(f"source dir not found: {source_dir}", file=sys.stderr)
        return 1

    sources = _list_sources(source_dir)
    if not sources:
        print(f"no audio files under {source_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        return _print_dry_run(sources)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH (required for real ingest)", file=sys.stderr)
        return 1

    new_count = 0
    existing_count = 0
    failed_count = 0
    for src in sources:
        try:
            _, was_new = ingest_clip(
                src,
                target_dir,
                force=args.force,
                copy_raw=not args.no_copy_raw,
            )
        except subprocess.CalledProcessError as exc:
            failed_count += 1
            print(f"  ! {src.name}  (ffmpeg failed: {exc})", file=sys.stderr)
            continue
        except OSError as exc:
            failed_count += 1
            print(f"  ! {src.name}  ({exc})", file=sys.stderr)
            continue
        if was_new:
            new_count += 1
            print(f"  + {src.name}")
        else:
            existing_count += 1
            print(f"  · {src.name}  (already ingested)")

    indexed = rebuild_index(target_dir)
    print(f"\ningested: {new_count} new, {existing_count} already present, {failed_count} failed")
    print(f"index.jsonl: {indexed} rows")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
