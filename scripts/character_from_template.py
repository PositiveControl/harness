"""Copy a character template into `character/<new_name>/` with placeholder
substitution.

Non-interactive — for the Phase-2 scaffold. The Phase-1 interactive
`harness character init` command (harness-ouo) will eventually sit on
top of this same template directory.

Usage:

    uv run python scripts/character_from_template.py airton_c1 --from atc

Refuses to overwrite an existing persona directory unless `--force` is
passed. Substitutes `{{CHARACTER_NAME}}` verbatim in every file under
the template.

Templates live in `src/harness/character_templates/<archetype>/`. Add
a new archetype by creating a sibling directory with the same file
layout; the script discovers archetypes by directory listing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = REPO / "src" / "harness" / "character_templates"
CHARACTERS_ROOT = REPO / "character"

PLACEHOLDER = "{{CHARACTER_NAME}}"


def list_templates() -> list[str]:
    if not TEMPLATES_ROOT.exists():
        return []
    return sorted(
        p.name for p in TEMPLATES_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")
    )


def copy_template(template_dir: Path, dest_dir: Path, character_name: str) -> list[Path]:
    """Copy the template tree into dest_dir and substitute CHARACTER_NAME.

    Returns the list of written files (relative to dest_dir). The README
    is intentionally NOT copied — it documents the template, not the
    instance."""

    written: list[Path] = []
    for src in template_dir.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(template_dir)
        # README documents the template itself; skip it for instances.
        if rel.name == "README.md" and rel.parent == Path("."):
            continue
        dest = dest_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        dest.write_text(text.replace(PLACEHOLDER, character_name), encoding="utf-8")
        written.append(rel)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("name", help="New character name (directory slug, e.g. airton_c1).")
    parser.add_argument(
        "--from",
        dest="template",
        required=True,
        help=f"Template archetype name. Available: {', '.join(list_templates()) or '(none)'}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite character/<name>/ if it already exists.",
    )
    args = parser.parse_args(argv)

    template_dir = TEMPLATES_ROOT / args.template
    if not template_dir.exists():
        print(f"template not found: {template_dir}", file=sys.stderr)
        print(f"available: {', '.join(list_templates()) or '(none)'}", file=sys.stderr)
        return 1

    dest_dir = CHARACTERS_ROOT / args.name
    if dest_dir.exists():
        if not args.force:
            print(f"destination exists: {dest_dir} (pass --force to overwrite)", file=sys.stderr)
            return 1
        shutil.rmtree(dest_dir)

    written = copy_template(template_dir, dest_dir, args.name)
    print(f"created character/{args.name}/ from template '{args.template}':")
    for rel in written:
        print(f"  {rel}")
    print()
    print("Next:")
    print(f"  1. Hand-edit character/{args.name}/core.yaml + constitution.md for your scope.")
    print(f"  2. Drop source documents in character/{args.name}/corpus/raw/.")
    print("  3. Run extract → chunk → ingest:")
    print(f"     HARNESS_CHARACTER_NAME={args.name} uv run python scripts/atc_extract.py ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
