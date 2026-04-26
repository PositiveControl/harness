"""Pre-push gate for the voice ablation manifest (harness-wrzz, plan
#8). Validates that `voice/ablation.yaml`'s entries still align with
the current `voice/canonical.yaml` — IDs match, gold + prompt strings
haven't drifted apart from canonical wording.

Fast (no MLX, no embedder, no eval run). Wired into pre-commit's
pre-push stage with a `files:` filter so the hook only runs when one
of the two YAMLs is in the pushed diff.

Exit codes:
    0 — manifest is structurally clean (or no manifest exists).
    1 — at least one structural problem detected. Push blocked.

Slow measurement work (stock vs ablate vs round-robin) lives in
`scripts/ablation_check.py` (separate, manual-invocation script)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.persona.ablation_validate import (  # noqa: E402
    format_problems,
    validate_ablation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--character",
        default="airton_c1",
        help="Character whose voice/ablation.yaml + voice/canonical.yaml to validate.",
    )
    parser.add_argument(
        "--character-path",
        type=Path,
        default=None,
        help="Override character dir. Defaults to character/<character>/.",
    )
    args = parser.parse_args(argv)

    character_path = args.character_path or REPO_ROOT / "character" / args.character
    canonical_path = character_path / "voice" / "canonical.yaml"
    ablation_path = character_path / "voice" / "ablation.yaml"

    if not canonical_path.exists():
        sys.stderr.write(
            f"canonical.yaml missing at {canonical_path} — character has no "
            "voice samples; ablation gate is a no-op.\n"
        )
        return 0

    problems = validate_ablation(
        canonical_path=canonical_path,
        ablation_path=ablation_path,
    )
    if not problems:
        if ablation_path.exists():
            sys.stdout.write(
                f"ablation manifest clean: {ablation_path.name} aligned with "
                f"{canonical_path.name}\n"
            )
        return 0

    sys.stderr.write(format_problems(problems) + "\n")
    sys.stderr.write(
        f"\n{len(problems)} structural problem(s) detected. Fix the "
        "manifest or canonical entries before pushing.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
