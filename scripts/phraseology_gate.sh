#!/usr/bin/env bash
# Pre-push gate for airton_c1 phraseology lint quality (harness-15dy).
#
# Runs `harness eval phraseology --compare-baseline` so a push that
# touches the lint tool, the eval fixture, or the eval scorer can't
# ship a verdict / citation accuracy regression silently. Wired via
# .pre-commit-config.yaml — pre-commit's `files:` filter decides
# whether the hook fires; this script does the actual work once it
# does. The corpus paths (character/airton_c1/corpus/*) are already
# gated by atc-retrieval; this gate covers the lint-side surface that
# atc-retrieval doesn't see.
#
# Env locked to airton_c1 + offline mode so the hook stays
# deterministic across machines and doesn't hit the HF Hub on every
# push. The eval is one MLX-pass per fixture case (~40 cases × ~5s
# at temperature 0); slower than the retrieval gate but still
# bounded.
#
# Exit codes:
#   0 — gate passed (or baseline absent — emit a hint, exit 0 so
#       brand-new clones don't block).
#   1 — verdict / citation accuracy regression caught; push blocked.
#
# Bypass: re-snapshot with `harness eval phraseology --save-baseline`
# once the new state is intentional, then re-push.

set -euo pipefail

BASELINE_PATH="character/airton_c1/phraseology_baseline.json"

if [[ ! -f "$BASELINE_PATH" ]]; then
    echo "[phraseology gate] no baseline at $BASELINE_PATH — skipping." >&2
    echo "[phraseology gate] snapshot one with:" >&2
    echo "  HARNESS_CHARACTER_NAME=airton_c1 uv run harness eval phraseology --save-baseline" >&2
    exit 0
fi

exec env HARNESS_CHARACTER_NAME=airton_c1 \
         HF_HUB_OFFLINE=1 \
         TRANSFORMERS_OFFLINE=1 \
         uv run harness eval phraseology --compare-baseline
