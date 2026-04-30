#!/usr/bin/env bash
# Pre-push gate for airton_c1 atc-audio eval (harness-gy5z).
#
# Runs `harness eval atc-audio --compare-baseline` so a push that
# touches the lint pipeline, the eval scorer, the transcribe/label
# pipeline, or the labelled fixture can't ship a verdict / citation /
# WER regression silently. Mirrors scripts/phraseology_gate.sh.
#
# Three exit modes:
#   0 — gate passed, OR no baseline yet, OR no labelled utterances yet.
#       (Brand-new clones + the period before labels exist must not
#       block pushes; the eval CLI exits 0 when utt/ is empty, and
#       this gate skips early when the baseline is absent.)
#   1 — verdict / citation / WER regression caught; push blocked.
#
# Bypass: re-snapshot with `--save-baseline` once the new state is
# intentional, then re-push.
#
# Env locked to airton_c1 + offline mode so the hook stays
# deterministic across machines (no HF Hub on every push).

set -euo pipefail

BASELINE_PATH="character/airton_c1/atc_audio_baseline.json"
UTT_DIR="character/airton_c1/atc_audio/utt"

if [[ ! -f "$BASELINE_PATH" ]]; then
    echo "[atc-audio gate] no baseline at $BASELINE_PATH — skipping." >&2
    echo "[atc-audio gate] snapshot one with:" >&2
    echo "  HARNESS_CHARACTER_NAME=airton_c1 uv run harness eval atc-audio --save-baseline" >&2
    exit 0
fi

if [[ ! -d "$UTT_DIR" ]] || ! compgen -G "$UTT_DIR/*.jsonl" >/dev/null; then
    echo "[atc-audio gate] no labelled utterances under $UTT_DIR — skipping." >&2
    echo "[atc-audio gate] label some clips with:" >&2
    echo "  uv run python scripts/atc_audio_label.py" >&2
    exit 0
fi

exec env HARNESS_CHARACTER_NAME=airton_c1 \
         HF_HUB_OFFLINE=1 \
         TRANSFORMERS_OFFLINE=1 \
         uv run harness eval atc-audio --compare-baseline
