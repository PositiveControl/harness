#!/usr/bin/env bash
# Pre-push gate for airton_c1 retrieval quality (harness-zxqs).
#
# Runs `harness eval atc-retrieval --compare-baseline` so a push that
# touches retrieval-affecting paths can't ship a recall@k regression
# silently. Wired via .pre-commit-config.yaml — pre-commit's `files:`
# filter decides whether the hook fires; this script does the actual
# work once it does.
#
# Env locked to airton_c1 + offline mode so the hook stays
# deterministic across machines and doesn't hit the HF Hub on every
# push. The comparator itself is fast (~seconds with a warm embedder).
#
# Exit codes:
#   0 — gate passed (or fixture/baseline unavailable in a way the
#       harness CLI handles gracefully).
#   1 — recall regression caught; push blocked.
#
# Bypass: re-snapshot with `harness eval atc-retrieval --save-baseline`
# once the new state is intentional, then re-push.

set -euo pipefail

exec env HARNESS_CHARACTER_NAME=airton_c1 \
         HF_HUB_OFFLINE=1 \
         TRANSFORMERS_OFFLINE=1 \
         uv run harness eval atc-retrieval --compare-baseline
