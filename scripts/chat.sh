#!/usr/bin/env bash
# Full-stack chat shortcut: MLX + persona + retrieval + tools (full
# profile: every built-in) + router + TUI. Pass extra args through —
# prune tools with --tools-drop, tighten retrieval with --memories 0,
# or override the model with --model-repo.
#
# Examples:
#   scripts/chat.sh
#   scripts/chat.sh --tools-drop shell,write_file
#   scripts/chat.sh --model-repo mlx-community/Qwen2.5-32B-Instruct-4bit
#   scripts/chat.sh --tool-set coding          # override the full profile
#
# Paired with the `full` tool-set in src/harness/tools/profiles.py so
# --tools-drop is the natural prune-down path (harness-1nu).

set -euo pipefail
cd "$(dirname "$0")/.."

# Default to offline HF: the embedder + model are already cached, and a
# flaky DNS / offline network makes sentence-transformers hang on HEAD
# requests to huggingface.co. Override with `HF_HUB_OFFLINE=0` to refresh.
: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

exec uv run harness chat \
    --model mlx \
    --persona \
    --tools \
    --tool-set full \
    --router \
    --tui \
    --summarize-tool-results \
    --memories 3 \
    --facts 5 \
    "$@"
