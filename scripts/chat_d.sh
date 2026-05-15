#!/usr/bin/env bash
# airton_d chat shortcut — notes curator over the per-character workspace.
#
# Pins:
#   HARNESS_CHARACTER_NAME=airton_d              — load the notes core/constitution
#   --workspace character/airton_d/workspace     — sandbox fs tools here
#   --tool-set notes                             — fs read/write + memory, no shell/git/web
#   --persona --tools                            — voice on, tool loop on
#   --router                                     — small-model intent router fronts the loop
#
# Override the workspace location via HARNESS_AIRTON_D_WORKSPACE=<path>.
#
# Pass extra args through:
#   scripts/chat_d.sh --tools-drop write_file       # read-only browse
#   scripts/chat_d.sh --model-repo mlx-community/Qwen2.5-32B-Instruct-4bit
#   scripts/chat_d.sh --tui                         # Textual app

set -euo pipefail
cd "$(dirname "$0")/.."

# Default to offline HF (embedder + model are cached). Override with
# HF_HUB_OFFLINE=0 to refresh.
: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

export HARNESS_CHARACTER_NAME=airton_d

NOTES_DIR="${HARNESS_AIRTON_D_WORKSPACE:-$PWD/character/airton_d/workspace}"
if [[ ! -d "$NOTES_DIR" ]]; then
    echo "airton_d: workspace not found at $NOTES_DIR" >&2
    echo "  set HARNESS_AIRTON_D_WORKSPACE=<path> or create the directory." >&2
    exit 1
fi

exec uv run harness chat \
    --model mlx \
    --persona \
    --tools \
    --tool-set notes \
    --router \
    --workspace "$NOTES_DIR" \
    --memories 3 \
    --facts 5 \
    "$@"
