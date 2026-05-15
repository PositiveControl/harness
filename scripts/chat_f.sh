#!/usr/bin/env bash
# airton_f chat shortcut — scholar reading the per-character doc tree.
#
# Pins:
#   HARNESS_CHARACTER_NAME=airton_f               — load scholar core/constitution
#   --tool-set contract                           — assemble_context + memory + introspect
#   --persona --tools                             — voice on, tool loop on
#   --router                                      — small-model intent router fronts the loop
#
# The forced assemble_context call (require_assemble_context=true)
# pulls section-shaped hits from character/airton_f/data/document_tree.sqlite,
# which is built at session start from the markdown listed under
# document_trees: in core.yaml.
#
# Pass extra args through:
#   scripts/chat_f.sh --tools-add read_file       # read raw markdown alongside the contract
#   scripts/chat_f.sh --tui                       # Textual app

set -euo pipefail
cd "$(dirname "$0")/.."

# Default to offline HF (embedder + model are cached).
: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

export HARNESS_CHARACTER_NAME=airton_f

exec uv run harness chat \
    --model mlx \
    --persona \
    --tools \
    --tool-set contract \
    --router \
    --memories 3 \
    --facts 5 \
    "$@"
