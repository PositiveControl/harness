#!/usr/bin/env bash
# airton_f chat shortcut — scholar reading the per-character doc tree.
#
# Pins:
#   HARNESS_CHARACTER_NAME=airton_f               — load scholar core/constitution
#   --tool-set scholar                            — contract + memory + introspect
#                                                    + search_web + fetch_url
#                                                    (allowlisted to scholar.google.com,
#                                                    arxiv.org, doi.org via core.yaml)
#   --persona --tools                             — voice on, tool loop on
#   --router                                      — small-model intent router fronts the loop
#
# The forced assemble_context call (require_assemble_context=true)
# pulls section-shaped hits from character/airton_f/data/document_tree.sqlite,
# which is built at session start from the markdown listed under
# document_trees: in core.yaml. External lookup is auxiliary — the
# corpus is the default authority.
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

# Workspace pinned to the per-character dir so the chat header
# shows the correct sandbox. The scholar profile doesn't include
# workspace-bound fs tools (read_file/write_file/shell), but the
# header is wrong-looking when --workspace defaults to the repo
# root. Override via HARNESS_AIRTON_F_WORKSPACE if you want a
# different sandbox.
NOTES_DIR="${HARNESS_AIRTON_F_WORKSPACE:-$PWD/character/airton_f/workspace}"
if [[ ! -d "$NOTES_DIR" ]]; then
    echo "airton_f: workspace not found at $NOTES_DIR" >&2
    echo "  set HARNESS_AIRTON_F_WORKSPACE=<path> or create the directory." >&2
    exit 1
fi

exec uv run harness chat \
    --model mlx \
    --persona \
    --tools \
    --tool-set scholar \
    --router \
    --workspace "$NOTES_DIR" \
    --memories 3 \
    --facts 5 \
    "$@"
