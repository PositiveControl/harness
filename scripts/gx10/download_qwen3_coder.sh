#!/usr/bin/env bash
# Download Qwen3-Coder models for vLLM serving on the GX10.
#
# Pulls:
#   Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  (~30 GB)
#   Qwen/Qwen3-Coder-Next-FP8              (~80 GB)
# Total: ~110 GB. Stored under $HF_HOME (default ~/.cache/huggingface).
#
# Prereqs:
#   pip install --upgrade 'huggingface_hub[hf_transfer]'
#   (optionally: hf auth login   # if any repo is gated)
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
  echo "missing 'hf' CLI. install with:" >&2
  echo "  pip install --upgrade 'huggingface_hub[hf_transfer]'" >&2
  exit 1
fi

export HF_HUB_ENABLE_HF_TRANSFER=1   # 5-10x faster downloads

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo "==> cache dir: $CACHE_DIR"
echo

for repo in \
  "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" \
  "Qwen/Qwen3-Coder-Next-FP8"
do
  echo "==> $repo"
  hf download "$repo"
  echo
done

echo "done. cached:"
du -sh "$CACHE_DIR"/models--Qwen--Qwen3-Coder-* 2>/dev/null || true
