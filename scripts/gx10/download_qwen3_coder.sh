#!/usr/bin/env bash
# Pre-stage Qwen3-Coder models into the vLLM container's HF cache.
#
# Runs `hf download` INSIDE the vllm/vllm-openai image so that:
#   - the model lands in the same bind-mounted HF cache the runtime container uses
#     (host /opt/models/hf -> /root/.cache/huggingface in the container)
#   - permissions match (write as root inside the container; host bind handles it)
#   - we don't need huggingface_hub installed on the host
#
# Pulls:
#   Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  (~30 GB)
#   Qwen/Qwen3-Coder-Next-FP8              (~80 GB)
# Total: ~110 GB on the host bind path.
#
# Pre-staging is optional — vllm_swap.sh will pull on first serve if missing —
# but it separates download time from serve time so the swap itself is fast.
set -euo pipefail

HF_CACHE="${VLLM_HF_CACHE:-/opt/models/hf}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

if [[ ! -d "$HF_CACHE" ]]; then
  echo "host cache dir does not exist: $HF_CACHE" >&2
  echo "create it first:" >&2
  echo "  sudo mkdir -p $HF_CACHE && sudo chown root:docker $HF_CACHE" >&2
  exit 1
fi

echo "==> staging into: $HF_CACHE  (bind-mounted into container)"
echo "==> using image:  $IMAGE"
echo "==> free space:"
df -h "$HF_CACHE"
echo

docker run --rm \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -c '
    set -euo pipefail
    if ! command -v hf >/dev/null 2>&1; then
      echo "    installing huggingface_hub[hf_transfer] into ephemeral container"
      pip install -q "huggingface_hub[hf_transfer]"
    fi
    for repo in \
      "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" \
      "Qwen/Qwen3-Coder-Next-FP8"
    do
      echo "==> $repo"
      hf download "$repo"
      echo
    done
    echo "==> on-disk sizes:"
    du -sh /root/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-* 2>/dev/null || true
  '
