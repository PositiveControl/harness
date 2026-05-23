#!/usr/bin/env bash
# Swap the running vLLM container to a new variant for harness-u70v
# tuning. Uses a known-good fixed set of `docker run` flags rather than
# replicating arbitrary env values from the prior container -- the
# image's entrypoint already sets every CUDA/path env it needs, so we
# only pass what's operationally meaningful.
#
# Args are built as a bash array so values containing spaces don't blow
# up docker's argument parser. The previous version stringified the env
# list and broke on NVIDIA_REQUIRE_CUDA which embeds spaces between
# brand=... tokens.
#
# Usage:
#   bash gx10_swap_variant.sh baseline      # 32B-AWQ + FA2 + Marlin (matches what /metrics saw)
#   bash gx10_swap_variant.sh flashinfer    # 32B-AWQ + FlashInfer attention + autotune
#   bash gx10_swap_variant.sh 14b           # Qwen2.5-Coder-14B-Instruct-AWQ
#   bash gx10_swap_variant.sh 7b            # Qwen2.5-Coder-7B-Instruct-AWQ

set -euo pipefail

VARIANT="${1:-}"
IMAGE="vllm/vllm-openai:latest"
HF_CACHE_HOST="/opt/models/hf"          # already populated; this is where weights live on gx10
HF_CACHE_GUEST="/root/.cache/huggingface"

if [ -z "$VARIANT" ]; then
    echo "usage: $0 {baseline|flashinfer|14b|7b}"
    exit 1
fi

# --- variant config -----------------------------------------------------
EXTRA_ENVS=()
EXTRA_ARGS=()
case "$VARIANT" in
    baseline)
        MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
        ;;
    flashinfer)
        # --flashinfer-autotune is a KernelConfig field in vLLM 0.21, not a
        # serve-time CLI flag (the CLI rejects it as 'unrecognized argument').
        # The big lever is the attention backend env var anyway -- autotune
        # is only a small further win on top of FLASHINFER selection.
        MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
        EXTRA_ENVS+=("-e" "VLLM_ATTENTION_BACKEND=FLASHINFER")
        ;;
    14b)
        MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
        ;;
    7b)
        MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
        ;;
    *)
        echo "unknown variant: $VARIANT"
        exit 2
        ;;
esac

NAME="vllm-harness-$VARIANT"

# --- stop any container currently bound to port 8000 -------------------
# Match by port (most reliable; survives name drift across attempts).
EXISTING=$(docker ps --filter "publish=8000" --format '{{.ID}}')
if [ -n "$EXISTING" ]; then
    for cid in $EXISTING; do
        echo "stopping existing container $cid (publish=8000)"
        docker stop "$cid" >/dev/null
        docker rm -f "$cid" >/dev/null 2>&1 || true
    done
fi

# Clean up any prior failed run of this variant.
docker rm -f "$NAME" >/dev/null 2>&1 || true

# --- build the docker run command as an array --------------------------
DOCKER_ARGS=(
    run -d
    --name "$NAME"
    --gpus all
    --shm-size=16g
    -v "${HF_CACHE_HOST}:${HF_CACHE_GUEST}"
    -p 8000:8000
    -e "VLLM_ENABLE_CUDA_COMPATIBILITY=0"
    "${EXTRA_ENVS[@]}"
    "$IMAGE"
    --model "$MODEL"
    --quantization awq_marlin
    --max-model-len 32768
    --enable-auto-tool-choice --tool-call-parser hermes
    --host 0.0.0.0
    "${EXTRA_ARGS[@]}"
)

echo "starting variant=$VARIANT model=$MODEL"
echo "+ docker ${DOCKER_ARGS[*]}"
NEW_CID=$(docker "${DOCKER_ARGS[@]}")
echo "container: $NEW_CID"

# --- wait for /health=200 ----------------------------------------------
echo "waiting for /health=200 (up to 10 min)..."
deadline=$(( $(date +%s) + 600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "READY at $(date -Iseconds)"
        break
    fi
    sleep 5
done

if ! curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "ERROR: /health never returned 200 within the deadline"
    echo "--- container logs (last 60 lines) ---"
    docker logs --tail 60 "$NEW_CID" 2>&1 || true
    exit 3
fi

# --- print the engine-init banner so we can see which kernels were picked
echo
echo "=== engine init (key lines) ==="
docker logs "$NEW_CID" 2>&1 | grep -iE \
    "version 0|quantization|awq|marlin|attention backend|FlashAttention|FlashInfer|enforce.eager|CUDA graph|KV cache size|Model loading took|Asynchronous scheduling" \
    | head -30
echo
echo "Probe this variant from the Mac:"
echo "  uv run python scripts/bench_vllm_remote.py"
echo
echo "Swap to another variant or restore: bash $0 baseline"
