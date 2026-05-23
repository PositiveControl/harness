#!/usr/bin/env bash
# Start / stop / swap the dockerized vLLM container on the GX10.
#
# This drives the existing `vllm-harness-flashinfer` container (image
# vllm/vllm-openai:latest, GPUs all, HF cache bind-mounted from /opt/models/hf).
# Each swap stops + removes the old container and `docker run`s a fresh one
# with the new --model argument.
#
# Usage:
#   ./vllm_swap.sh 30b      # serve Qwen3-Coder-30B-A3B-Instruct-FP8
#   ./vllm_swap.sh next     # serve Qwen3-Coder-Next-FP8 (80B MoE)
#   ./vllm_swap.sh stop     # stop and remove the container
#   ./vllm_swap.sh status   # show container state + /v1/models
#
# Env vars (all optional):
#   VLLM_PORT          host port to bind         (default 8000)
#   VLLM_CONTAINER     container name            (default vllm-harness-flashinfer)
#   VLLM_IMAGE         image                     (default vllm/vllm-openai:latest)
#   VLLM_HF_CACHE      host HF cache bind path   (default /opt/models/hf)
#   VLLM_MAX_LEN       --max-model-len           (default 131072)
#   VLLM_GPU_UTIL      --gpu-memory-utilization  (default 0.92)
set -euo pipefail

PORT="${VLLM_PORT:-8000}"
CONTAINER="${VLLM_CONTAINER:-vllm-harness-flashinfer}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE="${VLLM_HF_CACHE:-/opt/models/hf}"
MAX_LEN="${VLLM_MAX_LEN:-131072}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"

case "${1:-}" in
  30b)
    MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    ;;
  next)
    MODEL="Qwen/Qwen3-Coder-Next-FP8"
    ;;
  stop)
    MODEL=""
    ;;
  status)
    echo "==> container:"
    docker ps -a --filter "name=^${CONTAINER}$" \
      --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}' || true
    echo
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "==> /v1/models:"
      curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool
    else
      echo "==> /v1/models: not responding on :${PORT}"
    fi
    exit 0
    ;;
  *)
    echo "usage: $0 {30b|next|stop|status}" >&2
    exit 1
    ;;
esac

stop_container() {
  if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "    docker stop $CONTAINER"
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
    echo "    docker rm $CONTAINER"
    docker rm "$CONTAINER" >/dev/null 2>&1 || true
  else
    echo "    (no container named $CONTAINER)"
  fi
}

echo "==> stopping vLLM container"
stop_container

if [[ -z "$MODEL" ]]; then
  echo "==> stopped."
  exit 0
fi

echo "==> starting vLLM container"
echo "    name:     $CONTAINER"
echo "    image:    $IMAGE"
echo "    model:    $MODEL"
echo "    port:     $PORT"
echo "    ctx:      $MAX_LEN"
echo "    cache:    $HF_CACHE -> /root/.cache/huggingface"
echo "    gpu_util: $GPU_UTIL"

# Notes on flags vs. the prior Qwen2.5 Coder AWQ container:
#  - dropped --quantization awq_marlin (Qwen3 ships FP8; vLLM auto-detects)
#  - bumped --max-model-len 32768 -> 131072 (Qwen3 supports 256k, 128k is
#    a sensible default that leaves KV-cache headroom)
#  - added --enable-prefix-caching (big win for the harness's repeated
#    system prompt + retrieval blocks across turns)
#  - kept --enable-auto-tool-choice + --tool-call-parser hermes
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "$IMAGE" \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  >/dev/null

# Cold-load on the 80B can take a few minutes the first run (downloads
# weights into the bind mount, then loads into VRAM). Wait up to 10 min.
echo -n "==> waiting for /v1/models "
for _ in $(seq 1 600); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "==> ready on :${PORT}"
    curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool || true
    exit 0
  fi
  echo -n "."
  sleep 1
done

echo
echo "vLLM didn't come up in 10 min — check container logs:"
echo "  docker logs --tail 200 $CONTAINER"
exit 1
