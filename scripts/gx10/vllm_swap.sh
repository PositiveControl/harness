#!/usr/bin/env bash
# Start / stop / swap the vLLM server on the GX10.
#
# Usage:
#   ./vllm_swap.sh 30b     # serve Qwen3-Coder-30B-A3B-Instruct-FP8
#   ./vllm_swap.sh next    # serve Qwen3-Coder-Next-FP8 (80B MoE)
#   ./vllm_swap.sh stop    # stop the current server, leave port free
#   ./vllm_swap.sh status  # show whether the server is up and what it loaded
#
# Reads:
#   VLLM_PORT     (default 8000)
#   VLLM_LOG_DIR  (default ~/vllm-logs)
#   VLLM_GPU_UTIL (default 0.92)
#
# Notes:
#   - Qwen3-Coder ships a correct chat template baked into the repo, so we
#     do NOT pass --chat-template. (Qwen 2.5 Coder needed an override; Qwen3
#     does not.)
#   - --tool-call-parser hermes works for Qwen3's tool-call XML format.
#   - --enable-prefix-caching helps a lot with the harness's conversational
#     workload (large repeated system prompt + retrieval blocks).
set -euo pipefail

PORT="${VLLM_PORT:-8000}"
LOG_DIR="${VLLM_LOG_DIR:-$HOME/vllm-logs}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"
mkdir -p "$LOG_DIR"

case "${1:-}" in
  30b)
    MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    MAX_LEN=131072
    ;;
  next)
    MODEL="Qwen/Qwen3-Coder-Next-FP8"
    MAX_LEN=131072
    ;;
  stop)
    MODEL=""
    ;;
  status)
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM up on :${PORT}"
      curl -s "http://localhost:${PORT}/v1/models" | python3 -m json.tool
    else
      echo "vLLM not responding on :${PORT}"
      pgrep -af "vllm" || true
    fi
    exit 0
    ;;
  *)
    echo "usage: $0 {30b|next|stop|status}" >&2
    exit 1
    ;;
esac

stop_server() {
  local pids
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server\|^vllm serve\| vllm serve" || true)
  if [[ -z "$pids" ]]; then
    echo "    (no vLLM process found)"
    return 0
  fi
  echo "    sending SIGTERM to: $pids"
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      sleep 1
      if ! pgrep -f "vllm" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "    didn't exit cleanly in 30s; SIGKILL"
  pkill -9 -f "vllm" 2>/dev/null || true
  sleep 2
}

echo "==> stopping any running vLLM"
stop_server

if [[ -z "$MODEL" ]]; then
  echo "==> stopped."
  exit 0
fi

LOG="$LOG_DIR/vllm-$(echo "$MODEL" | tr '/' '_')-$(date +%Y%m%d-%H%M%S).log"
echo "==> starting vLLM"
echo "    model:   $MODEL"
echo "    port:    $PORT"
echo "    ctx:     $MAX_LEN"
echo "    gpu_util: $GPU_UTIL"
echo "    log:     $LOG"

nohup vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  --gpu-memory-utilization "$GPU_UTIL" \
  > "$LOG" 2>&1 &

# Wait up to 5 minutes for the OpenAI /v1/models endpoint.
# Cold load of an 80B-FP8 model from disk can take a few minutes the first run.
echo -n "==> waiting for /v1/models "
for i in $(seq 1 300); do
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
echo "vLLM didn't come up in 5 min — tail the log:"
echo "  tail -f $LOG"
exit 1
