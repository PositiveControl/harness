#!/usr/bin/env bash
# Mac-side wrapper: launch `harness chat` pointed at the GX10's vLLM server.
#
# Usage:
#   ./harness_chat_gx10.sh                # plain chat (persona + tools)
#   ./harness_chat_gx10.sh --no-tools     # any extra args pass through
#   ./harness_chat_gx10.sh --tui          # TUI mode
#
# Config (env vars):
#   GX10_HOST        hostname or IP of the GX10 (default: gx10)
#                    typically a Tailscale MagicDNS name like "gx10.tail-xxxx.ts.net"
#   GX10_PORT        vLLM port on the GX10 (default: 8000)
#   GX10_HEALTH_TIMEOUT  seconds to wait for /v1/models (default: 5)
#
# The script:
#   1. Pre-flights the vLLM endpoint so you get an actionable error
#      instead of a stack trace mid-turn.
#   2. Reports which model the server has loaded (so a stale swap is
#      obvious before you start talking to the wrong brain).
#   3. Defaults to --persona --tools, but you can override by passing
#      conflicting flags — they take precedence.
set -euo pipefail

HOST="${GX10_HOST:-gx10}"
PORT="${GX10_PORT:-8000}"
HEALTH_TIMEOUT="${GX10_HEALTH_TIMEOUT:-5}"
BASE_URL="http://${HOST}:${PORT}/v1"

# ---- pre-flight: is vLLM reachable? ----
echo "==> probing ${BASE_URL}/models"
if ! resp=$(curl -sf --max-time "$HEALTH_TIMEOUT" "${BASE_URL}/models" 2>&1); then
  cat >&2 <<EOF
==> vLLM not reachable at ${BASE_URL}

possible fixes:
  - is the GX10 up + on Tailscale?    ping ${HOST}
  - is vLLM running on the GX10?      ssh ${HOST} '~/vllm-scripts/vllm_swap.sh status'
  - want to start it?                 ssh ${HOST} '~/vllm-scripts/vllm_swap.sh 30b'
  - or point elsewhere:               GX10_HOST=otherhost $0

curl error:
${resp}
EOF
  exit 1
fi

# ---- report what's loaded ----
loaded_model=$(printf '%s' "$resp" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    items = data.get("data") or []
    print(items[0]["id"] if items else "(no model)")
except Exception as e:
    print(f"(parse error: {e})")
')
echo "==> vLLM up — loaded: ${loaded_model}"

# ---- find the harness repo (script lives in <repo>/scripts/gx10/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- launch ----
cd "$REPO_ROOT"
echo "==> harness chat --model vllm --model-repo ${BASE_URL} --persona --tools $*"
exec uv run harness chat \
  --model vllm \
  --model-repo "${BASE_URL}" \
  --persona \
  --tools \
  "$@"
