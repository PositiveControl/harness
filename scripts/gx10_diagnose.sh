#!/usr/bin/env bash
# Run this on gx10-5fb9 (NOT on the Mac). Prints everything needed to
# diagnose why per-request generation pins at ~12 tok/s on a single GPU.
#
# Usage:
#   scp scripts/gx10_diagnose.sh gx10-5fb9:/tmp/
#   ssh gx10-5fb9 'bash /tmp/gx10_diagnose.sh'
#
# Safe — read-only. No restarts, no config changes.

set -u

echo "=== host ==="
uname -a
echo

echo "=== nvidia-smi (one snapshot) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,power.limit --format=csv 2>&1
echo

echo "=== nvidia-smi processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
echo

echo "=== vllm process command line ==="
pgrep -af "vllm serve\|vllm.entrypoints\|python.*vllm" | head -5
echo

echo "=== vllm process: full /proc/<pid>/cmdline ==="
for pid in $(pgrep -f "vllm serve\|vllm.entrypoints"); do
    echo "--- pid=$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null
    echo
done
echo

echo "=== systemd unit (if any) ==="
systemctl --no-pager status vllm 2>&1 | head -20 || true
echo

echo "=== docker (if running in container) ==="
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.Status}}' 2>&1 | head -10 || true
echo

echo "=== python / vllm / torch / flash-attn versions ==="
# Find the vllm process's venv and probe it. Falls back to system python.
VLLM_PID=$(pgrep -f "vllm serve" | head -1)
if [ -n "${VLLM_PID:-}" ] && [ -r "/proc/$VLLM_PID/exe" ]; then
    PY=$(readlink -f "/proc/$VLLM_PID/exe")
    echo "using python: $PY"
else
    PY=$(command -v python3 || command -v python)
    echo "using python (fallback): $PY"
fi
"$PY" - <<'PYEOF' 2>&1
import importlib, importlib.metadata as md
for pkg in ("vllm", "torch", "flash_attn", "flashinfer", "xformers", "triton", "autoawq"):
    try:
        v = md.version(pkg)
        print(f"  {pkg:14s} {v}")
    except md.PackageNotFoundError:
        print(f"  {pkg:14s} (not installed)")

print()
print("torch cuda info:")
try:
    import torch
    print(f"  torch.__version__       = {torch.__version__}")
    print(f"  torch.version.cuda      = {torch.version.cuda}")
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device count            = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  device[{i}]               = {p.name}  cap=({p.major}.{p.minor})  mem={p.total_memory/1e9:.1f}GB")
except Exception as exc:
    print(f"  torch probe failed: {exc}")
PYEOF
echo

echo "=== recent vllm log (last 80 lines) ==="
# Try common log locations. Adjust if your setup uses something different.
for path in /var/log/vllm.log /var/log/vllm/server.log ~/vllm.log ~/.vllm/server.log; do
    if [ -r "$path" ]; then
        echo "--- $path"
        tail -80 "$path"
        echo
        break
    fi
done
# If running via journalctl/systemd:
journalctl -u vllm --no-pager -n 80 2>&1 | head -100 || true
# If running via docker:
if [ -n "$(docker ps -q --filter 'name=vllm' 2>/dev/null)" ]; then
    echo "--- docker logs vllm (last 80)"
    docker logs --tail 80 $(docker ps -q --filter 'name=vllm') 2>&1 | head -100 || true
fi
echo

echo "=== done ==="
