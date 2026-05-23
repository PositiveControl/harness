#!/usr/bin/env bash
# Round 2 of harness-u70v diagnosis: vllm is running inside docker on
# gx10, so we have to probe through `docker exec` / `docker inspect`.
# Read-only; no restart, no config change.
#
# Usage:
#   scp scripts/gx10_diagnose_v2.sh gx10-5fb9:/tmp/
#   ssh gx10-5fb9 'bash /tmp/gx10_diagnose_v2.sh'

set -u

# Locate the running vllm container — match by image, fall back to name.
CID=$(docker ps --filter "ancestor=vllm/vllm-openai:latest" --format '{{.ID}}' | head -1)
if [ -z "${CID:-}" ]; then
    CID=$(docker ps --filter "name=vllm" --format '{{.ID}}' | head -1)
fi
if [ -z "${CID:-}" ]; then
    echo "ERROR: no running vllm container found"
    docker ps
    exit 1
fi
echo "=== container ==="
docker ps --filter "id=$CID" --format 'id={{.ID}}  image={{.Image}}  status={{.Status}}  ports={{.Ports}}'
echo

echo "=== full launch args (docker inspect) ==="
docker inspect "$CID" --format '{{.Path}} {{range .Args}}{{.}} {{end}}'
echo
echo "--- entrypoint + cmd ---"
docker inspect "$CID" --format 'Entrypoint: {{json .Config.Entrypoint}}'
docker inspect "$CID" --format 'Cmd:        {{json .Config.Cmd}}'
docker inspect "$CID" --format 'Image:      {{.Config.Image}}'
echo

echo "=== image labels (vllm version + build date) ==="
docker inspect "$CID" --format 'CreatedAt:  {{.Created}}'
IMG=$(docker inspect "$CID" --format '{{.Config.Image}}')
docker inspect "$IMG" --format 'Image RepoDigests: {{json .RepoDigests}}'
docker inspect "$IMG" --format 'Image Created:     {{.Created}}'
docker inspect "$IMG" --format 'Image Labels:      {{json .Config.Labels}}'
echo

echo "=== vllm + torch + kernel libs (inside container) ==="
docker exec "$CID" python -c "
import importlib.metadata as md
for pkg in ('vllm', 'torch', 'flash_attn', 'flashinfer', 'xformers', 'triton', 'autoawq', 'vllm-flash-attn'):
    try:
        v = md.version(pkg)
        print(f'  {pkg:20s} {v}')
    except md.PackageNotFoundError:
        print(f'  {pkg:20s} (not installed)')
print()
import torch
print(f'torch={torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  device[{i}]: {p.name}  cap=({p.major},{p.minor})  mem={p.total_memory/1e9:.1f}GB')
" 2>&1
echo

echo "=== vllm engine: which AWQ + attention backend was selected ==="
# vLLM logs the chosen kernel + backend during model load. Grep the
# startup chunk of container logs for the lines that name them.
docker logs "$CID" 2>&1 | grep -iE \
    "quantization|awq|marlin|attention backend|attn impl|flashinfer|flash-attn|flash attn|FlashAttention|FlashInfer|SDPA|xformers|enforce.eager|cuda graph|capture|compilation|MoE|Loading model|GPU memory" \
    | head -50
echo

echo "=== first 120 lines of container logs (engine init) ==="
docker logs "$CID" 2>&1 | head -120
echo

echo "=== nvidia-smi (during another probe — kick a request in parallel for live util) ==="
nvidia-smi --query-gpu=name,driver_version,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu --format=csv 2>&1
echo

echo "=== done ==="
