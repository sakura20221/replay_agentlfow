#!/bin/bash
# Launch one vLLM instance on a shared GPU with a hard self-imposed memory budget.
#
# Usage: launch_vllm.sh <gpu_index> <port> [budget_mib]
#
# On a shared GPU, --gpu-memory-utilization is a fraction of TOTAL memory, not of
# what is free. Copying a 0.9x value from a single-tenant machine would try to
# claim ~72GB of an 80GB card and take down whoever else is on it. This script
# reads the current occupancy and derives the fraction that corresponds to our
# own budget, then refuses to start if that budget is not actually available.

set -euo pipefail

GPU="${1:?usage: launch_vllm.sh <gpu_index> <port> [budget_mib]}"
PORT="${2:?usage: launch_vllm.sh <gpu_index> <port> [budget_mib]}"
BUDGET_MIB="${3:-32768}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="Qwen/Qwen3-8B"
# These must match what is actually serving, and what the proxy assumes.
#
# They were left at 16384/32/8192 while the live instances ran 40960/64/16384, so
# anyone restarting from this script would have silently changed the protocol
# mid-experiment. The proxy exposes a conservative 32768-token protocol window;
# the backend has additional headroom, but must never be smaller than the proxy.
#
# 64 sequences, not more: measured on 2026-08-23 the two instances were pinned at
# 64 running with vLLM reporting Waiting 0, KV cache at ~35% and zero preemptions,
# while both GPUs sat at 100% utilisation shared with eight other users. Raising it
# would mostly redistribute SM time away from them, and restarting an instance to
# find out risks losing it: GPU 1 had 12 GB free at the time, less than our own
# 32 GB budget needs, so a neighbour claiming that window during a restart leaves
# the sweep at half capacity for hours.
MAX_MODEL_LEN=40960
MAX_NUM_SEQS=64
MAX_BATCHED_TOKENS=16384
HEADROOM_MIB=4096   # never plan to leave a neighbour less than this

export HF_HOME="$ROOT/hf_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1          # weights are already local; never phone home mid-run
export VLLM_LOGGING_LEVEL=INFO

read -r TOTAL USED FREE < <(
  nvidia-smi --id="$GPU" --query-gpu=memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits | tr -d ',' | awk '{print $1, $2, $3}'
)

echo "GPU $GPU: total=${TOTAL}MiB used_by_others=${USED}MiB free=${FREE}MiB budget=${BUDGET_MIB}MiB"

if (( FREE < BUDGET_MIB + HEADROOM_MIB )); then
  echo "REFUSING: only ${FREE}MiB free, need ${BUDGET_MIB}+${HEADROOM_MIB}MiB." >&2
  echo "Another user's job likely grew. Pick a different GPU or lower the budget." >&2
  exit 3
fi

# Fraction of the whole card vLLM may claim -- OUR BUDGET ONLY.
#
# It used to be (already_used + our_budget)/total, on the assumption that vLLM
# counts a neighbour's resident memory towards the fraction. This vLLM version
# does the opposite: it requires fraction x total to be FREE at startup, so with a
# neighbour holding 27.6 GB the derived 0.714 asked for 56.6 GB when only 53.5 GB
# was free, and the engine refused to start at all. Deriving from the budget alone
# is both correct for this version and safer for the neighbour: it never asks for
# more than we intend to use.
UTIL=$(awk -v b="$BUDGET_MIB" -v t="$TOTAL" 'BEGIN{printf "%.4f", b/t}')
echo "derived --gpu-memory-utilization=$UTIL"

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/vllm_gpu${GPU}_port${PORT}.log"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/envs/vllm/bin/vllm" serve "$MODEL" \
  --served-model-name "$MODEL" qwen3-8b \
  --dtype bfloat16 \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --host 127.0.0.1 \
  --port "$PORT" \
  >> "$LOG" 2>&1
