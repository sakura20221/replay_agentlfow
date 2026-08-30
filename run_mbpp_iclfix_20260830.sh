#!/usr/bin/env bash
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNS=runs_mbpp_iclfix_20260830
TAG=runs_mbpp_iclfix_20260830
METHODS=(maas daao aflow flowbank gdesigner gdesigner_authordefault card card_authordefault)

cd "$ROOT" || exit 1

echo "[$(date -u '+%F %T')] formal MBPP sweep starting"
env SWEEP_RUNS="$RUNS" SWEEP_TAG="$TAG" \
    envs/maas/bin/python sweep.py \
    --methods "${METHODS[@]}" \
    --datasets mbpp --repeats 1 --jobs 8 --timeout 43200
sweep_rc=$?
echo "[$(date -u '+%F %T')] sweep finished rc=$sweep_rc"

flowbank_rc=0
if [[ -f "$RUNS/flowbank/mbpp/repeat1/status" ]] && \
   [[ "$(<"$RUNS/flowbank/mbpp/repeat1/status")" == "ok" ]]; then
    echo "[$(date -u '+%F %T')] FlowBank stages 2-3 starting"
    env SWEEP_RUNS="$RUNS" SWEEP_TAG="$TAG" \
        envs/maas/bin/python flowbank_pipeline.py \
        --dataset SHARED_MBPP --max-k 6 --run-tag "$TAG" --repeat 1 \
        --runs-dir "$RUNS" --optimized-path workspace
    flowbank_rc=$?
    echo "[$(date -u '+%F %T')] FlowBank stages 2-3 finished rc=$flowbank_rc"
else
    flowbank_rc=1
    echo "[$(date -u '+%F %T')] FlowBank stage 1 unavailable; stages 2-3 not started"
fi

echo "[$(date -u '+%F %T')] collecting current results"
env SWEEP_RUNS="$RUNS" SWEEP_TAG="$TAG" envs/maas/bin/python collect.py \
    --json "$RUNS/summary_table.json"
collect_rc=$?
echo "[$(date -u '+%F %T')] formal MBPP chain done: sweep=$sweep_rc flowbank=$flowbank_rc collect=$collect_rc"

if (( sweep_rc != 0 || flowbank_rc != 0 || collect_rc != 0 )); then
    exit 1
fi
