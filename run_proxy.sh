#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs
export PROXY_LOG_PATH="$ROOT/logs/api_calls.jsonl"
export VLLM_UPSTREAMS="http://127.0.0.1:8001/v1/chat/completions,http://127.0.0.1:8002/v1/chat/completions"
export UPSTREAM_MODEL="Qwen/Qwen3-8B"
export PROXY_PER_UPSTREAM_CONCURRENCY=64
export PROXY_MAX_MODEL_LEN=32768
exec envs/tools/bin/python vllm_proxy.py >> logs/proxy.log 2>&1
