#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs shared/models
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
envs/tools/bin/hf download Qwen/Qwen3-8B >> logs/hf_dl.log 2>&1
envs/tools/bin/hf download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir shared/models/all-MiniLM-L6-v2 >> logs/hf_dl.log 2>&1
echo DL_DONE >> logs/hf_dl.log
