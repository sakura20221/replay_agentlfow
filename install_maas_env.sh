#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_CACHE_DIR="$ROOT/.uv_cache"
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=16
envs/tools/bin/uv pip install --python envs/maas/bin/python -q "torch==2.10.0" >> logs/maas_env.log 2>&1
echo "TORCH_OK" >> logs/maas_env.log
envs/tools/bin/python shims/resolve_imports.py \
  --python envs/maas/bin/python --cwd third_party/maas \
  --module maas.ext.maas.benchmark.shared_shim >> logs/maas_env.log 2>&1
echo "RESOLVE_OK" >> logs/maas_env.log
echo MAAS_ENV_DONE >> logs/maas_env.log
