#!/usr/bin/env bash
set -euo pipefail
# Driver 570.124.06 = CUDA 12.8, so every component must be a CUDA 12 build.
# PyPI torch 2.11.0 is cu130, which makes vllm 0.20-0.26 link libcudart.so.13.
# vllm 0.19.1 pins torch 2.10.0, the newest torch whose default wheel is CUDA 12.
#
# The system python3.10 has no Python.h (python3-dev is not installed and we are
# not root), which makes Triton's runtime JIT compile fail. So the env is built
# on a uv-managed standalone CPython that ships its own headers.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_HTTP_TIMEOUT=180
export UV_CONCURRENT_DOWNLOADS=16
export UV_CACHE_DIR="$ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv_python"
if [[ -e envs/vllm ]]; then
  echo "REFUSING: envs/vllm already exists; move it aside before recreating it." >&2
  exit 2
fi
envs/tools/bin/uv python install 3.12 >> logs/uv_vllm.log 2>&1
envs/tools/bin/uv venv --python 3.12 envs/vllm >> logs/uv_vllm.log 2>&1
envs/tools/bin/uv pip install --python envs/vllm/bin/python "vllm==0.19.1" >> logs/uv_vllm.log 2>&1
echo UV_DONE >> logs/uv_vllm.log
