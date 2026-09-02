# New-server migration

## What Git contains

- orchestration, proxy, scorer, collectors, and audits;
- all five shim installers and their source templates;
- frozen dataset splits and provenance manifest;
- complete adapted author-repository source under `third_party/`, with exact
  source commits recorded in `upstreams.lock.json` and marker files;
- the current reproduction protocol and modification boundaries.

Git does not contain generated results, transcripts, checkpoints, model weights,
Python environments, package caches, or nested upstream Git histories.

## Restore order

```bash
git clone <private-repository-url> agent_wf
cd agent_wf
python3 scripts/bootstrap_upstreams.py --check
```

The adapted author source is already present after `git clone`. The bootstrap
command verifies each vendored source marker, applies the reviewable installers,
and runs their assertions. If a vendored directory is absent, it fetches the
exact commit from `upstreams.lock.json` and reconstructs it first.

Create the required Python environments next. The experiment uses separate
environments named `tools`, `maas`, `gdesigner`, `pyg`, and `vllm` under
`envs/`; do not copy the old environments between servers. Recreate them for the
new server's Python/CUDA/driver combination. The serving environment used
`vllm==0.19.1` with `torch==2.10.0` on the original CUDA 12.8 host.
The key versions captured from the working host are in `environments.lock.json`.

The repository does not assume a system Python package manager. On a Linux host,
install `uv` first, then create the five environments with the Python versions in
the lock file. Install the root tooling requirements into `tools`, the author
requirements into the method environments, and run the two existing setup scripts
for the serving and MaAS environments. A practical order is:

```bash
uv venv --python 3.10 envs/tools
uv pip install --python envs/tools/bin/python \
  datasets==5.0.1 numpy==2.2.6 pandas==2.3.3 aiohttp==3.14.3 \
  huggingface_hub
uv venv --python 3.12 envs/maas
uv pip install --python envs/maas/bin/python -r third_party/maas/requirements.txt
uv pip install --python envs/maas/bin/python -r third_party/aflow/requirements.txt
uv pip install --python envs/maas/bin/python -r third_party/flowbank/DiverseFlow/requirements.txt
uv venv --python 3.12 envs/gdesigner
uv pip install --python envs/gdesigner/bin/python -r third_party/gdesigner/requirements.txt
uv pip install --python envs/gdesigner/bin/python torch-geometric
uv venv --python 3.12 envs/pyg
uv pip install --python envs/pyg/bin/python \
  torch transformers sentence-transformers openai sympy \
  scikit-learn pandas pyyaml python-dotenv requests aiohttp loguru
./install_maas_env.sh
./install_vllm.sh
```

The exact Torch and PyG wheels must match the new host's CUDA/driver. The lock
file records the versions used for this snapshot; verify them with `--check`
before a run instead of assuming a CPU wheel is interchangeable with a CUDA
wheel. MasRouter has no separate requirements file; the explicit list above is
the dependency set used by its shared runner. The author requirement files are
retained for reference, but their original CUDA pins are not the experiment
environment pins and should not silently override the versions in the lock file.

Download these untracked models:

- `Qwen/Qwen3-8B` into the Hugging Face cache used by `launch_vllm.sh`;
- `sentence-transformers/all-MiniLM-L6-v2` into
  `shared/models/all-MiniLM-L6-v2`.

Then start two vLLM instances and the proxy. The backend servers have a 40,960
token allocation, while the frozen experiment protocol deliberately exposes a
32,768-token window through the proxy. This matches the protocol fingerprints
written by the latest smoke run; changing it creates a new protocol and requires
a new result tree.

Before a full run:

```bash
envs/maas/bin/python make_train_then_eval.py --check
python3 scripts/verify_data.py
envs/maas/bin/python preflight.py
```

Run a small smoke matrix and inspect saved transcripts before spending on the
full matrix. Never reuse an old `runs_*` directory under a changed prompt,
scorer, sampling setting, shim, or dataset split.

## Data and result backup

The frozen matrix has 35 primary cells (7 methods x 5 datasets), plus 10
explicit G-Designer/CARD author-default control cells. Search/evaluation counts
are MATH 119/486, AMC 165/648, MBPP 256/500, DROP 256/1000, and MMLU-Pro
252/1120. The committed data files are enough to recreate the exact task inputs.
Old `logs/`, `runs_*`, `archive/`, and checkpoints are evidence rather than source;
copy them to object storage or a separate archival disk if they must be retained.
Do not add them to this Git history.

## Security and publication

On Linux/A800, all five gold-answer replay gates must pass, including MBPP
1000/1000 wrapper checks. macOS can fail four MBPP checks because two reference
tasks differ under the platform `libm`; do not loosen the Linux gate.

No A800 password or remote credential belongs in this repository. Local API keys
with the literal value `local` are non-secret placeholders for loopback vLLM.
Use a private remote unless the benchmark redistribution licenses have been
reviewed.
