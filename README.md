# Agent Workflow Bake-off

This repository contains the reproducibility layer for comparing AFlow, MaAS,
DAAO, G-Designer, CARD, FlowBank, and MasRouter on shared datasets and one local
Qwen3-8B serving protocol.

It is a source repository, not an experiment dump. The adapted author
repositories are vendored under `third_party/` and marked with their exact
commits from `upstreams.lock.json`. Model weights, virtual environments, caches,
logs, checkpoints, and result trees are intentionally excluded.

## Start here

1. Read `MIGRATION.md` for a new-server restore.
2. Read `REPRODUCTION_CHANGES.md` for method and protocol modifications.
3. Run `python3 scripts/bootstrap_upstreams.py --check` to verify the vendored
   author code. The script only downloads a pinned upstream when a vendored
   directory is absent.
4. Run `pipeline.py` so static checks and the complete 45-cell smoke gate pass
   before any full experiment.

The committed `shared/data/` files are the frozen search/evaluation splits.
Their provenance and hashes are recorded in `shared/data/manifest.json`.
Every job also records source, data, sampling, and identity fingerprints;
collection refuses stale or mixed-protocol artifacts.

This snapshot is suitable for a private Git repository. Public redistribution
requires a separate license review of the included benchmark rows, especially
the AMC split whose original provenance is not documented by its upstream repo.
