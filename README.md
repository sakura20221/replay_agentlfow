# Agent Workflow Bake-off

This repository contains the reproducibility layer for comparing AFlow, MaAS,
DAAO, G-Designer, CARD, FlowBank, and MasRouter on shared datasets and one local
Qwen3-8B serving protocol.

It is a source repository, not an experiment dump. Author repositories are
checked out at exact commits from `upstreams.lock.json` and transformed by the
reviewable installers under `shims/`. Model weights, virtual environments,
caches, logs, checkpoints, and result trees are intentionally excluded.

## Start here

1. Read `MIGRATION.md` for a new-server restore.
2. Read `REPRODUCTION_CHANGES.md` for method and protocol modifications.
3. Run `python3 scripts/bootstrap_upstreams.py` to clone and patch author code.
4. Run the static and smoke gates before any full experiment.

The committed `shared/data/` files are the frozen search/evaluation splits.
Their provenance and hashes are recorded in `shared/data/manifest.json`.

This snapshot is suitable for a private Git repository. Public redistribution
requires a separate license review of the included benchmark rows, especially
the AMC split whose original provenance is not documented by its upstream repo.
