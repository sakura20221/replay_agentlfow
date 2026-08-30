#!/usr/bin/env python3
"""Move every artefact produced against the old MATH-500 split out of the live tree.

2026-08-24: math.jsonl / math_search.jsonl were replaced by the AFlow/FlowBank
official Level-5 split (119 validate + 486 test). Scores produced against the old
file are not comparable with scores produced against the new one, so nothing that
was computed from MATH-500 may stay where collect.py or a resumed sweep could read
it. This does what separate_runs.py must NOT be used for here: separate_runs moves
*every* dataset's workspaces, and the drop/mmlu_pro artefacts still in them are
exactly what the finishers and the final collect need.

Only math-scoped paths move; everything else is untouched:

  runs_v5/<method>/math/                              job dirs (killed + old-set ok)
  third_party/{maas,daao}/.../optimized/SHARED_MATH   per-item CSVs + searched graphs
  third_party/aflow/workspace/SHARED_MATH             rounds, results.json
  third_party/flowbank/DiverseFlow/workspace/SHARED_MATH
  third_party/{gdesigner,card}/result/*/*_math_r*.json
  third_party/masrouter/logs/math_*.txt, math_router_epoch*.pth

Each invocation moves into its own timestamped subdir of archive/math500_products/
and writes a manifest, so it can be re-run later to sweep smoke artefacts out of
the same paths before the real math jobs start. Refuses to run while any live
process is working on a math cell. Installers must be re-run afterwards: archiving
a workspace carries away its seed template (learned in v4).

    envs/tools/bin/python archive_math500_products.py          # dry run
    envs/tools/bin/python archive_math500_products.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST_BASE = ROOT / "archive" / "math500_products"
METHODS = ("maas", "daao", "gdesigner", "card", "gdesigner_authordefault",
           "card_authordefault", "masrouter", "aflow", "flowbank")

# Anything a running math job could have on its command line. SHARED_MATH covers
# the maas family and aflow/flowbank; the --*dataset forms cover the gdesigner
# family and masrouter, which take the plain dataset name.
LIVE_PATTERN = re.compile(
    r"SHARED_MATH|--shared_dataset math\b|--dataset math\b|--datasets math\b")


def live_math_processes() -> list[str]:
    hits = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        except OSError:
            continue
        if str(ROOT / "envs") in argv and LIVE_PATTERN.search(argv):
            hits.append(f"{entry.name}: {argv[:140]}")
    return hits


def targets(runs: Path) -> list[Path]:
    fixed = [runs / m / "math" for m in METHODS]
    fixed += [ROOT / "third_party/maas/maas/ext/maas/scripts/optimized/SHARED_MATH",
              ROOT / "third_party/daao/daao/ext/maas/scripts/optimized/SHARED_MATH",
              ROOT / "third_party/aflow/workspace/SHARED_MATH",
              ROOT / "third_party/flowbank/DiverseFlow/workspace/SHARED_MATH"]
    globs = [*ROOT.glob("third_party/gdesigner/result/*/*_math_r*.json"),
             *ROOT.glob("third_party/card/result/*/*_math_r*.json"),
             *ROOT.glob("third_party/masrouter/logs/math_*.txt"),
             *ROOT.glob("third_party/masrouter/math_router_epoch*.pth")]
    return [p for p in fixed + sorted(globs) if p.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs_v5")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    live = live_math_processes()
    if live:
        print("refusing: live math process(es):")
        for line in live:
            print("  " + line)
        raise SystemExit(1)

    found = targets(ROOT / args.runs)
    if not found:
        print("nothing to move: no math-scoped artefacts in the live tree")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = DEST_BASE / stamp
    moved = []
    for path in found:
        rel = path.relative_to(ROOT)
        print(f"  {'move' if args.apply else 'would move'}  {rel}")
        if args.apply:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            moved.append(str(rel))
    if args.apply:
        (dest / "MANIFEST.json").write_text(json.dumps(
            {"when": stamp, "why": "MATH-500 -> official L5 split; scores not comparable",
             "moved": moved}, ensure_ascii=False, indent=2))
        print(f"moved {len(moved)} path(s) -> {dest.relative_to(ROOT)}")
        print("now re-run the installers: archiving carried away the seed templates")
    else:
        print(f"dry run: {len(found)} path(s). Re-run with --apply.")


if __name__ == "__main__":
    main()
